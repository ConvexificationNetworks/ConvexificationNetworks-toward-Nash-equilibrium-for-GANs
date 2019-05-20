import os
import numpy as np
from PIL import Image
import gc
import chainer
import chainer.cuda
from chainer import Variable
from chainer import serializers
import math
import chainer.functions as F
import scipy.linalg

#sys.path.append(os.path.dirname(__file__))
from inception_score import inception_score, Inception


def out_generated_image(gen, rows, cols, seed, dst):
    @chainer.training.make_extension(trigger=(1, 'epoch'))
    def make_image(trainer):
        np.random.seed(seed)
        n_images = rows * cols
        xp = gen.xp
        z = Variable(xp.asarray(gen.make_hidden(n_images)))

        with chainer.using_config('train', False):
            x = gen(z)
        x = chainer.cuda.to_cpu(x.data)
        np.random.seed()

        x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
        _, _, H, W = x.shape
        x = x.reshape((rows, cols, 3, H, W))
        x = x.transpose(0, 3, 1, 4, 2)
        x = x.reshape((rows * H, cols * W, 3))

        preview_dir = '{}/preview'.format(dst)
        preview_path = preview_dir + '/image_epoch_{:0>4}.png'.format(trainer.updater.epoch)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        Image.fromarray(x).save(preview_path)
    return make_image


def load_inception_model():
    infile = "%s/inception_score.model"%os.path.dirname(__file__)
    model = Inception()
    serializers.load_hdf5(infile, model)
    model.to_gpu()
    return model


def calc_inception(gen, batchsize=100):
    @chainer.training.make_extension()
    def evaluation(trainer):
        model = load_inception_model()

        ims = []
        xp = gen.xp

        n_ims = 1000
        for i in range(0, n_ims, batchsize):
            z = Variable(xp.asarray(gen.make_hidden(batchsize)))
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                x = gen(z)
            x = chainer.cuda.to_cpu(x.data)
            x = np.asarray(np.clip(x * 255.0, 0.0, 255.0), dtype=np.uint8)
            ims.append(x)
        ims = np.asarray(ims)
        _, _, _, h, w = ims.shape
        ims = ims.reshape((n_ims, 1, h, w)).astype("f")
        ims = np.tile(ims, (1, 3, 1, 1))

        mean, std = inception_score(model, ims)

        chainer.reporter.report({
            'inception_mean': mean,
            'inception_std': std
        })

    return evaluation


def get_mean_cov(model, ims, batch_size=100):
    n, c, w, h = ims.shape
    n_batches = int(math.ceil(float(n) / float(batch_size)))

    xp = model.xp

    print('Batch size:', batch_size)
    print('Total number of images:', n)
    print('Total number of batches:', n_batches)

    ys = xp.empty((n, 2048), dtype=xp.float32)

    for i in range(n_batches):
        print('Running batch', i + 1, '/', n_batches, '...')
        batch_start = (i * batch_size)
        batch_end = min((i + 1) * batch_size, n)

        ims_batch = ims[batch_start:batch_end]
        ims_batch = xp.asarray(ims_batch)  # To GPU if using CuPy
        ims_batch = Variable(ims_batch)

        # Resize image to the shape expected by the inception module
        if (w, h) != (299, 299):
            ims_batch = F.resize_images(ims_batch, (299, 299))  # bilinear

        # Feed images to the inception module to get the features
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            y = model(ims_batch, get_feature=True)
        
        ys[batch_start:batch_end] = y.data

    mean = chainer.cuda.to_cpu(xp.mean(ys, axis=0))
    cov = np.cov(chainer.cuda.to_cpu(ys).T)
    
    del ys
    gc.collect()
    
    return mean, cov


def FID(m0,c0,m1,c1):
    ret = 0
    ret += np.sum((m0-m1)**2)
    ret += np.trace(c0 + c1 - 2.0*scipy.linalg.sqrtm(np.dot(c0, c1)))
    return np.real(ret)

def calc_FID(gen, batchsize=100, stat_file="%s/cifar-10-fid.npz"%os.path.dirname(__file__)):
    """Frechet Inception Distance proposed by https://arxiv.org/abs/1706.08500"""
    @chainer.training.make_extension()
    def evaluation(trainer):
        model = load_inception_model()
        stat = np.load(stat_file)

        n_ims = 1000
        xp = gen.xp
        xs = []
        for i in range(0, n_ims, batchsize):
            z = Variable(xp.asarray(gen.make_hidden(batchsize)))
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                x = gen(z)
            x = chainer.cuda.to_cpu(x.data)
            x = np.asarray(np.clip(x * 255.0, 0.0, 255.0), dtype="f")
            xs.append(x)
        xs = np.asarray(xs)
        _, _, _, h, w = xs.shape
        

        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            mean, cov = get_mean_cov(model, np.asarray(xs).reshape((-1, 3, h, w)))
        
        del xs
        gc.collect()
        
        fid = FID(stat["mean"], stat["cov"], mean, cov)

        chainer.reporter.report({
            'FID': fid,
        })

    return evaluation