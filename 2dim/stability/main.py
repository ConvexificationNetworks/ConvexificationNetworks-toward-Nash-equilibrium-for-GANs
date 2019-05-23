# coding: UTF-8

import argparse
import os

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers
from chainer import training
from chainer import Variable
from chainer import cuda
from chainer.dataset import iterator as iterator_module
from chainer.training import extensions
from chainer.dataset import convert
from chainer import Variable
from chainer import variable
from data import GmmDataset
from utils import out_generated, calcEMD

# Updater

class WGANUpdater(training.StandardUpdater):
    def __init__(self, iterator, generator, critic,
                 n_c, opt_g, opt_c, lam, lam2, n_hidden, device):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator
        self.generator = generator
        self.critic = critic
        self.n_c = n_c
        self._optimizers = {'generator': opt_g, 'critic': opt_c}
        self.lam = lam
        self.lam2 = lam2
        self.device = device
        self.converter = convert.concat_examples
        self.iteration = 0
        self.n_hidden = n_hidden

    def update_core(self):
        batch = self.get_iterator('main').next()
        batchsize = len(batch)

        # Step1 Generate
        z = Variable(np.asarray(self.generator.make_hidden(batchsize)))
        #print(z)
        x_gen = self.generator(z)
        y_gen = self.critic(x_gen)

        # Step2 real
        x_real = Variable(np.array(batch))
        y_real = self.critic(x_real)

        # Step3 Compute loss for wgan_gp
        eps = np.random.uniform(0, 1, (batchsize, 1)).astype("f")
        x_mid = eps * x_real + (1.0 - eps) * x_gen
        x_mid_v = Variable(x_mid.data)
        y_mid = self.critic(x_mid_v)
        dydx = chainer.grad([y_mid], [x_mid_v], enable_double_backprop=True)[0]
        dydx = F.sqrt(F.sum(F.square(dydx), axis=1))
        loss_gp = self.lam * F.mean_squared_error(dydx, np.ones_like(dydx.data))
        loss_cri = F.sum(-y_real) / batchsize
        loss_cri += F.sum(y_gen) / batchsize
        
        loss_sp = self.lam2 * F.absolute(F.sum(self.critic.inter.W) - 1)
        loss_all = loss_cri + loss_gp + loss_sp
        
        # Step4 Update critic
        self.critic.cleargrads()
        loss_all.backward()
        self._optimizers['critic'].update()

        # Step5 Update generator
        
        if self.iteration % self.n_c == 0:
            loss_gen = F.sum(-y_gen) / batchsize
            self.generator.cleargrads()
            loss_gen.backward()
            self._optimizers['generator'].update()
            chainer.reporter.report({'loss/generator': loss_gen})

        # Step6 Report
        chainer.reporter.report({'loss/critic': loss_cri})


def main():
    parser = argparse.ArgumentParser(description='WGAN')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--datasize', '-d', type=int, default=10000,
                        help='Number of samples')
    parser.add_argument('--lam', '-l', type=int, default=0.5,
                        help='Hyperparameter of gp')
    parser.add_argument('--lam2', '-l2', type=int, default=1.0,
                        help='Hyperparameter of gp')
    parser.add_argument('--n_hidden', type=int, default=100,
                        help='latent variable')
    parser.add_argument('--seed', '-s', type=int, default=111,
                        help='seed number')
    parser.add_argument('--epoch', '-e', type=int, default=300,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--setting', '-set', type=int, default=10,
                        help='Number of units')
    args = parser.parse_args(args=[])

    # parameter set
    ## Optimizers
    alpha = 0.002
    beta1 = 0.9
    beta2 = 0.999

    ## Network
    dim = 2
    num_nets = 1
    wscale = 0.02

    # Networks
    generator = Generator(dim, num_nets, args.n_hidden, wscale)
    critic = Critic(num_nets, wscale)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        generator.to_gpu()
        critic.to_gpu()

    # Optimizer set
    opt_g = chainer.optimizers.Adam(alpha = alpha, beta1 = beta1, beta2 = beta2, weight_decay_rate=0.0001)
    opt_g.setup(generator)
    opt_g.add_hook(chainer.optimizer.GradientClipping(1))

    opt_c = chainer.optimizers.Adam(alpha = alpha, beta1 = beta1, beta2 = beta2, weight_decay_rate=0.0001)
    opt_c.setup(critic)
    opt_c.add_hook(chainer.optimizer.GradientClipping(1))

    # Dataset
    dataset = GmmDataset(args.datasize, args.seed, std=0.02, scale=2)
    train_iter = chainer.iterators.SerialIterator(dataset, args.batchsize)

    # Trainer
    updater = WGANUpdater(train_iter, generator, critic, 5, opt_g, opt_c, args.lam, args.lam2, args.n_hidden, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Extensions
    trainer.extend(extensions.LogReport())

    trainer.extend(
        extensions.PlotReport(['loss/critic'],
                              'epoch', file_name='criticloss_2dim%d.png' % args.setting))

    trainer.extend(
        extensions.PlotReport(
            ['loss/generator'], 'epoch', file_name='genloss_2dim%d.png' % args.setting))

    trainer.extend(extensions.PrintReport(
        ['epoch', 'loss/critic', 'loss/generator', 'elapsed_time']))

    out = "/Users/keiikegami/Dropbox/research/Imaizumi-Sensei/paper_result/2dim/result%d" % args.setting

    trainer.extend(calcEMD(generator), trigger=(1, 'epoch'))
    trainer.extend(extensions.LogReport(log_name = "log_2dim%d" % args.setting))
    trainer.extend(out_generated(generator, args.seed + 10, out, radius=2), trigger=(10, 'epoch'))
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # Run
    trainer.run()


