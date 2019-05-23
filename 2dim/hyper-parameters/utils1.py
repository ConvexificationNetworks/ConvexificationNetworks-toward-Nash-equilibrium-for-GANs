import pathlib
import numpy as np
import seaborn as sns
import chainer
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import chainer.backends.cuda
from chainer import Variable
from data import GmmDataset
import ot
import gc
import pickle

sns.set()


def out_generated(gen, seed, dst, datasize=1000, **kwards):
    """
    Trainer extension that plot Generated data
    Parameters
    -------------
    gen: Model
        Generator
    seed: int
        fix random by value
    dst: PosixPath
        file path to save plotted result
    datasize: int
        the number of plotted datas
    **kwards: key, value pairings
        other keyward argument to pass to seaborn.kdeplot
    Return
    ------------
    make_image:function
        function that returns make_images that has Trainer object
        as argument.
    """
    @chainer.training.make_extension(trigger=(1, 'epoch'))
    def make_image(trainer):

        np.random.seed(seed)  # fix seed
        xp = gen.xp  # get module

        with chainer.using_config('train', False):
            with chainer.using_config('enable_backprop', False):
                z = Variable(np.random.uniform(-1, 1, (datasize, gen.n_hidden)).astype(np.float32))
                x = gen(z)

        x = chainer.backends.cuda.to_cpu(x.data)
        np.random.seed()

        preview_dir = pathlib.Path('{}/preview'.format(dst))
        preview_path = preview_dir /\
            '{:}epoch_kde.jpg'.format(trainer.updater.epoch)
        if not preview_dir.exists():
            preview_dir.mkdir()
        # norm = Normalize(vmin=x.data.min(),  vmax=x.data.max())  # colorbar range of kde
        plot_kde_data(x, trainer.updater.epoch, preview_path, shade=True, cbar=False,
                      cmap="Blues", shade_lowest=False, **kwards)
        preview_path = preview_dir /\
            '{:}epoch_scatter.jpg'.format(trainer.updater.epoch)
        plot_scatter_data(x, trainer.updater.epoch, preview_path, **kwards)

    return make_image


def plot_scatter_data(data, epoch, preview_path, **kwards):
    """
    Plot the data
    Parameters
    --------------
    data: array whose shape is (datasize, 2)
        plotted data
    epoch: int
        Epoch number
    preview_path: PosixPath
        file path to save plotted results
    **kwards: key, value pairings
        other keyward argument to pass to seaborn.kdeplot
    """
    radius = kwards.pop('radius')
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    axes.scatter(data[:, 0], data[:, 1],
                 alpha=0.5, color='darkgreen', s=17)
    axes.set_title('epoch: {:>3}'.format(epoch))
    axes.set_xlim(-radius-5.0, radius+5.0)
    axes.set_ylim(-radius-5.0, radius+5.0)
    fig.tight_layout()
    fig.savefig(str(preview_path))
    plt.close(fig)


def plot_kde_data(data, epoch, preview_path, **kwards):
    """
    Plot the data
    Parameters
    --------------
    data: array whose shape is (datasize, 2)
        plotted data
    epoch: int
        Epoch number
    preview_path: PosixPath
        file path to save plotted result
    **kwards: key, value pairings
        other keyward argument to pass to seaborn.kdeplot
    """
    radius = kwards.pop('radius')
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    axes = sns.kdeplot(data=data[:, 0], data2=data[:, 1],
                       ax=axes, **kwards)
    axes.set_title('epoch: {:>3}'.format(epoch))
    axes.set_xlim(-radius-5.0, radius+5.0)
    axes.set_ylim(-radius-5.0, radius+5.0)
    fig.tight_layout()
    #print(type(preview_path))
    fig.savefig(str(preview_path))
    plt.close(fig)


def plot_kde_data_real(data, file_path, **kwards):
    """
    Plot the data
    Parameters
    --------------
    data: array whose shape is (datasize, 2)
        plotted data
    epoch: int
        Epoch number
    file_path: PosixPath
        file path to save plotted result
    **kwards: key, value pairings
        other keyward argument to pass to seaborn.kdeplot
    """
    radius = kwards.pop('radius')
    # norm = Normalize(vmin=0,  vmax=1)  # colorbar range of kde
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    axes = sns.kdeplot(data=data[:, 0], data2=data[:, 1],
                       ax=axes, **kwards)
    axes.set_title('Real Data')
    axes.set_xlim(-radius-5.0, radius+5.0)
    axes.set_ylim(-radius-5.0, radius+5.0)
    fig.tight_layout()
    fig.savefig(file_path + '/training_data_kde.jpg')
    plt.close(fig)


def plot_scatter_real_data(data, file_path, **kwards):
    """
    Plot the data
    Parameters
    --------------
    data: array whose shape is (datasize, 2)
        plotted data
    preview_path: PosixPath
        file path to save plotted results
    **kwards: key, value pairings
        other keyward argument to pass to seaborn.kdeplot
    """
    radius = kwards.pop('radius')
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    axes.scatter(data[:, 0], data[:, 1],
                 alpha=0.5, color='darkgreen', s=17)
    axes.set_title('Real Data')
    axes.set_xlim(-radius-5.0, radius+5.0)
    axes.set_ylim(-radius-5.0, radius+5.0)
    fig.tight_layout()
    fig.savefig(file_path + '/training_data_scatter.jpg')
    plt.close(fig)





if __name__ == "__main__":
    gmm = GmmDataset(10000, 123, num_cluster=8, std=0.02, scale=2)
    file = "/Users/keiikegami/Dropbox/research/Imaizumi-Sensei/image_results/2dim"
    data = gmm._data
    plot_kde_data_real(data, file, radius = 2)
    plot_scatter_real_data(data, file, radius = 2)



def calcEMD(gen, combination):
    @chainer.training.make_extension()
    def evaluation(trainer):
        num_sample = 10000
        xp = gen.xp
        #xs = []
        """
        for i in range(0, num_sample, batchsize):
            z = Variable(xp.asarray(gen.make_hidden(batchsize)))
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                x = gen(z)
            x = chainer.cuda.to_cpu(x.data)
            xs.append(x)
        xs = np.asarray(xs)
        """
        z = Variable(xp.asarray(gen.make_hidden(num_sample)))
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x = gen(z)
        xs = chainer.cuda.to_cpu(x.data)
        real_data = GmmDataset(num_sample, 123, num_cluster=8, std=0.02, scale=2)._data
        a, b = np.ones((num_sample,)) / num_sample, np.ones((num_sample,)) / num_sample
        #print(xs)
        #print(real_data)
        M = ot.dist(xs, real_data)
        M /= M.max()
        distance = ot.emd2(a, b, M)
        
        del xs
        gc.collect()
        del real_data
        gc.collect()
        
        #path="/Users/keiikegami/Dropbox/research/Imaizumi-Sensei/paper_result/2dim/emd.pickle"
        with open("emd.txt", 'ab') as f:
            f.write("{%d: %f}\n".encode() % (combination, distance))
        #pickle.dump({combination: distance}, path)
            
    return evaluation
















