# coding: UTF-8
import argparse
import os
import chainer
import numpy as np
from PIL import Image
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

from evaluate import out_generated_image, calc_FID
from chainer import variable
from inception_score import Inception
from chainer import serializers
import sys

xp = cuda.cupy


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

        xp = cuda.cupy

        batch = self.get_iterator('main').next()
        batchsize = len(batch)

        # Step1 Generate
        z = Variable(xp.asarray(self.generator.make_hidden(batchsize)))
        x_gen = self.generator(z)
        y_gen = self.critic(x_gen)

        # Step2 real
        x_real = Variable(xp.array(batch)) / 255.
        y_real = self.critic(x_real)

        # Step3 Compute loss for DCGAN
        loss_cri = F.sum(F.softplus(-y_real)) / batchsize
        loss_cri += F.sum(F.softplus(y_gen)) / batchsize
        
        loss_sp = self.lam2 * F.absolute(F.sum(self.critic.inter.W) - 1)
        loss_all = loss_cri + loss_sp

        # Step4 Update critic
        self.critic.cleargrads()
        loss_all.backward()
        self._optimizers['critic'].update()

        # Step5 Update generator
        if self.iteration < 2500 and self.iteration % 100 == 0:
            loss_gen = F.sum(F.softplus(-y_gen)) / batchsize
            loss_sp = self.lam2 * F.absolute(F.sum(self.generator.inter.W) - 1)
            loss_gen += loss_sp
            self.generator.cleargrads()
            loss_gen.backward()
            self._optimizers['generator'].update()
            chainer.reporter.report({'loss/generator': loss_gen})

        if self.iteration > 2500 and self.iteration % self.n_c == 0:
            loss_gen = F.sum(F.softplus(-y_gen)) / batchsize
            loss_sp = self.lam2 * F.absolute(F.sum(self.generator.inter.W) - 1)
            loss_gen += loss_sp
            self.generator.cleargrads()
            loss_gen.backward()
            self._optimizers['generator'].update()
            chainer.reporter.report({'loss/generator': loss_gen})

        # Step6 Report
        chainer.reporter.report({'loss/critic': loss_cri})
        

def main():
    parser = argparse.ArgumentParser(description='WGAN')
    parser.add_argument('--batchsize', '-b', type=int, default=50,
                        help='Number of images in each mini-batch')
    parser.add_argument('--lam', '-l', type=int, default=10,
                        help='Hyperparameter of gp')
    parser.add_argument('--lam2', '-l2', type=int, default=20,
                        help='Hyperparameter of gp')
    parser.add_argument('--n_hidden', type=int, default=128,
                        help='latent variable')
    parser.add_argument('--seed', '-s', type=int, default=111,
                        help='seed number')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--evaluation_interval', '-v', type=int, default=1,
                        help='Number of units')
    parser.add_argument('--sampleimage_interval', '-i', type=int, default=1,
                        help='Number of units')
    parser.add_argument('--setting', '-set', type=int, default=5,
                        help='Number of units')
    args = parser.parse_args(args=[])

    
    # parameter set
    ## Optimizers
    alpha = 0.0002
    beta1 = 0.5
    beta2 = 0.9
    dim = 32*32*3
    num_nets = 1
    weight_decay = 0.0001
    eps = 1e-08
    wscale = 0.02
    
    # Networks
    generator = DCGANGenerator(dim, num_nets, args.n_hidden, bottom_width=4, ch=512, wscale=0.02, hidden_activation=F.relu, use_bn=True)
    critic = WGANDiscriminator(num_nets, bottom_width=4, ch=512, wscale=0.02, output_dim=1)

    if args.gpu >= 0:
        
        chainer.cuda.get_device(args.gpu).use()
        generator.to_gpu()
        critic.to_gpu()

    # Optimizer set
    opt_g = chainer.optimizers.Adam(alpha = alpha, beta1 = beta1, beta2 = beta2, weight_decay_rate = weight_decay)
    opt_g.setup(generator)

    opt_c = chainer.optimizers.Adam(alpha = alpha*3, beta1 = beta1, beta2 = beta2, weight_decay_rate = weight_decay)
    opt_c.setup(critic)

    # Dataset
    train, test = chainer.datasets.get_cifar10(withlabel=False, scale=255.)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    # Trainer
    updater = WGANUpdater(train_iter, generator, critic, 5, opt_g, opt_c, args.lam, args.lam2, args.n_hidden, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    
    # Extensions

    # loss plot -- critic
    trainer.extend(extensions.PlotReport(['loss/critic'], 'epoch', file_name='DCcriticloss_net%d.png' % args.setting))
    # loss plot -- generator
    trainer.extend(extensions.PlotReport(['loss/generator'], 'epoch', file_name='DCgenloss_net%d.png' % args.setting))

    # print report
    trainer.extend(extensions.PrintReport(['epoch', 'loss/critic', 'loss/generator', 'elapsed_time']))

    # sample image
    out = "/home/ubuntu/images/DC%d" % args.setting
    trainer.extend(out_generated_image(generator, 10, 10, args.seed + 10, out), trigger=(args.sampleimage_interval, "epoch"))

    # FID
    trainer.extend(calc_FID(generator), trigger=(args.evaluation_interval, 'epoch'))

    # logging
    trainer.extend(extensions.LogReport(log_name = "DClog%d" % args.setting))

    # progress bar
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_xpz(args.resume, trainer)

    # Run
    trainer.run()