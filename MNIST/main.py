# coding: UTF-8

import argparse
import os

import numpy as np
from PIL import Image
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
from evaluate import out_generated_image, calc_FID
from chainer import variable
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
        batch = self.get_iterator('main').next()
        batchsize = len(batch)

        # Step1 GeneratorExit(" error")
        z = Variable(xp.asarray(self.generator.make_hidden(batchsize))) / 255.
        x_gen = self.generator(z)
        y_gen = self.critic(x_gen)
 
        # Step2 real
        x_real = Variable(xp.array(batch)) / 255.
        y_real = self.critic(x_real)

        
        # Step3 Compute loss for wgan_gp
        eps = xp.random.uniform(0, 1, (batchsize, 1, 1, 1)).astype("f")
        x_mid = eps * x_real + (1.0 - eps) * x_gen
        x_mid_v = Variable(x_mid.data)
        y_mid = self.critic(x_mid_v)
        dydx = chainer.grad([y_mid], [x_mid_v], enable_double_backprop=True)[0]
        dydx = F.sqrt(1e-08+F.sum(F.square(dydx), axis=1))
        loss_gp = self.lam * F.mean_squared_error(dydx, xp.ones_like(dydx.data))
        loss_cri = F.sum(-y_real) / batchsize
        loss_cri += F.sum(y_gen) / batchsize

        # extra step calculate regularization term about the last layer
        loss_sp = self.lam2 * F.absolute(F.sum(self.critic.inter.W) - 1)
        loss_all = loss_cri + loss_gp + loss_sp

        # Step4 Update critic
        self.critic.cleargrads()
        loss_all.backward(loss_scale = 0.001)
        self._optimizers['critic'].update()

        # Step5 Update generator
        if self.iteration < 2500 and self.iteration % 100 == 0:
            loss_gen = F.sum(-y_gen) / batchsize
            loss_sp = self.lam2 * F.absolute(F.sum(self.generator.inter.W) - 1)
            loss_gen += loss_sp
            self.generator.cleargrads()
            loss_gen.backward(loss_scale = 0.001)
            self._optimizers['generator'].update()
            chainer.reporter.report({'loss/generator': loss_gen})

        if self.iteration > 2500 and self.iteration % self.n_c == 0:
            loss_gen = F.sum(-y_gen) / batchsize
            loss_sp = self.lam2 * F.absolute(F.sum(self.generator.inter.W) - 1)
            loss_gen += loss_sp
            self.generator.cleargrads()
            loss_gen.backward(loss_scale = 0.001)
            self._optimizers['generator'].update()
            chainer.reporter.report({'loss/generator': loss_gen})

        # Step6 Report
        chainer.reporter.report({'loss/critic': loss_cri})

def main():
    parser = argparse.ArgumentParser(description='WGAN')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--lam', '-l', type=int, default=10,
                        help='Hyperparameter of gp')
    parser.add_argument('--lam2', '-l2', type=int, default=20,
                        help='Hyperparameter of gp')
    parser.add_argument('--n_hidden', type=int, default=100,
                        help='latent variable')
    parser.add_argument('--seed', '-s', type=int, default=111,
                        help='seed number')
    parser.add_argument('--epoch', '-e', type=int, default=200,
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
    parser.add_argument('--sampleimage_interval', '-i', type=int, default=5,
                        help='Number of units')
    parser.add_argument('--setting', '-set', type=int, default=3,
                        help='Number of units')
    args = parser.parse_args(args=[])

    # parameter set
    ## Optimizers
    alpha = 0.0001
    beta1 = 0.0
    beta2 = 0.9
    weight_decay_rate = 0.0001
    eta = 1
    eps = 1e-08

    ## Network
    dim = 784
    num_nets = 8
    wscale = 0.02

    # Networks
    generator = Generator(dim, num_nets, args.n_hidden, wscale)
    critic = Critic(num_nets, wscale)
    
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        generator.to_gpu()
        critic.to_gpu()

    # Optimizer set
    opt_g = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2, eps=eps, weight_decay_rate=weight_decay_rate, eta=eta)
    opt_g.setup(generator)
    opt_g.add_hook(chainer.optimizer_hooks.GradientHardClipping(-100, 100))

    opt_c = chainer.optimizers.Adam(alpha=alpha * 3, beta1=beta1, beta2=beta2, eps=eps, weight_decay_rate=weight_decay_rate, eta=eta)
    opt_c.setup(critic)
    opt_c.add_hook(chainer.optimizer_hooks.GradientHardClipping(-100, 100))

    # Dataset
    train, test = chainer.datasets.get_mnist(withlabel=False, ndim=3, scale=255.)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    # Trainer
    updater = WGANUpdater(train_iter, generator, critic, 5, opt_g, opt_c, args.lam, args.lam2, args.n_hidden, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Extensions

    # loss plot -- critic
    trainer.extend(extensions.PlotReport(['loss/critic'], 'epoch', file_name='criticloss_numpara%d.png' % args.setting))
    # loss plot -- generator
    trainer.extend(extensions.PlotReport(['loss/generator'], 'epoch', file_name='genloss_numpara%d.png' % args.setting))

    # print report
    trainer.extend(extensions.PrintReport(['epoch', 'loss/critic', 'loss/generator', 'elapsed_time']))

    # sample image
    out = "/home/ubuntu/images/numpara%d" % args.setting
    trainer.extend(out_generated_image(generator, 10, 10, args.seed + 10, out), trigger=(args.sampleimage_interval, "epoch"))

    # FID
    trainer.extend(calc_FID(generator), trigger=(args.evaluation_interval, 'epoch'))

    # logging
    trainer.extend(extensions.LogReport(log_name = "log_numpara%d" % args.setting))

    if args.resume:
        chainer.serializers.load_xpz(args.resume, trainer)

    # Run
    trainer.run()