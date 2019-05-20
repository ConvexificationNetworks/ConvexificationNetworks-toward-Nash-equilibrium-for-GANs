
class Intersection(chainer.Link):

    def __init__(self, outdim, numnet):
        super(Intersection, self).__init__()
        self.outdim = outdim
        self.numnet = numnet
        with self.init_scope():
            W = chainer.initializers.One()
            self.W = variable.Parameter(W)
            self.W.initialize((self.numnet, 1))

    def __call__(self, x):
        if self.outdim == 1:
            weight = F.relu(self.W.T)
        else:
            weight = F.relu(self.make_weight(self.W))

        return F.matmul(weight, x)

    def make_weight(self, array):
        weight_matrix = xp.zeros((self.outdim, self.outdim * self.numnet), dtype=xp.float32)
        for i in range(self.numnet):
            q = xp.array(array[i, 0].data, dtype=xp.float32)
            weight_matrix[:, i * self.outdim:(i + 1) * self.outdim] = xp.identity(self.outdim, dtype=xp.float32) * q
        return Variable(weight_matrix)


    
class DCGANGenerator(chainer.Chain):
    
    def __init__(self, dim=32*32*3, num_nets=1, n_hidden=100, bottom_width=4, ch=512, wscale=0.02, hidden_activation=F.relu, use_bn=True):
        super(DCGANGenerator, self).__init__()
        self.n_hidden = n_hidden
        self.ch = ch
        self.bottom_width = bottom_width
        self.hidden_activation = hidden_activation
        self.use_bn = use_bn
        self.dim = dim
        self.num_nets = num_nets
        self.wscale = wscale

        with self.init_scope():
            self.inter = Intersection(self.dim, self.num_nets)
            
            for net in range(self.num_nets):
                w = chainer.initializers.Normal(self.wscale)
                b = chainer.initializers.Normal(self.wscale)

                # set network
                setattr(self, "l0_{}".format(net), L.Linear(self.n_hidden, bottom_width * bottom_width * ch, initialW=w))
                setattr(self, "dc1_{}".format(net), L.Deconvolution2D(ch, ch // 2, 4, stride=2, pad=1, initialW=w))
                setattr(self, "dc2_{}".format(net), L.Deconvolution2D(ch // 2, ch // 4, 4, stride=2, pad=1, initialW=w))
                setattr(self, "dc3_{}".format(net), L.Deconvolution2D(ch // 4, ch // 8, 4, stride=2, pad=1, initialW=w))
                setattr(self, "dc4_{}".format(net), L.Deconvolution2D(ch//8, 3, 3, stride=1, pad=1, initialW=w))
                
                # set batchnormalization
                setattr(self, "bn1_{}".format(net), L.BatchNormalization(size=ch))
                setattr(self, "bn2_{}".format(net), L.BatchNormalization(size=ch//2))
                setattr(self, "bn3_{}".format(net), L.BatchNormalization(size=ch//4))
                setattr(self, "bn4_{}".format(net), L.BatchNormalization(size=ch//8))
                
                

    def make_hidden(self, batchsize):
        return xp.random.uniform(-1,1, (batchsize, self.n_hidden, 1, 1)).astype(xp.float32)


    def __call__(self, z):

        for net in range(self.num_nets):
            h = F.reshape(self.hidden_activation(getattr(self, 'l0_{}'.format(net))(z)), (len(z), self.ch, self.bottom_width, self.bottom_width))
            h = self.hidden_activation(getattr(self, 'bn1_{}'.format(net))(h))
            h = self.hidden_activation(getattr(self, 'bn2_{}'.format(net))(getattr(self, 'dc1_{}'.format(net))(h)))
            h = self.hidden_activation(getattr(self, 'bn3_{}'.format(net))(getattr(self, 'dc2_{}'.format(net))(h)))
            h = self.hidden_activation(getattr(self, 'bn4_{}'.format(net))(getattr(self, 'dc3_{}'.format(net))(h)))
            h2 = F.sigmoid(getattr(self, 'dc4_{}'.format(net))(h))

            if net == 0:
                X = h2
            else:
                X = F.concat((X, h2), axis = 1)
                
        batchsize = X.shape[0]
        X = X.reshape(batchsize, self.num_nets*self.dim)
        x = self.inter(X.T).T
        x = F.reshape(x, (-1,3, 32, 32))
        return x

class WGANDiscriminator(chainer.Chain):

    def __init__(self, num_nets=1, bottom_width=4, ch=512, wscale=0.02, output_dim=1):
        
        super(WGANDiscriminator, self).__init__()
        
        self.num_nets = num_nets
        self.wscale = wscale
        w = chainer.initializers.Normal(wscale)

        with self.init_scope():
            
            self.inter = Intersection(1, self.num_nets)
            
            for net in range(self.num_nets):
                w = chainer.initializers.Normal(self.wscale)
                #b = chainer.initializers.Normal(self.wscale)
                setattr(self, "c0_{}".format(net), L.Convolution2D(in_channels=3, out_channels=64, ksize=3, stride=1, pad=1, initialW=w))
                setattr(self, "c1_{}".format(net), L.Convolution2D(in_channels=ch//8, out_channels=128, ksize=4, stride=2, pad=1, initialW=w))
                setattr(self, "c2_{}".format(net), L.Convolution2D(in_channels=ch//4, out_channels=256, ksize=4, stride=2, pad=1, initialW=w))
                setattr(self, "c3_{}".format(net), L.Convolution2D(in_channels=ch//2, out_channels=512, ksize=4, stride=2, pad=1, initialW=w))
                setattr(self, "l4_{}".format(net), L.Linear(bottom_width * bottom_width * ch, output_dim, 1, initialW=w))
                setattr(self, "bn1_{}".format(net), L.BatchNormalization(size=ch//4, use_gamma=False))
                setattr(self, "bn2_{}".format(net), L.BatchNormalization(size=ch//2, use_gamma=False))
                setattr(self, "bn3_{}".format(net), L.BatchNormalization(size=ch//1, use_gamma=False))
                

    def __call__(self, x):

        for net in range(self.num_nets):
            self.h0 = F.leaky_relu(getattr(self, 'c0_{}'.format(net))(x))
            self.h1 = F.leaky_relu(getattr(self, "bn1_{}".format(net))(getattr(self, 'c1_{}'.format(net))(self.h0)))
            self.h2 = F.leaky_relu(getattr(self, "bn2_{}".format(net))(getattr(self, 'c2_{}'.format(net))(self.h1)))
            self.h3 = F.leaky_relu(getattr(self, "bn3_{}".format(net))(getattr(self, 'c3_{}'.format(net))(self.h2)))
            self.h4 = getattr(self, 'l4_{}'.format(net))(self.h3)

            if net == 0:
                Y = self.h4
            else:
                Y = F.concat((Y, self.h4), axis = 1)
           
        #print(type(Y.T))
        y = self.inter(Y.T).T
        
        return y
