
# convexifying layer
class Intersection2(chainer.Link):

    def __init__(self, outdim, numnet):
        super(Intersection2, self).__init__()
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
        weight_matrix = np.zeros((self.outdim, self.outdim * self.numnet), dtype=np.float32)
        for i in range(self.numnet):
            q = np.array(array[i, 0].data, dtype=np.float32)
            weight_matrix[:, i * self.outdim:(i + 1) * self.outdim] = np.identity(self.outdim, dtype=np.float32) * q
        return Variable(weight_matrix)

    
class Generator(chainer.Chain):

    def __init__(self, dim=2, num_nets=2, latent=100, wscale=0.02):
        super(Generator, self).__init__()
        self.dim = dim
        self.num_nets = num_nets
        self.wscale = wscale
        self.n_hidden = latent

        with self.init_scope():
            self.inter = Intersection2(self.dim, self.num_nets)

            for net in range(self.num_nets):
                w = chainer.initializers.Normal(self.wscale)
                b = chainer.initializers.Normal(self.wscale)

                setattr(self, "l1_{}".format(net), L.Linear(None, 48, initialW=w, initial_bias=b))
                setattr(self, "l2_{}".format(net), L.Linear(None, 48, initialW=w, initial_bias=b))
                setattr(self, "l3_{}".format(net), L.Linear(None, 2, initialW=w, initial_bias=b))


    def make_hidden(self, batchsize):
        return np.random.uniform(-1, 1, (batchsize, self.n_hidden)).astype(np.float32)

    def __call__(self, z, test=False):

        for net in range(self.num_nets):
            h = F.relu(getattr(self, 'l1_{}'.format(net))(z))
            h2 = F.relu(getattr(self, 'l2_{}'.format(net))(h))
            h2 = getattr(self, 'l3_{}'.format(net))(h2)

            if net == 0:
                X = h2
            else:
                X = F.concat((X, h2), axis=1)

        batchsize = X.shape[0]
        X = X.reshape(batchsize, self.num_nets * self.dim)
        x = self.inter(X.T).T
        return x


class Critic(chainer.Chain):
    def __init__(self, num_nets=784, wscale=0.02):
        super(Critic, self).__init__()
        self.num_nets = num_nets
        self.wscale = wscale

        with self.init_scope():
            self.inter = Intersection2(1, self.num_nets)

            for net in range(self.num_nets):
                w = chainer.initializers.Normal(self.wscale)
                b = chainer.initializers.Normal(self.wscale)

                setattr(self, "l1_{}".format(net), L.Linear(None, 48, initialW=w, initial_bias=b))
                setattr(self, "l2_{}".format(net), L.Linear(None, 48, initialW=w, initial_bias=b))
                setattr(self, "l3_{}".format(net), L.Linear(None, 1, initialW=w, initial_bias=b))


    def __call__(self, x, test=False):

        x = x.reshape(64, 2)
        for net in range(self.num_nets):
            h = F.leaky_relu(getattr(self, 'l1_{}'.format(net))(x))
            h = F.leaky_relu(getattr(self, 'l2_{}'.format(net))(h))
            h2 = getattr(self, 'l3_{}'.format(net))(h)

            if net == 0:
                Y = h2

            else:
                Y = F.concat((Y, h2), axis = 1)

        y = self.inter(Y.T)

        return y