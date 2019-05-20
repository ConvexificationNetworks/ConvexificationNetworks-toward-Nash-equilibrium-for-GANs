
# this is used for regularization
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
        weight_matrix = xp.zeros((self.outdim, self.outdim * self.numnet), dtype=xp.float32)
        for i in range(self.numnet):
            q = xp.array(array[i, 0].data, dtype=xp.float32)
            weight_matrix[:, i * self.outdim:(i + 1) * self.outdim] = xp.identity(self.outdim, dtype=xp.float32) * q
        return Variable(weight_matrix)


class Generator(chainer.Chain):

    def __init__(self, dim=784, num_nets=784, latent=100, wscale=0.02):
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

                setattr(self, "l1_{}".format(net), L.Linear(None, 100, initialW=w, initial_bias=b))
                setattr(self, "l2_{}".format(net), L.Linear(None, 100, initialW=w, initial_bias=b))
                setattr(self, "l3_{}".format(net), L.Linear(None, 28 * 28, initialW=w, initial_bias=b))

                # set batchnormalization
                setattr(self, "bn1_{}".format(net), L.BatchNormalization(size=100))
                setattr(self, "bn2_{}".format(net), L.BatchNormalization(size=100))

    def make_hidden(self, batchsize):
        return xp.random.normal(0, 1, (batchsize, self.n_hidden, 1, 1)).astype(xp.float32)

    def __call__(self, z, test=False):

        for net in range(self.num_nets):
            h = F.relu(getattr(self, 'bn1_{}'.format(net))(getattr(self, 'l1_{}'.format(net))(z)))
            # h = F.relu(getattr(self, 'bn1_{}'.format(net))(getattr(self, 'l1_{}'.format(net))(z)))
            # h2 = F.relu(getattr(self, 'l2_{}'.format(net))(h))
            h = F.relu(getattr(self, 'bn2_{}'.format(net))(getattr(self, 'l2_{}'.format(net))(h)))
            h2 = F.sigmoid(getattr(self, 'l3_{}'.format(net))(h))

            if net == 0:
                X = h2
            else:
                X = F.concat((X, h2), axis=1)

        batchsize = X.shape[0]
        X = X.reshape(batchsize, self.num_nets * self.dim)
        # x = self.inter(X.T).T.data
        x = self.inter(X.T).T
        # x = Variable(xp.reshape(x, (batchsize, 1, 28, 28)))
        x = F.reshape(x, (-1, 1, 28, 28))
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

                setattr(self, "l1_{}".format(net), L.Linear(None, 100, initialW=w, initial_bias=b))
                setattr(self, "l2_{}".format(net), L.Linear(None, 100, initialW=w, initial_bias=b))
                # setattr(self, "l3_{}".format(net), L.Linear(None, 800, initialW = w, initial_bias = b))
                setattr(self, "l4_{}".format(net), L.Linear(None, 1, initialW=w, initial_bias=b))

                # set batchnormalization
                # setattr(self, "bn1_{}".format(net), L.BatchNormalization(size=800))
                # setattr(self, "bn2_{}".format(net), L.BatchNormalization(size=800))
                # setattr(self, "bn3_{}".format(net), L.BatchNormalization(size=800))

            # self.bn = L.BatchNormalization(size=2)

    def __call__(self, x, test=False):

        x = x.reshape(100, 784)
        for net in range(self.num_nets):
            # ここでhがnanになることで全てがnanになる（xは確かにnanではない）（そしてそれはその前のupdateでWがnanになってるから）
            # h = F.leaky_relu(getattr(self, 'bn1_{}'.format(net))(getattr(self, 'l1_{}'.format(net))(x)))
            h = F.leaky_relu(getattr(self, 'l1_{}'.format(net))(x))
            # h = F.leaky_relu(getattr(self, 'bn2_{}'.format(net))(getattr(self, 'l2_{}'.format(net))(h)))
            h = F.leaky_relu(getattr(self, 'l2_{}'.format(net))(h))
            # h = F.leaky_relu(getattr(self, 'bn3_{}'.format(net))(getattr(self, 'l3_{}'.format(net))(h)))
            # h = F.leaky_relu(getattr(self, 'l3_{}'.format(net))(h))
            h2 = getattr(self, 'l4_{}'.format(net))(h)

            if net == 0:
                # Y = h2.reshape(64, 1)
                Y = h2

            else:
                # Y = F.concat((Y, h2.reshape(64, 1)), axis = 1)
                Y = F.concat((Y, h2), axis=1)

        y = self.inter(Y.T).T
        # y = self.inter(self.bn(Y).T).T
        return y

