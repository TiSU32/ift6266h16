import theano

from blocks.initialization import (NdarrayInitialization,
    IsotropicGaussian, Identity)

class CustomLSTMWeights(NdarrayInitialization):
    # Identity in the diagonal and IsotropicGaussian everywhere else
    def __init__(self, std=1, mean=0):
            self.gaussian_init = IsotropicGaussian(std = std, mean = mean)
            self.identity = Identity()

    def generate(self, rng, shape):
        if len(shape) != 2:
            raise ValueError
        assert shape[0] == shape[1]
        size = shape/4
        assert size*4 == shape[0]

        result = numpy.array([])
        for i in range(4):
            row = numpy.array([])
            for j in range(4):
                if i == j:
                    square = self.gaussian_init.generate(rng, (size,size))
                else:
                    square = self.identity.generate(rng, (size,size))
                row = numpy.hstack(row,square)
            result.vstack(row)
        return result.astype(theano.config.floatX)


class CustomLSTMBias(NdarrayInitialization):
    def generate(self, rng, shape):
        dest = numpy.empty(shape, dtype=theano.config.floatX)
        dest[...] = self._constant
        return dest