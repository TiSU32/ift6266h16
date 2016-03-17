import theano
import ipdb

from theano import tensor, config

from blocks.bricks import (Activation, Initializable, MLP, Random,
                        Identity, NDimensionalSoftmax, Logistic, Tanh, Linear)
from blocks.bricks.base import application
from blocks.bricks.sequence_generators import (AbstractEmitter,
                        AbstractFeedback)

from blocks.utils import print_shape
from blocks.bricks.recurrent import SimpleRecurrent
from cle.cle.cost import Gaussian
from cle.cle.utils import predict

from play.utils import GMM

floatX = config.floatX

import numpy

class SoftPlus(Activation):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return tensor.nnet.softplus(input_)

class DeepTransitionFeedback(AbstractFeedback, Initializable):
    def __init__(self, mlp, **kwargs):
        super(DeepTransitionFeedback, self).__init__(**kwargs)

        self.mlp = mlp
        self.feedback_dim = mlp.output_dim
        self.children = [self.mlp]

    @application(outputs=['feedback'])
    def feedback(self, outputs):
        return self.mlp.apply(outputs)

    def get_dim(self, name):
        if name == 'feedback':
            return self.feedback_dim
        return super(DeepTransitionFeedback, self).get_dim(name)

class GaussianMLP(Initializable):
    """An mlp brick that branchs out to output
    sigma and mu for Gaussian dist
    Parameters
    ----------
    mlp: MLP brick
        the main mlp to wrap around.
    dim:
        output dim
    """

    def __init__(self, mlp, dim, const=0., **kwargs):
        super(GaussianMLP, self).__init__(**kwargs)

        self.dim = dim
        self.const = const
        input_dim = mlp.output_dim
        self.mu = MLP(activations=[Identity()],
                      dims=[input_dim, dim],
                      weights_init=self.weights_init,
                      biases_init=self.biases_init,
                      name=self.name + "_mu")
        self.sigma = MLP(activations=[SoftPlus()],
                         dims=[input_dim, dim],
                         weights_init=self.weights_init,
                         biases_init=self.biases_init,
                         name=self.name + "_sigma")

        self.mlp = mlp
        self.children = [self.mlp, self.mu, self.sigma]
        self.children.extend(self.mlp.children)

    @application
    def apply(self, inputs):
        state = self.mlp.apply(inputs)
        mu = self.mu.apply(state)
        sigma = self.sigma.apply(state) + self.const

        return mu, sigma

    @property
    def output_dim(self):
        return self.dim

class GaussianEmitter(AbstractEmitter, Initializable, Random):
    """A Gaussian emitter for the case of real outputs.
    Parameters
    ----------
    initial_output :
        The initial output.
    """
    def __init__(self, gaussianmlp, output_size, **kwargs):
        super(GaussianEmitter, self).__init__(**kwargs)
        self.gaussianmlp = gaussianmlp
        self.children = [self.gaussianmlp]
        self.output_size = output_size

    def components(self, readouts):
        # Returns Mu and Sigma
        return self.gaussianmlp.apply(readouts)

    @application
    def emit(self, readouts):
        mu, sigma = self.components(readouts)

        epsilon = self.theano_rng.normal(size=mu.shape,
                                         avg=0., std=1.,
                                         dtype=mu.dtype)

        return mu + sigma*epsilon

    @application
    def cost(self, readouts, outputs):
        mu, sigma = self.components(readouts)
        return Gaussian(outputs, mu, sigma)

    @application
    def initial_outputs(self, batch_size):
        return tensor.zeros((batch_size, self.output_size), dtype=floatX)

    def get_dim(self, name):
        if name == 'outputs':
            return self.gaussianmlp.output_dim
        return super(GaussianEmitter, self).get_dim(name)


class GMMMLP(Initializable):
    """An mlp brick that branchs out to output
    sigma and mu for GMM
    Parameters
    ----------
    mlp: MLP brick
        the main mlp to wrap around.
    dim:
        output dim
    """
    def __init__(self, mlp, dim, k, const=1e-5, **kwargs):
        super(GMMMLP, self).__init__(**kwargs)

        self.dim = dim
        self.const = const
        self.k = k
        input_dim = mlp.output_dim
        self.mu = MLP(activations=[Identity()],
                      dims=[input_dim, dim],
                      name=self.name + "_mu")
        self.sigma = MLP(activations=[SoftPlus()],
                         dims=[input_dim, dim],
                         name=self.name + "_sigma")

        self.coeff = MLP(activations=[Identity()],
                         dims=[input_dim, k],
                         name=self.name + "_coeff")

        self.coeff2 = NDimensionalSoftmax()
        self.mlp = mlp
        self.children = [self.mlp, self.mu,
                         self.sigma, self.coeff, self.coeff2]
        #self.children.extend(self.mlp.children)

    @application
    def apply(self, inputs):
        state = self.mlp.apply(inputs)
        mu = self.mu.apply(state)
        sigma = self.sigma.apply(state) + self.const
        coeff = self.coeff2.apply(self.coeff.apply(state),
            extra_ndim=state.ndim - 2) + self.const
        return mu, sigma, coeff

    @property
    def output_dim(self):
        return self.dim



class GMMEmitter(AbstractEmitter, Initializable, Random):
    """A GMM emitter for the case of real outputs.
    Parameters
    ----------
    """
    def __init__(self, gmmmlp, output_size, k, **kwargs):
        super(GMMEmitter, self).__init__(**kwargs)
        self.gmmmlp = gmmmlp
        self.children = [self.gmmmlp]
        self.output_size = output_size
        self.k = k

    def components(self, readouts):
        # Returns Mu and Sigma
        return self.gmmmlp.apply(readouts)

    @application
    def emit(self, readouts):
        mu, sigma, coeff = self.gmmmlp.apply(readouts)

        frame_size = mu.shape[-1]/coeff.shape[-1]
        k = coeff.shape[-1]
        shape_result = coeff.shape
        shape_result = tensor.set_subtensor(shape_result[-1],frame_size)
        ndim_result = coeff.ndim

        mu = mu.reshape((-1, frame_size, k))
        sigma = sigma.reshape((-1, frame_size,k))
        coeff = coeff.reshape((-1, k))

        sample_coeff = self.theano_rng.multinomial(pvals = coeff, dtype=coeff.dtype)
        idx = predict(sample_coeff, axis = -1)
        #idx = predict(coeff, axis = -1) use this line for using most likely coeff.

        mu = mu[tensor.arange(mu.shape[0]), :, idx]
        sigma = sigma[tensor.arange(sigma.shape[0]), :, idx]

        epsilon = self.theano_rng.normal(
            size=mu.shape,avg=0.,
            std=1.,
            dtype=mu.dtype)

        result = mu + sigma*epsilon#*0.6 #reduce variance.

        return result.reshape(shape_result, ndim = ndim_result)

    @application
    def cost(self, readouts, outputs):
        mu, sigma, coeff = self.components(readouts)
        #ipdb.set_trace()
        return GMM(outputs, mu, sigma, coeff)

    @application
    def initial_outputs(self, batch_size):
        return tensor.zeros((batch_size, self.output_size), dtype=floatX)

    def get_dim(self, name):
        if name == 'outputs':
            return self.gmmmlp.output_dim
        return super(GMMEmitter, self).get_dim(name)




###############################
#
# This code is un construction.  Later it will be part of its own
# model files.
#
###############################

class PitchGaussianEmitter(AbstractEmitter, Initializable, Random):
    """A Pitch emitter for the case of real + voiced/unvoiced.
    Parameters
    ----------
    initial_output :
        The initial output.
    """
    def __init__(self, mlp, const=1e-5, **kwargs):
        super(PitchGaussianEmitter, self).__init__(**kwargs)
        self.mlp = mlp
        input_dim = self.mlp.output_dim
        self.const = const

        self.mu = MLP(activations=[Identity()],
                      dims=[input_dim, 1],
                      name=self.name + "_mu")
        self.sigma = MLP(activations=[SoftPlus()],
                         dims=[input_dim, 1],
                         name=self.name + "_sigma")
        self.binary = MLP(activations=[Logistic()],
                 dims=[input_dim, 1],
                 name=self.name + "_binary")

        self.children = [self.mlp, self.mu, self.sigma, self.binary]

    def components(self, readouts):
        state = self.mlp.apply(readouts)
        mu = self.mu.apply(state)
        sigma = self.sigma.apply(state) + self.const
        binary = self.binary.apply(state)
        binary = (binary+self.const)*(1-2*self.const)
        return mu, sigma, binary

    @application
    def emit(self, readouts):
        mu, sigma, binary = self.components(readouts)

        epsilon = self.theano_rng.normal(size=mu.shape,
                                         avg=0., std=1.,
                                         dtype=mu.dtype)

        un = self.theano_rng.uniform(size=binary.shape)
        binary_sample = tensor.cast(un < binary, floatX)

        f0_sample = mu + sigma*epsilon
        f0_sample = f0_sample*binary_sample

        return tensor.concatenate([f0_sample,binary_sample], axis = -1)

    @application
    def cost(self, readouts, outputs):
        mu, sigma, binary = self.components(readouts)
        # Add term to model nll of voiced
        outputs_shape = outputs.shape
        outputs_ndim = outputs.ndim
        #outputs_shape[-1] should be 2
        outputs = outputs.reshape((-1, outputs_shape[-1]))
        outputs = outputs.T
        f0 = outputs[0]
        voiced = outputs[1]
        f0 = f0.reshape(outputs_shape[:-1], ndim = outputs_ndim-1)
        voiced = voiced.reshape(outputs_shape[:-1], ndim = outputs_ndim-1)

        f0 = f0.dimshuffle(*range(f0.ndim) + ['x'])

        binary = binary.flatten(ndim = binary.ndim-1)

        c_b =  tensor.xlogx.xlogy0(voiced,  binary) + tensor.xlogx.xlogy0(1 - voiced, 1 - binary)

        return Gaussian(f0, mu, sigma) * voiced - c_b

    @application
    def initial_outputs(self, batch_size):
        return tensor.zeros((batch_size, 2), dtype=floatX)

    def get_dim(self, name):
        if name == 'outputs':
            return 2
        return super(PitchGaussianEmitter, self).get_dim(name)

# Full spectrum + pitch emitter. It uses gmm for the spectrum.
class SPF0Emitter(AbstractEmitter, Initializable, Random):
    """A Pitch and SP emitter.

    """
    def __init__(self, mlp, frame_size = 259, k = 20, const=1e-5, **kwargs):
        super(SPF0Emitter, self).__init__(**kwargs)
        self.mlp = mlp
        input_dim = self.mlp.output_dim
        self.const = const
        self.frame_size = frame_size

        mlp_gmm = GMMMLP(mlp = mlp,
                  dim = (frame_size-2)*k,
                  k = k,
                  const = const)

        self.gmm_emitter = GMMEmitter(
                  gmmmlp = mlp_gmm,
                  output_size = frame_size-2,
                  k = k,
                  name = "gmm_emitter"
                  )

        self.mu = MLP(activations=[Identity()],
                      dims=[input_dim, 1],
                      name=self.name + "_mu")
        self.sigma = MLP(activations=[SoftPlus()],
                         dims=[input_dim, 1],
                         name=self.name + "_sigma")
        self.binary = MLP(activations=[Logistic()],
                 dims=[input_dim, 1],
                 name=self.name + "_binary")

        self.children = [self.mlp, self.mu, self.sigma,
            self.binary, self.gmm_emitter]

    def components(self, readouts):
        state = self.mlp.apply(readouts)
        mu = self.mu.apply(state)
        sigma = self.sigma.apply(state) + self.const
        binary = self.binary.apply(state)
        binary = (binary+self.const)*(1-2*self.const)
        return mu, sigma, binary

    @application
    def emit(self, readouts):
        mu, sigma, binary = self.components(readouts)

        epsilon = self.theano_rng.normal(size=mu.shape,
                                         avg=0., std=1.,
                                         dtype=mu.dtype)

        un = self.theano_rng.uniform(size=binary.shape)*0. + 0.5
        binary_sample = tensor.cast(un < binary, floatX)

        f0_sample = mu + sigma*epsilon
        f0_sample = f0_sample*binary_sample

        sample_gmm = self.gmm_emitter.emit(readouts)

        return tensor.concatenate([sample_gmm,f0_sample,binary_sample], axis = -1)

    @application
    def cost(self, readouts, outputs):
        mu, sigma, binary = self.components(readouts)
        # Add term to model nll of voiced
        outputs_shape = outputs.shape
        outputs_ndim = outputs.ndim
        #outputs_shape[-1] should be 2
        outputs = outputs.reshape((-1, outputs_shape[-1]))
        outputs = outputs.T

        #ipdb.set_trace()
        sp = outputs[:-2]
        sp = sp.T
        sp = sp.reshape(tensor.set_subtensor(outputs_shape[-1], -1), ndim = outputs_ndim )
        #tensor.set_subtensor(outputs_shape[-1], -1)
        f0 = outputs[-2]
        voiced = outputs[-1]
        f0 = f0.reshape(outputs_shape[:-1], ndim = outputs_ndim-1)
        voiced = voiced.reshape(outputs_shape[:-1], ndim = outputs_ndim-1)

        f0 = f0.dimshuffle(*range(f0.ndim) + ['x'])

        binary = binary.flatten(ndim = binary.ndim-1)

        c_b =  tensor.xlogx.xlogy0(voiced,  binary) + tensor.xlogx.xlogy0(1 - voiced, 1 - binary)

        cost_gmm = self.gmm_emitter.cost(readouts, sp)
        return Gaussian(f0, mu, sigma) * voiced - c_b + cost_gmm

    @application
    def initial_outputs(self, batch_size):
        return tensor.zeros((batch_size, self.frame_size), dtype=floatX)

    def get_dim(self, name):
        if name == 'outputs':
            return self.frame_size
        return super(SPF0Emitter, self).get_dim(name)
