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

from play.utils import FRNN_NLL
from play.bricks.custom import SoftPlus

floatX = config.floatX

import numpy

class FRNNEmitter(AbstractEmitter, Initializable, Random):
    """An RNN emitter for the case of real outputs.
    Parameters
    ----------
    """
    def __init__(self, mlp, target_size, frame_size, k, frnn_hidden_size, \
            frnn_step_size, const=1e-5, **kwargs):

        super(FRNNEmitter, self).__init__(**kwargs)

        self.mlp = mlp
        self.target_size = target_size
        self.frame_size = frame_size
        self.k = k
        self.frnn_hidden_size = frnn_hidden_size
        self.const = const
        self.input_dim = self.mlp.output_dim

        self.frnn_step_size = frnn_step_size

        # adding a step if the division is not exact.
        self.number_of_steps = frame_size // frnn_step_size
        self.last_steps = frame_size % frnn_step_size
        if self.last_steps != 0:
            self.number_of_steps += 1

        self.mu = MLP(activations=[Identity()],
                dims=[frnn_hidden_size, k*frnn_step_size],
                name=self.name + "_mu")
        self.sigma = MLP(activations=[SoftPlus()],
                dims=[frnn_hidden_size, k*frnn_step_size],
                name=self.name + "_sigma")

        self.coeff = MLP(activations=[Identity()],
                dims=[frnn_hidden_size, k],
                name=self.name + "_coeff")

        self.coeff2 = NDimensionalSoftmax()

        self.frnn_initial_state = Linear(
            input_dim = self.input_dim,
            output_dim=frnn_hidden_size,
            name="frnn_initial_state")

        #self.frnn_hidden = Linear(
        #    input_dim=frnn_hidden_size,
        #    output_dim=frnn_hidden_size,
        #    activation=Tanh(),
        #    name="frnn_hidden")

        self.frnn_activation = Tanh(
            name="frnn_activation")

        self.frnn_linear_transition_state = Linear (
            input_dim = frnn_hidden_size,
            output_dim= frnn_hidden_size,
            name="frnn_linear_transition_state")

        self.frnn_linear_transition_input = Linear (
            input_dim = self.frnn_step_size,
            output_dim = frnn_hidden_size,
            name="frnn_linear_transition_input")

        #self.frnn_linear_transition_output = Linear (
        #    input_dim = frnn_hidden_size,
        #    output_dim = self.rnn_hidden_dim,
        #    name="frnn_linear_transition_output")

        self.children = [self.mlp,self.mu,self.sigma,self.coeff,
            self.coeff2,self.frnn_initial_state,self.frnn_activation,
            self.frnn_linear_transition_state,
            self.frnn_linear_transition_input]

    @application
    def emit(self,readouts):
        """
        keep_parameters is True if mu,sigma,coeffs must be stacked and returned
        if false, only the result is given, the others will be empty list.

        """
        # initial state
        state = self.frnn_initial_state.apply(\
            self.mlp.apply(readouts))

        results = []

        for i in range(self.number_of_steps):
            last_iteration = (i == self.number_of_steps - 1)

            # First generating distribution parameters and sampling.
            mu = self.mu.apply(state)
            sigma = self.sigma.apply(state) + self.const
            coeff = self.coeff2.apply(self.coeff.apply(state),\
                extra_ndim=state.ndim - 2) + self.const

            shape_result = coeff.shape
            shape_result = tensor.set_subtensor(shape_result[-1],self.frnn_step_size)
            ndim_result = coeff.ndim

            mu = mu.reshape((-1, self.frnn_step_size,self.k))
            sigma = sigma.reshape((-1, self.frnn_step_size,self.k))
            coeff = coeff.reshape((-1, self.k))

            sample_coeff = self.theano_rng.multinomial(pvals = coeff, dtype=coeff.dtype)
            idx = predict(sample_coeff, axis = -1)
            #idx = predict(coeff, axis = -1) use this line for using most likely coeff.

            #shapes (ls*bs)*(fs)
            mu = mu[tensor.arange(mu.shape[0]), :,idx]
            sigma = sigma[tensor.arange(sigma.shape[0]), :,idx]

            epsilon = self.theano_rng.normal(
                size=mu.shape,
                avg=0.,
                std=1.,
                dtype=mu.dtype)

            result = mu + sigma*epsilon#*0.6 #reduce variance.
            result = result.reshape(shape_result, ndim = ndim_result)
            results.append(result)

            # if the total size does not correspond to the frame_size,
            #this removes the need for padding
            if not last_iteration:
                state = self.frnn_activation.apply(
                        self.frnn_linear_transition_state.apply(state) +
                        self.frnn_linear_transition_input.apply(result))

        results = tensor.stack(results,axis=-1)
        results = tensor.flatten(results,outdim=results.ndim-1)

        # truncate if not good size
        if self.last_steps != 0:
            results = results[tuple([slice(0,None)] * \
                (results.ndim-1) +[slice(0,self.frame_size)])]

        return results

    @application
    def cost(self, readouts, outputs):
        # initial state
        state = self.frnn_initial_state.apply(\
            self.mlp.apply(readouts))

        inputs = outputs

        mus = []
        sigmas = []
        coeffs = []

        for i in range(self.number_of_steps):
            last_iteration = (i == self.number_of_steps - 1)

            # First generating distribution parameters and sampling.
            freq_mu = self.mu.apply(state)
            freq_sigma = self.sigma.apply(state) + self.const
            freq_coeff = self.coeff2.apply(self.coeff.apply(state),\
                extra_ndim=state.ndim - 2) + self.const

            freq_mu = freq_mu.reshape((-1,self.frnn_step_size,self.k))
            freq_sigma = freq_sigma.reshape((-1,self.frnn_step_size,self.k))
            freq_coeff = freq_coeff.reshape((-1,self.k))
            #mu,sigma: shape (-1,fs,k)
            #coeff: shape (-1,k)

            mus.append(freq_mu)
            sigmas.append(freq_sigma)
            coeffs.append(freq_coeff)

            index = self.frnn_step_size
            freq_inputs = inputs[tuple([slice(0,None)] * \
                (inputs.ndim-1) +[slice(index,index+self.frnn_step_size)])]

            if not last_iteration:
                state = self.frnn_activation.apply(
                    self.frnn_linear_transition_state.apply(state) +
                    self.frnn_linear_transition_input.apply(freq_inputs))

        mus = tensor.stack(mus,axis=-2)
        sigmas = tensor.stack(sigmas,axis=-2)
        coeffs = tensor.stack(coeffs,axis=-2)

        mus = mus.reshape((-1,self.frnn_step_size*self.number_of_steps,self.k))
        sigmas = sigmas.reshape((-1,self.frnn_step_size*self.number_of_steps,self.k))
        coeffs = coeffs.repeat(self.frnn_step_size,axis=-2)

        mus = mus[tuple([slice(0,None)] * \
                (mus.ndim-2) +[slice(0,self.frame_size)] + [slice(0,None)])]
        sigmas = sigmas[tuple([slice(0,None)] * \
                (sigmas.ndim-2) +[slice(0,self.frame_size)] + [slice(0,None)])]
        coeffs = coeffs[tuple([slice(0,None)] * \
                (coeffs.ndim-2) +[slice(0,self.frame_size)] + [slice(0,None)])]
        # actually prob not necessary
        mu = mus.reshape((-1,self.target_size))
        sigma = sigmas.reshape((-1,self.target_size))
        coeff = coeffs.reshape((-1, self.target_size))

        return FRNN_NLL (y=outputs, mu=mu, sig=sigma, coeff=coeff,\
            frame_size=self.frame_size,k=self.k)

    @application
    def initial_outputs(self, batch_size):
        return tensor.zeros((batch_size, self.frame_size), dtype=floatX)

    def get_dim(self, name):
	# modification here to ensure the right dim.
        if name == 'outputs':
            return self.frame_size
        return super(FRNNEmitter, self).get_dim(name)
