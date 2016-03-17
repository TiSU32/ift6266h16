import ipdb
import numpy
import theano
import matplotlib
import os
import sys
matplotlib.use('Agg')

from matplotlib import pyplot
from scipy.io import wavfile

from blocks.algorithms import (GradientDescent, Scale,
                               RMSProp, Adam,
                               StepClipping, CompositeRule)
from blocks.bricks import (Tanh, MLP,
                        Rectifier, Activation, Identity)

from blocks.bricks.sequence_generators import (
                        Readout, SequenceGenerator)
from blocks.bricks.recurrent import LSTM, RecurrentStack, GatedRecurrent
from blocks.extensions import FinishAfter, Printing, Timing, ProgressBar
from blocks.extensions.monitoring import (TrainingDataMonitoring, DataStreamMonitoring)
from blocks.extensions.predicates import OnLogRecord
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.training import TrackTheBest
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, IsotropicGaussian
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.utils import shared_floatx_zeros, shared_floatx

from fuel.transformers import (Mapping, FilterSources,
                            ForceFloatX, ScaleAndShift,Batch)
from fuel.schemes import (SequentialScheme,ConstantScheme)
from fuel.streams import DataStream

from theano import tensor, config, function

from play.bricks.custom import DeepTransitionFeedback
from play.bricks.frnn_model import FRNNEmitter

from play.datasets.monk_music import MonkMusic
from play.extensions import Flush, LearningRateSchedule, TimedFinish
from play.extensions.plot import Plot
from play.toy.segment_transformer import SegmentSequence

import pysptk as SPTK

#from fuel.datasets.youtube_audio import YouTubeAudio
from fuel.transformers.sequences import Window

###################
# Define parameters of the model
###################
sys.setrecursionlimit(1000000000)

#stride = 4000000
#batchSize = 50000
#miniBatches = 80
#freq = 16000

window_size = 16000
#batchSize = 50000
miniBatches = 80
#sequenceSize = batchSize*miniBatches


run_small_model = False
batch_size = 64 #for tpbtt
#seq_size = 128
k = 20

frame_size = 257
depth_x = 4
hidden_size_mlp_x = 1500
depth_theta = 4
hidden_size_mlp_theta = 1500
hidden_size_recurrent = 1500
depth_recurrent = 3
frnn_hidden_size = 500
frnn_step_size = 16

target_size = frame_size * k
lr = 5e-4
#lr = shared_floatx(lr, "learning_rate")

floatX = theano.config.floatX

save_dir = os.environ['RESULTS_DIR']
print 'save_dir: {}'.format(save_dir)
save_dir = os.path.join(save_dir,'monk_music/')

experiment_name = "monk_music_0"

######################
# Creating directories
######################
directories = [os.path.join(save_dir,'progress/'),\
    os.path.join(save_dir,'pkl/')]

for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

#################
# Prepare dataset
#################

def _transpose(data):
    return tuple(array.swapaxes(0,1) for array in data)

def _cut_top(data):
    return tuple(array[:, :, :frame_size] for array in data)

#data_dir = os.environ['FUEL_DATA_PATH']
data_dir = "/data/lisatmp4/sylvaint/data/"
data_dir = os.path.join(data_dir, 'monk_music/')

dataset = MonkMusic(which_sets = ('train',), filename = "XqaJ2Ol5cC4.hdf5",
    load_in_memory=True)

data_stats_file = os.path.join (data_dir,"monk_standardize.npz")
data_stats = None
data_stream = dataset.get_example_stream()

if os.path.exists(data_stats_file):
    data_stats = numpy.load(data_stats_file)
else:
    data_stats = {}
    data_stream = dataset.get_example_stream()
    it = data_stream.get_epoch_iterator()
    sequence = next(it)
    length = len(sequence[0])
    temp = numpy.random.choice(length,100000)
    temp = map(lambda l: float(l[0]),sequence[0][temp])
    temp = numpy.array(temp)
    data_stats["mean"] = temp.mean()
    data_stats["std"] = temp.std()
    numpy.save(data_stats_file,data_stats)


data_stream = Batch(data_stream=data_stream,iteration_scheme=ConstantScheme(batch_size))

data_stream = ScaleAndShift(data_stream,
                             scale = 1./data_stats["std"],
                             shift = -data_stats["mean"]/data_stats["std"])
#data_stream = Mapping(data_stream, _transpose)
#data_stream = SegmentSequence(data_stream, 16*seq_size, add_flag=True)
data_stream = ForceFloatX(data_stream)
train_stream = data_stream

"""
dataset = MonkMusic(which_sets = ('valid',), filename = "XqaJ2Ol5cC4.hdf5",
    load_in_memory=True)

data_stream = dataset.get_example_stream()
data_stream = Batch(data_stream=data_stream,iteration_scheme=ConstantScheme(batch_size))
data_stream = ScaleAndShift(data_stream,
                             scale = 1/data_stats["std"],
                             shift = -data_stats["mean"]/data_stats["std"])
#data_stream = SegmentSequence(data_stream, 16*seq_size, add_flag=True)
data_stream = ForceFloatX(data_stream)
valid_stream = data_stream
"""

x_tr = next(train_stream.get_epoch_iterator())


#################
# Model
#################

start_flag = tensor.scalar('start_flag')
x = tensor.tensor3('features')
#x = tensor.tensor3('features')

activations_x = [Rectifier()]*depth_x

dims_x = [frame_size] + [hidden_size_mlp_x]*(depth_x-1) + \
         [hidden_size_recurrent]

activations_theta = [Rectifier()]*depth_theta

dims_theta = [hidden_size_recurrent] + \
             [hidden_size_mlp_theta]*depth_theta

mlp_x = MLP(activations = activations_x,
            dims = dims_x)

feedback = DeepTransitionFeedback(mlp = mlp_x)

transition = [GatedRecurrent(dim=hidden_size_recurrent,
                   name = "gru_{}".format(i) ) for i in range(depth_recurrent)]

transition = RecurrentStack( transition,
            name="transition", skip_connections = True)

mlp_theta = MLP( activations = activations_theta,
             dims = dims_theta)

emitter = FRNNEmitter(
                  mlp               = mlp_theta,
                  target_size       = target_size,
                  frame_size        = frame_size,
                  k                 = k,
                  frnn_hidden_size   = frnn_hidden_size,
                  frnn_step_size    = frnn_step_size,
                  const             = 0.00001,
                  name              = "emitter")

source_names = [name for name in transition.apply.states if 'states' in name]
readout = Readout(
    readout_dim = hidden_size_recurrent,
    source_names =source_names,
    emitter=emitter,
    feedback_brick = feedback,
    name="readout")

generator = SequenceGenerator(readout=readout,
                              transition=transition,
                              name = "generator")

generator.weights_init = IsotropicGaussian(0.01)
generator.biases_init = Constant(0.)
generator.push_initialization_config()

generator.transition.biases_init = IsotropicGaussian(0.01,1)
generator.transition.push_initialization_config()

#steps = 2048
steps = 8
n_samples = 1

sample = ComputationGraph(generator.generate(n_steps=steps,
     batch_size=n_samples, iterate=True))
sample_fn = sample.get_theano_function()

generator.initialize()

#-2
outputs = sample_fn()[0]
print outputs

states = {}
states = generator.transition.apply.outputs

states = {name: shared_floatx_zeros((batch_size, hidden_size_recurrent))
        for name in states}

cost_matrix = generator.cost_matrix(x, **states)
cost = cost_matrix.mean()
cost.name = "nll"

cg = ComputationGraph(cost)
model = Model(cost)

#import ipdb; ipdb.set_trace()



print function([x], cost)(x_tr[0])

transition_matrix = VariableFilter(
            theano_name_regex="state_to_state")(cg.parameters)
for matr in transition_matrix:
    matr.set_value(0.98*numpy.eye(hidden_size_recurrent, dtype=floatX))

from play.utils import regex_final_value
extra_updates = []
for name, var in states.items():
  update = tensor.switch(start_flag, 0.*var,
              VariableFilter(theano_name_regex=regex_final_value(name)
                  )(cg.auxiliary_variables)[0])
  extra_updates.append((var, update))

#################
# Monitoring vars
#################
data_monitoring = []
# mean_data = x.mean(axis = (0,1)).copy(name="data_mean")
# sigma_data = x.std(axis = (0,1)).copy(name="data_std")
# max_data = x.max(axis = (0,1)).copy(name="data_max")
# min_data = x.min(axis = (0,1)).copy(name="data_min")
#
# data_monitoring = [mean_data, sigma_data,
#                      max_data, min_data]
#
# readout = generator.readout
# readouts = VariableFilter( applications = [readout.readout],
#     name_regex = "output")(cg.variables)[0]
#
# # the last value is the readouts
# mu, sigma, coeff, _ = readout.emitter.components(readouts,keep_parameters=True)
#
# mu = mu.reshape((-1, frame_size, k))
# sigma = sigma.reshape((-1, frame_size,k))
# #BEFORE: coeff = coeff.reshape((-1, k))
# #now there are coeffs for each draw
# coeff = coeff.reshape((-1, frame_size,k))
#
# min_sigma = sigma.min(axis=(0,2)).copy(name="sigma_min")
# mean_sigma = sigma.mean(axis=(0,2)).copy(name="sigma_mean")
# max_sigma = sigma.max(axis=(0,2)).copy(name="sigma_max")
#
# min_mu = mu.min(axis=(0,2)).copy(name="mu_min")
# mean_mu = mu.mean(axis=(0,2)).copy(name="mu_mean")
# max_mu = mu.max(axis=(0,2)).copy(name="mu_max")
#
# #min_coeff = coeff.min().copy(name="coeff_min")
# #mean_coeff = coeff.mean().copy(name="coeff_mean")
# #max_coeff = coeff.max().copy(name="coeff_max")
#
# min_coeff = coeff.min(axis=(0,2)).copy(name="coeff_min")
# mean_coeff = coeff.mean(axis=(0,2)).copy(name="coeff_mean")
# max_coeff = coeff.max(axis=(0,2)).copy(name="coeff_max")
#
# data_monitoring += [mean_sigma, min_sigma,
#     min_mu, max_mu, mean_mu, max_sigma,
#     mean_coeff, min_coeff, max_coeff]

#################
# Algorithm
#################
#n_batches = 200
n_batches = 5


algorithm = GradientDescent(
    cost=cost, \
    parameters=cg.parameters, \
    step_rule=CompositeRule([StepClipping(10.0), Adam(lr)]))
algorithm.add_updates(extra_updates)
lr = algorithm.step_rule.components[1].learning_rate

#monitoring_variables = [cost,lr]
monitoring_variables = [cost,lr]

train_monitor = TrainingDataMonitoring(
    variables=monitoring_variables + data_monitoring,
    every_n_batches=n_batches,
    prefix="train")

valid_monitor = DataStreamMonitoring(
     monitoring_variables,
     valid_stream,
     every_n_batches=n_batches,
     prefix="valid")

extensions=[
    ProgressBar(),
    Timing(every_n_batches=n_batches),
    train_monitor,
    valid_monitor,
    TrackTheBest('valid_nll', every_n_batches=n_batches),
    #Plot(save_dir+ "progress/" +(experiment_name)+".png",
    # [['train_nll',
    #   'valid_nll'], ['valid_learning_rate']],
    # every_n_batches=n_batches,
    # email=False),
    Checkpoint(
        save_dir+"pkl/best_"+experiment_name+".pkl",
        save_main_loop = False,
        use_cpickle=True
    ).add_condition(
        ['after_batch'], predicate=OnLogRecord('valid_nll_best_so_far')),
    Printing(every_n_batches = n_batches),
    Flush(every_n_batches=n_batches,
          before_first_epoch = True),
    LearningRateSchedule(lr,
          'valid_nll',
          #path = save_dir+"pkl/best_"+experiment_name+".pkl",
          path=None,
          states = states.values(),
          every_n_batches = n_batches,
          before_first_epoch = True)
    ]

main_loop = MainLoop(
    model=model,
    data_stream=train_stream,
    algorithm=algorithm,
    extensions = extensions)

print "Preparing to run main_loop"
main_loop.run()
