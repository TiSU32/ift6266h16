try:
   import cPickle as pickle
except:
   import pickle

import gzip
import numpy as np
 
def sigmoid (x):
    return 1./ (1. + np.exp(-x))

def sigmoid_diff (x):
    return sigmoid(x)*(1-sigmoid(x))

def softmax(x):
    temp = np.exp(x)
    return temp/(np.sum(temp))

class MLP(object):
    def __init__(self):
            # Simple alg, not trying to clean the code
            self.num_hidden = 300
            self.num_input = 784
            self.num_output = 10

            self.learning_rate = 0.1

            self.iteration_count = 0

            self.batch_count = 0
            self.batch_size = 1

            self.epoch_count = 0

            # hidden weights
            self.W = np.random.rand(self.num_hidden,self.num_input)/10.
            self.grad_loss_W = np.zeros((self.num_hidden,self.num_input))

            # output weights
            self.V = np.random.rand(self.num_output,self.num_hidden)/10.
            self.grad_loss_V = np.zeros((self.num_output,self.num_hidden))

            # hidden bias
            self.b = np.random.rand(self.num_hidden)/10.
            self.grad_loss_b = np.zeros(shape=self.num_hidden)

            # input bias
            self.c = np.random.rand(self.num_output)/10.
            self.grad_loss_c = np.zeros(shape=self.num_output)

            # values of hidden, output layers
            self.hidden = np.zeros(shape=self.num_hidden)
            self.output = np.zeros(shape=self.num_output)

    def forward_prop(self,x):
        self.hidden = sigmoid(np.dot(self.W,x) + self.b)
        self.output = softmax(np.dot(self.V,self.hidden) + self.c)

        return self.output

    def backward_prop(self,x,y):

        y_one_hot_encoded = np.array([1. if i == y else 0. for i in range(0,self.num_output)])
        grad_loss_activations = self.forward_prop(x) - y_one_hot_encoded
        self.grad_loss_c = grad_loss_activations
        self.grad_loss_V = np.outer(grad_loss_activations,self.hidden)


        #grad_loss_hidden = np.dot(grad_loss_activations,self.V)
        grad_loss_hidden = np.dot(self.V.T,grad_loss_activations)


        grad_loss_activations_input = np.multiply(grad_loss_hidden,np.multiply(self.hidden,1-self.hidden)) #
        #print grad_loss_activations_input.shape
        self.grad_loss_b = grad_loss_activations_input
        self.grad_loss_W = np.outer(grad_loss_activations_input,x)


    def update(self,train_set,test_set):
        lr = self.learning_rate

        # change later
        self.backward_prop(train_set[0][self.batch_count],train_set[1][self.batch_count])

        assert(self.b.shape == self.grad_loss_b.shape)
        assert(self.c.shape == self.grad_loss_c.shape)
        assert(self.W.shape == self.grad_loss_W.shape)
        assert(self.V.shape == self.grad_loss_V.shape)

        self.b -= lr*self.grad_loss_b
        self.c -= lr*self.grad_loss_c
        self.W -= lr*self.grad_loss_W
        self.V -= lr*self.grad_loss_V

        print ("Iteration: {}".format(self.iteration_count,self))
        if (self.iteration_count % 10000 == 0):
            print "Test Precision: {}".format(self.precision(test_set))
            #print "Train Precision: {}".format(self.precision(train_set))
            #print "Test Loss: {}".format()
            #print "Train Precision: {}".format(self.precision(test_set))

        self.iteration_count += 1
        self.batch_count += 1

    def precision(self,test_set):
        results = np.apply_along_axis (lambda l: np.argmax(self.forward_prop(l)),axis = 1,arr=test_set[0])

        return np.sum(results==test_set[1])/float(test_set[1].shape[0])

    def new_epoch(self):
        self.epoch_count += 1
        self.batch_count = 0
        

with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f)
    NN = MLP()

    while NN.epoch_count < 10:
        while NN.batch_count < train_set[0].shape[0]:
            NN.update(train_set,test_set)
        NN.new_epoch()

    # saving as a pickle representation
    output_file = open("NN.pkl","w")
    pickle.dump(NN,output_file)














