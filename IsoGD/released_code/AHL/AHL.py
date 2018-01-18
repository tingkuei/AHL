import numpy
import theano
import theano.tensor as T
import theano.tensor.shared_randomstreams
from theano import shared


from mlp import _dropout_from_layer, HiddenLayer
from logistic_sgd import LogisticRegression
from util import hardtanh


"""
Adaptive Hidden Layer is a new network layer composed of two components:
    (1) neuron group
    (2) selector
params definition:
    :type rng: numpy.random.RandomState
    :param rng: a random number generator used to initialize weights of neuron group (if None) and selector

    :type input: input theano.tensor.TensorType 
    :input: input for the neuron group, each neuron in the group  will share the same input
 
    :type selector_input: theano.tensor.TensorType 
    :selector_input: input for the selector, typically the input of the selector is same as the input of neuron group

    :type n_in: int
    :param n_in: number of input units 
 
    :type n_hidden: int
    :param n_hidden: number of output units
 
    :type n_selector_in: int
    :param n_selector_in: number of input units for the selector
 
    :type n_selector_hidden: a list of ints
    :param n_selector_hidden: this list defines the number of layers for selector (by len(n_selector_hidden))
 
    :type n_group: int
    :param n_group: number of neuron group

    :type dropout_rate: a float lies in between (0, 1)
    :param dropout_rate: the ratio of dropping input for neuron group

    :type dropout_s: a list of floats lying in between (0, 1)
    :param dropout_s: the ratio of dropping input for
"""

class adaptiveHiddenLayer(object):
    def __init__(self, rng, input, dropped_input, n_in, n_hidden,n_selector_hidden=[256, 128], 
                n_group=2, activation=T.nnet.relu, W_n=None, b_n=None, W_s=None, b_s=None, 
                dropout_rate=0, dropout_s=[0.5, 0.5, 0.5], belta=1):
        n_selector_in = n_in
        
        #create shared variables of weights for neuron groups
        if W_n is None:
            W_n = []
            for group_num in range(n_group):
                W_values = numpy.asarray(
                    rng.randn(n_in, n_hidden
                    ) * numpy.sqrt(2.0/(n_in + n_hidden)),
                    dtype=theano.config.floatX
                )
                if activation == theano.tensor.nnet.sigmoid:
                    W_values *= 4
                W_n.extend([theano.shared(value=W_values, name='W'+str(group_num), borrow=True)])

        #create shared variables of bias for neuron groups
        if b_n is None:
            b_n = []
            for group_num in range(n_group):
                b_values = numpy.zeros((n_hidden,), dtype=theano.config.floatX)
                theano.shared(value=b_values, name='b'+str(group_num), borrow=True)
                b_n.extend([theano.shared(value=b_values, name='b'+str(group_num), borrow=True)])

        #create shared variables of weights for the selector
        if W_s is None:
            W_s = []  
            for layer_idx in range(len(n_selector_hidden)):
                if layer_idx == 0 :
                    W_values = numpy.asarray(
                        rng.randn(n_selector_in, n_selector_hidden[0]
                        ) * numpy.sqrt(2.0/(n_selector_in + n_selector_hidden[0])),
                        dtype=theano.config.floatX
                    )
                else:
                    W_values = numpy.asarray(
                        rng.randn(n_selector_hidden[layer_idx - 1], n_selector_hidden[layer_idx]
                        ) * numpy.sqrt(2.0/(n_selector_hidden[layer_idx - 1] + n_selector_hidden[layer_idx])),
                        dtype=theano.config.floatX
                    )
                if activation == theano.tensor.nnet.sigmoid:
                    W_values *= 4
                W_s.extend([theano.shared(value=W_values, name='Ws'+str(layer_idx), borrow=True)])
            if len(n_selector_hidden) != 0 :
                W_values = numpy.zeros(
                    (n_selector_hidden[-1], n_group),
                    dtype=theano.config.floatX
                )
            else:
                W_values = numpy.zeros(
                    (n_in, n_group),
                    dtype=theano.config.floatX
                )
            W_s.extend([theano.shared(value=W_values, name='Wo', borrow=True)])
    
        
        #create shared variables of biases for the selector
        if b_s is None:
            b_s = []
            for layer_idx in range(len(n_selector_hidden)):
                if layer_idx == 0 :
                    b_values = numpy.zeros((n_selector_hidden[0],), dtype=theano.config.floatX)
                else:
                    b_values = numpy.zeros((n_selector_hidden[layer_idx],), dtype=theano.config.floatX)
                b_s.extend([theano.shared(value=b_values, name='bs'+str(layer_idx), borrow=True)])
    
            b_values = numpy.zeros((n_group,), dtype=theano.config.floatX)
            b_s.extend([theano.shared(value=b_values, name='bo', borrow=True)])
    
    


        
        #We construct the model sets. For the same input, we duplicate the 'n_group' times of outputs of those models.
        #And we collectes these layers into self.hidden_layers.  

        self.hidden_layers = []
        self.dropout_hidden_layers = []

        ###preparing dropped input for dropout neuron groups
        srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
        dropout_input = _dropout_from_layer(srng, dropped_input, p=dropout_rate)

        ###constructing neuron groups and its corresponding dropout layers
        if activation is 'softmax':
            for group_num in range(n_group):
                dropoutHiddenLayer = LogisticRegression(
                    input=dropout_input,
                    n_in=n_in,
                    n_out=n_hidden,
                    W=W_n[group_num],
                    b=b_n[group_num],
                )
                hiddenLayer = LogisticRegression(
                    input=input,
                    n_in=n_in,
                    n_out=n_hidden,
                    W=W_n[group_num] * (1 - dropout_rate),
                    b=b_n[group_num],
                )
                self.dropout_hidden_layers.append(dropoutHiddenLayer)
                self.hidden_layers.append(hiddenLayer)
        else:
            for group_num in range(n_group):
                dropoutHiddenLayer = HiddenLayer(
                    rng=rng,
                    input=dropout_input,
                    n_in=n_in,
                    n_out=n_hidden,
                    activation=activation,
                    W=W_n[group_num],
                    b=b_n[group_num],
                )
                hiddenLayer = HiddenLayer(
                    rng=rng,
                    input=input,
                    n_in=n_in,
                    n_out=n_hidden,
                    activation=activation,
                    W=W_n[group_num] * (1 - dropout_rate),
                    b=b_n[group_num],
                )
                self.hidden_layers.append(hiddenLayer)
                self.dropout_hidden_layers.append(dropoutHiddenLayer)

        ###constructing selector
        self.selector_hidden_layers = []
        self.dropout_selector_hidden_layers = []



        if len(n_selector_hidden) == 0 :
            ###preparing dropped input for dropout neuron groups
            dropout_selector_input = _dropout_from_layer(srng, dropped_input, p=dropout_s[0])
            selectorDropoutLogRegressionLayer = LogisticRegression(
                input=dropout_selector_input,
                n_in=n_in,
                n_out=n_group,
                W=W_s[0],
                b=b_s[0]
            )
    
            selectorLogRegressionLayer = LogisticRegression(
                input=input,
                n_in=n_in,
                n_out=n_group,
                W=W_s[0] * (1 - dropout_s[0]),
                b=b_s[0]
            )
            self.selector_hidden_layers.append(selectorLogRegressionLayer)
            self.dropout_selector_hidden_layers.append(selectorDropoutLogRegressionLayer)
        else:
            for layer_idx in range(len(n_selector_hidden)):
                if layer_idx == 0:
                    dropout_selector_input = _dropout_from_layer(srng, dropped_input, p=dropout_s[0])
                else:
                    dropout_selector_input = _dropout_from_layer(srng, self.dropout_selector_hidden_layers[-1].output, p=dropout_s[layer_idx])

                if layer_idx == 0:
                    dropoutSelectorHiddenLayer = HiddenLayer(
                        rng=rng,
                        input=dropout_selector_input,
                        n_in=n_selector_in,
                        n_out=n_selector_hidden[0],
                        activation=T.nnet.relu,
                        W=W_s[layer_idx],
                        b=b_s[layer_idx],
                    )
                    selectorHiddenLayer = HiddenLayer(
                        rng=rng,
                        input=input,
                        n_in=n_selector_in,
                        n_out=n_selector_hidden[0],
                        activation=T.nnet.relu,
                        W=W_s[layer_idx] * (1 - dropout_s[layer_idx]) ,
                        b=b_s[layer_idx],
                    )
                else:
                    dropoutSelectorHiddenLayer = HiddenLayer(
                        rng=rng,
                        input=dropout_selector_input,
                        n_in=n_selector_hidden[layer_idx-1],
                        n_out=n_selector_hidden[layer_idx],
                        activation=T.nnet.relu,
                        W=W_s[layer_idx],
                        b=b_s[layer_idx],
                    )
                    selectorHiddenLayer = HiddenLayer(
                        rng=rng,
                        input=self.selector_hidden_layers[-1].output,
                        n_in=n_selector_hidden[layer_idx-1],
                        n_out=n_selector_hidden[layer_idx],
                        activation=T.nnet.relu,
                        W=W_s[layer_idx] * (1 - dropout_s[layer_idx]) ,
                        b=b_s[layer_idx],
                    )

                self.dropout_selector_hidden_layers.append(dropoutSelectorHiddenLayer)
                self.selector_hidden_layers.append(selectorHiddenLayer)

            ###preparing dropped input for dropout selector
            dropout_selector_input = _dropout_from_layer(srng, self.dropout_selector_hidden_layers[-1].output, p=dropout_s[-1])
            selectorDropoutLogRegressionLayer = LogisticRegression(
                input=dropout_selector_input,
                n_in=n_selector_hidden[-1],
                n_out=n_group,
                W=W_s[-1],
                b=b_s[-1]
            )
    
            selectorLogRegressionLayer = LogisticRegression(
                input=self.selector_hidden_layers[-1].output,
                n_in=n_selector_hidden[-1],
                n_out=n_group,
                W=W_s[-1] * (1 - dropout_s[-1]),
                b=b_s[-1]
            )
            self.selector_hidden_layers.append(selectorLogRegressionLayer)
            self.dropout_selector_hidden_layers.append(selectorDropoutLogRegressionLayer)

            ####ADD hard thersholding
            #self.dropout_selector_hidden_layers[-1].p_y_given_x = hardtanh(self.dropout_selector_hidden_layers[-1].p_y_given_x)

            shape_size = input.shape[0]
            zero_vector = (T.zeros((shape_size, ), dtype='int32'))
            self.params = W_n + W_s + b_n + b_s
            self.params_n = W_n + b_n
            self.params_s = W_s + b_s
            self.W = W_n + W_s
            self.L2_norm = 0
            for w in self.W:
                self.L2_norm = self.L2_norm + (w**2).sum()

            ###SBR penalty
            eplison = 1e-8
            p1 = T.reshape(hardtanh(self.dropout_selector_hidden_layers[-1].p_y_given_x)[T.arange(shape_size), zero_vector + 0] ,newshape = (shape_size,1))
            p2 = T.reshape(hardtanh(self.dropout_selector_hidden_layers[-1].p_y_given_x)[T.arange(shape_size), zero_vector + 1] ,newshape = (shape_size,1))
            denominator = T.sum(self.dropout_selector_hidden_layers[-1].p_y_given_x) + eplison
            p1_r = T.clip(T.sum(T.switch(p1 >= 0.5, p1, 0)) / denominator, eplison, 1)
            p2_r = T.clip(T.sum(T.switch(p2 >= 0.5, p2, 0)) / denominator, eplison, 1)
            self.SBR = -(p1_r * T.log(p1_r) + p2_r * T.log(p2_r)) * belta

             
            ###define network output
            p1_test = T.reshape(self.selector_hidden_layers[-1].p_y_given_x[T.arange(shape_size), zero_vector + 0] ,newshape = (shape_size,1))
            p2_test = T.reshape(self.selector_hidden_layers[-1].p_y_given_x[T.arange(shape_size), zero_vector + 1] ,newshape = (shape_size,1))
            
            ###hard assigning
            #p1_test = T.switch(p1_test >= 0.5, 1, 0)
            #p2_test = T.switch(p2_test > 0.5, 1, 0)

            if activation is 'softmax':
                self.output = p1_test * self.hidden_layers[0].p_y_given_x + p2_test * self.hidden_layers[1].p_y_given_x
                self.dropout_output = p1 * self.dropout_hidden_layers[0].p_y_given_x + p2 * self.dropout_hidden_layers[1].p_y_given_x

                self.p_y_given_x = p1_test * self.hidden_layers[0].p_y_given_x + p2_test * self.hidden_layers[1].p_y_given_x
                self.dropped_p_y_given_x = p1 * self.dropout_hidden_layers[0].p_y_given_x + p2 * self.dropout_hidden_layers[1].p_y_given_x
            else:
                self.output = p1_test * self.hidden_layers[0].output + p2_test * self.hidden_layers[1].output
                self.dropped_output = p1 * self.dropout_hidden_layers[0].output + p2 * self.dropout_hidden_layers[1].output

            self.W_n = W_n
            self.b_n = b_n



            



input = T.matrix('f1')
rng = numpy.random.RandomState(11111)

AHL1 = adaptiveHiddenLayer(rng, input, dropped_input=input, n_in=1024, n_hidden=512, n_selector_hidden=[512, 256], 
                n_group=2, activation=T.nnet.relu, dropout_rate=0.5, dropout_s=[0.5, 0.5, 0.5], belta=1)

print AHL1.L2_norm
print AHL1.SBR
