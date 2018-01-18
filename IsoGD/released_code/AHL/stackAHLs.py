
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import shared


from logistic_sgd import LogisticRegression, load_data, DropoutLogisticRegression
from mlp import HiddenLayer
from mlp import DropoutHiddenLayer
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import PIL.Image as Image
from os.path import expanduser 
import cPickle
from theano.ifelse import ifelse
from collections import OrderedDict
from mlp import _dropout_from_layer
from AHL import adaptiveHiddenLayer

# start-snippet-1
class stackAHLs(object):
    def __init__(
        self,
        numpy_rng,
        n_ins_rgb=26880,
        n_ins_depth=26880,
        n_hidden_rgb=512,
        n_hidden_depth=512,
        n_selector_hidden_rgb=[256, 128],
        n_selector_hidden_depth=[256, 128],
        n_selector_hidden_joint=[256, 128],
        dropout_rate_rgb=0.5,
        dropout_rate_depth=0.5,
        dropout_rate_joint=0.5,
        dropout_s_rgb=[0.5, 0.5, 0.5],
        dropout_s_depth=[0.5, 0.5, 0.5],
        dropout_s_joint=[0.5, 0.5, 0.5],
        n_group=2,
        n_class=249,
        weight_decay=0.00004,
        activation=T.nnet.relu,
        selection_penalty=1
    ):
        self.prediction1 = T.matrix('x1')
        self.prediction2 = T.matrix('x2')
        self.x1 = T.matrix('f1')
        self.x2 = T.matrix('f2')
        self.y = T.ivector('y')   # the labels are presented as 1D vector of
 

        self.selection_penalty = theano.shared(numpy.asarray(selection_penalty, dtype=theano.config.floatX))

        self.params = []
        self.paramsForNeuronGroup=[]
        self.paramsForSelector=[]
        self.W = []

        rng = numpy_rng
        srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))


        dropout_input_rgb = _dropout_from_layer(srng, self.x1, p=dropout_rate_rgb)
        dropout_input_depth = _dropout_from_layer(srng, self.x2, p=dropout_rate_rgb)
        self.rgb_AHL = adaptiveHiddenLayer(rng=rng, 
                                        input=self.x1, 
                                        dropped_input=self.x1, 
                                        n_in=n_ins_rgb, 
                                        n_hidden=n_hidden_rgb, 
                                        n_selector_hidden=n_selector_hidden_rgb, 
                                        n_group=2, 
                                        activation=T.nnet.relu, 
                                        dropout_rate=dropout_rate_rgb, 
                                        dropout_s=dropout_s_rgb, 
                                        belta=1)
        self.params.extend(self.rgb_AHL.params)
        self.paramsForNeuronGroup.extend(self.rgb_AHL.params_n)
        self.paramsForSelector.extend(self.rgb_AHL.params_s)



        self.depth_AHL = adaptiveHiddenLayer(rng=rng, 
                                        input=self.x2, 
                                        dropped_input=self.x2, 
                                        n_in=n_ins_depth, 
                                        n_hidden=n_hidden_depth, 
                                        n_selector_hidden=n_selector_hidden_depth, 
                                        n_group=2, 
                                        activation=T.nnet.relu, 
                                        dropout_rate=dropout_rate_depth, 
                                        dropout_s=dropout_s_depth, 
                                        belta=1)

        self.params.extend(self.depth_AHL.params)
        self.paramsForNeuronGroup.extend(self.depth_AHL.params_n)
        self.paramsForSelector.extend(self.depth_AHL.params_s)

        
        joint_input = T.concatenate([self.rgb_AHL.output, self.depth_AHL.output], axis=1)     
        joint_dropout_input = T.concatenate([self.rgb_AHL.dropped_output, self.depth_AHL.dropped_output], axis=1)
     


        self.joint_AHL = adaptiveHiddenLayer(rng=rng, 
                                        input=joint_input, 
                                        dropped_input=joint_dropout_input, 
                                        n_in=n_hidden_rgb + n_hidden_depth, 
                                        n_hidden=n_class, 
                                        n_selector_hidden=n_selector_hidden_joint, 
                                        n_group=2, 
                                        activation='softmax', 
                                        dropout_rate=dropout_rate_joint, 
                                        dropout_s=dropout_s_joint, 
                                        belta=1)
        self.params = self.joint_AHL.params + self.depth_AHL.params + self.rgb_AHL.params
        self.paramsForNeuronGroup = self.joint_AHL.params_n + self.depth_AHL.params_n + self.rgb_AHL.params_n
        self.paramsForSelector = self.joint_AHL.params_s + self.depth_AHL.params_s + self.rgb_AHL.params_s
        cost1 = T.log(self.joint_AHL.dropped_p_y_given_x[T.arange(self.y.shape[0]), self.y])  
        cost3 = self.joint_AHL.SBR + self.depth_AHL.SBR + self.rgb_AHL.SBR
        L2_norm = self.joint_AHL.L2_norm + self.depth_AHL.L2_norm + self.rgb_AHL.L2_norm
        self.y_pred = T.argmax(self.joint_AHL.p_y_given_x, axis=1)
        self.dropout_finetune_cost = -T.mean(cost1) +  weight_decay * L2_norm  - cost3 
        self.errors = T.mean(T.neq(self.y_pred, self.y))

        # end-snippet-4
    def build_finetune_functions(self, train_set_feature1, train_set_feature2, train_set_y, batch_size, initial_learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on
        a batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                         the has to contain three pairs, `train`,
                         `valid`, `test` in this order, where each pair
                         is formed of two Theano variables, one for the
                         datapoints, the other for the labels

        :type batch_size: int
        :param batch_size: size of a minibatch

        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        '''

        # compute number of minibatches for training, validation and testing

        index = T.lscalar('index')  # index to a [mini]batch
        epoch = T.scalar('epoch')   # index to current epoch
        mom_start = 0.5
        mom_end = 0.9
        mom = 0.9
        mom_epoch_interval = 10000
        learning_rate_decay = 0.5
        selection_penalty_decay = 0.5
             
        #define decay learning rate
        learning_rate = theano.shared(numpy.asarray(initial_learning_rate,
        dtype=theano.config.floatX))
        decay_learning_rate = theano.function(inputs=[], outputs=learning_rate,
            updates={learning_rate: learning_rate * learning_rate_decay})

        decay_selection_penalty= theano.function(inputs=[], outputs=self.selection_penalty,
            updates={self.selection_penalty: self.selection_penalty * selection_penalty_decay})


        params_all = self.params
        params_n = self.paramsForNeuronGroup
        params_s = self.paramsForSelector
        deltas_all = []
        deltas_n = []
        deltas_s = []
        for param in params_n:
            delta = theano.shared(numpy.zeros(param.get_value(borrow=True).shape,
                dtype=theano.config.floatX))
            deltas_n.append(delta)
        for param in params_s:
            delta = theano.shared(numpy.zeros(param.get_value(borrow=True).shape,
                dtype=theano.config.floatX))
            deltas_s.append(delta)

        for param in params_all:
            delta = theano.shared(numpy.zeros(param.get_value(borrow=True).shape,
                dtype=theano.config.floatX))
            deltas_all.append(delta)

        # compute the gradients with respect to the model parameters
        gparams_n = T.grad(self.dropout_finetune_cost, params_n)
        gparams_s = T.grad(self.dropout_finetune_cost, params_s)
        gparams_all = T.grad(self.dropout_finetune_cost, params_all)

        updates_n = OrderedDict()
        updates_s = OrderedDict()
        updates_all = OrderedDict()
        for w, d, g in zip(params_n, deltas_n, gparams_n):
            updates_n[d] =  mom * d - learning_rate * g
            updates_n[w] =  w + mom * mom * d - (1 + mom) * learning_rate * g
        for w, d, g in zip(params_s, deltas_s, gparams_s):
            updates_s[d] =  mom * d - learning_rate * g * 1
            updates_s[w] =  w + mom * mom * d - (1 + mom) * learning_rate * g * 1
        for w, d, g in zip(params_s, deltas_s, gparams_s):
            updates_all[d] =  mom * d - learning_rate * g * 0.1
            updates_all[w] =  w + mom * mom * d - (1 + mom) * learning_rate * g * 0.1




        train_fn_n = theano.function(
            inputs=[self.x1, self.x2, self.y],
            outputs=self.dropout_finetune_cost,
            updates=updates_n,
            name='train'
        )

        train_fn_s = theano.function(
            inputs=[self.x1, self.x2, self.y],
            outputs=self.dropout_finetune_cost,
            updates=updates_s,
            name='train'
        )
        train_fn_all = theano.function(
            inputs=[self.x1, self.x2, self.y],
            outputs=self.dropout_finetune_cost,
            updates=updates_all,
            name='train'
        )
        

        return train_fn_n, train_fn_s, train_fn_all, decay_learning_rate, decay_selection_penalty
