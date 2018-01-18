from stackAHLs import stackAHLs

import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano import shared


from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from dA import dA
from rbm import RBM
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import PIL.Image as Image
from os.path import expanduser 
import cPickle

train_set_feature1 = numpy.load('../extracted_feature/rgb_train_spp_output.npy')
train_set_feature2 = numpy.load('../extracted_feature/depth_train_spp_output.npy')
train_set_y = numpy.load('../extracted_feature/train_y.npy')



valid_set_feature1 = numpy.load('../extracted_feature/rgb_valid_spp_output.npy')
valid_set_feature2 = numpy.load('../extracted_feature/depth_valid_spp_output.npy')
valid_set_y = numpy.load('../extracted_feature/valid_y.npy')


train_size = train_set_feature1.shape[0]
valid_size = valid_set_feature1.shape[0]

print train_size

## start training the model
batch_size = 13
finetune_lr=0.001
n_group = 2
class_num = 249


# numpy random generator
numpy_rng = numpy.random.RandomState(11111)

print '... building train model'
dropout_rate_rgb = 0.5
dropout_rate_depth = 0.5
dropout_rate_joint = 0.9
dropout_s_rgb=[0.5, 0.5, 0.5]
dropout_s_depth=[0.5, 0.5, 0.5]
dropout_s_joint=[0.8, 0.7, 0.6]
 

mulmod_sda =stackAHLs(
            numpy_rng,
            n_ins_rgb=26880,
            n_ins_depth=26880,
            n_hidden_rgb=512,
            n_hidden_depth=512,
            n_selector_hidden_rgb=[256, 128],
            n_selector_hidden_depth=[256, 128],
            n_selector_hidden_joint=[256, 128],
            dropout_rate_rgb=dropout_rate_rgb,
            dropout_rate_depth=dropout_rate_depth,
            dropout_rate_joint=dropout_rate_joint,
            dropout_s_rgb=dropout_s_rgb,
            dropout_s_depth=dropout_s_depth,
            dropout_s_joint=dropout_s_joint,
            n_group=2,
            n_class=249,
            weight_decay=0.00004,
            activation=T.nnet.relu
)

## set intial weight via pre-trained model
f = open('rgb_model', 'rb')
rgb_layer0_model = cPickle.load(f)
f.close()
f = open('depth_model', 'rb')
depth_layer0_model= cPickle.load(f)
f.close()

f = open('joint_model', 'rb')
joint_layer0_model= cPickle.load(f)
f.close()

for i in range(len(mulmod_sda.rgb_AHL.params)):
    mulmod_sda.rgb_AHL.params[i].set_value(rgb_layer0_model.joint_AHL.params[i].get_value())
    print  rgb_layer0_model.joint_AHL.params[i].get_value().shape

mulmod_sda.rgb_AHL.params[0].set_value(rgb_layer0_model.joint_AHL.params[0].get_value() * (0.5 / (1 - dropout_rate_rgb)))
mulmod_sda.rgb_AHL.params[1].set_value(rgb_layer0_model.joint_AHL.params[1].get_value() * (0.5 / (1 - dropout_rate_rgb)))
mulmod_sda.rgb_AHL.params[2].set_value(rgb_layer0_model.joint_AHL.params[2].get_value() * (0.5 / (1 - dropout_s_rgb[0])))
mulmod_sda.rgb_AHL.params[3].set_value(rgb_layer0_model.joint_AHL.params[3].get_value() * (0.5 / (1 - dropout_s_rgb[1])))
mulmod_sda.rgb_AHL.params[4].set_value(rgb_layer0_model.joint_AHL.params[4].get_value() * (0.5 / (1 - dropout_s_rgb[2])))




for i in range(len(mulmod_sda.depth_AHL.params)):
    mulmod_sda.depth_AHL.params[i].set_value(depth_layer0_model.joint_AHL.params[i].get_value())
    print  depth_layer0_model.joint_AHL.params[i].get_value().shape

mulmod_sda.depth_AHL.params[0].set_value(depth_layer0_model.joint_AHL.params[0].get_value() * (0.5 / (1 - dropout_rate_depth)))
mulmod_sda.depth_AHL.params[1].set_value(depth_layer0_model.joint_AHL.params[1].get_value() * (0.5 / (1 - dropout_rate_depth)))
mulmod_sda.depth_AHL.params[2].set_value(depth_layer0_model.joint_AHL.params[2].get_value() * (0.5 / (1 - dropout_s_depth[0])))
mulmod_sda.depth_AHL.params[3].set_value(depth_layer0_model.joint_AHL.params[3].get_value() * (0.5 / (1 - dropout_s_depth[1])))
mulmod_sda.depth_AHL.params[4].set_value(depth_layer0_model.joint_AHL.params[4].get_value() * (0.5 / (1 - dropout_s_depth[2])))




for i in range(len(mulmod_sda.joint_AHL.params)):
    mulmod_sda.joint_AHL.params[i].set_value(joint_layer0_model.joint_AHL.params[i].get_value())
    print  joint_layer0_model.joint_AHL.params[i].get_value().shape

mulmod_sda.joint_AHL.params[0].set_value(joint_layer0_model.joint_AHL.params[0].get_value() * (0.5 / (1 - dropout_rate_joint)))
mulmod_sda.joint_AHL.params[1].set_value(joint_layer0_model.joint_AHL.params[1].get_value() * (0.5 / (1 - dropout_rate_joint)))




#define validation model and training loss


index = T.lscalar('index')
y = T.ivector('y')

validate_model = theano.function(
        inputs=[mulmod_sda.x1, mulmod_sda.x2, mulmod_sda.y],
        outputs=mulmod_sda.errors,
    )


training_loss = theano.function(
        inputs=[mulmod_sda.x1, mulmod_sda.x2, mulmod_sda.y],
        outputs=mulmod_sda.errors,
    )


train_fn_n, train_fn_s, train_fn_all, decay_learning, decay_selection_penalty  = mulmod_sda.build_finetune_functions(
    train_set_feature1 = train_set_feature1,
    train_set_feature2 = train_set_feature2,
    train_set_y  = train_set_y,
    batch_size=batch_size,
    initial_learning_rate=finetune_lr
)
print '... starting training'
n_train_batches =  int(train_size / batch_size * 1.0)
n_valid_batches =  int(valid_size / batch_size * 1.0)

# early-stopping parameters
patience = 30 * n_train_batches  # look as this many examples regardless
patience_increase = 2.  # wait this much longer when a new best is
                        # found
improvement_threshold = 0.999  # a relative improvement of this much is
                               # considered significant
validation_frequency = min(n_train_batches, patience / 2)
                               # go through this many
                               # minibatche before checking the network
                               # on the validation set; in this case we
                               # check every epoch
start_time = timeit.default_timer()
    
done_looping = False
epoch = 0 
training_epochs = 80
best_validation_loss = numpy.inf


    

while (epoch < training_epochs) and (not done_looping):
    epoch = epoch + 1
    c = []
    minibatch_cost = []
    for minibatch_index in xrange(n_train_batches):
        minibatch_cost = train_fn_all(train_set_feature1[minibatch_index * batch_size:(minibatch_index + 1) * batch_size], train_set_feature2[minibatch_index * batch_size:(minibatch_index + 1) * batch_size], train_set_y[minibatch_index * batch_size:(minibatch_index + 1) * batch_size])
        c.append(minibatch_cost)
        iter = (epoch - 1) * n_train_batches + minibatch_index
        if (iter+1) % 15000 == 0:
            decay_learning()

        if (iter + 1) % validation_frequency == 0:
            validation_losses = [validate_model(valid_set_feature1[i * batch_size:(i + 1) * batch_size], valid_set_feature2[i * batch_size:(i + 1) * batch_size], valid_set_y[i * batch_size:(i + 1) * batch_size]) for i
                                     in xrange(n_valid_batches)]
            training_losses = [training_loss(train_set_feature1[i * batch_size:(i + 1) * batch_size], train_set_feature2[i * batch_size:(i + 1) * batch_size], train_set_y[i * batch_size:(i + 1) * batch_size])for i
                                     in xrange(n_train_batches)]
            this_validation_loss = numpy.mean(validation_losses)
            this_training_loss = numpy.mean(training_losses)
            
            print('epoch %i, minibatch %i/%i, validation accu %f %%, total cost %f training_error %f %%' %
                  (epoch, minibatch_index + 1, n_train_batches,
                   (100 - this_validation_loss * 100.), numpy.mean(c), this_training_loss * 100))
            text_file = open('log.txt', 'a')
            text_file.write(str((100 - this_validation_loss * 100.))+'\n')
            text_file.close()
            from six.moves import cPickle
            f = open('joint_learned best_model_v1', 'wb')
            cPickle.dump(mulmod_sda, f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()

             
            ### suffle train data
            random_list = numpy.random.permutation(train_size)
            train_set_feature1 = train_set_feature1[random_list]
            train_set_feature2 = train_set_feature2[random_list]
            train_set_y  = train_set_y[random_list]
