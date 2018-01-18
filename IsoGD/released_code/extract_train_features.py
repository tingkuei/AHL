import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import io
import sys
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import tensorlayer as tl
import inputs as data
import c3d_clstm as net 
import time
from datetime import datetime
import threading

seq_len = 32
batch_size = 12

num_classes = 249
dataset_name = 'isogr'

curtime = '%s' % datetime.now()
d = curtime.split(' ')[0]
t = curtime.split(' ')[1]
strtime = '%s%s%s-%s%s%s' %(d.split('-')[0],d.split('-')[1],d.split('-')[2], 
                            t.split(':')[0],t.split(':')[1],t.split(':')[2])

x = tf.placeholder(tf.float32, [batch_size, seq_len, 112, 112, 3], name='x')
y = tf.placeholder(tf.int32, shape=[batch_size, ], name='y')
  
sess = tf.InteractiveSession()

networks = net.c3d_clstm(x, num_classes, False, False)
network_pred = tf.nn.softmax(networks.outputs)
network_y_op = tf.argmax(tf.nn.softmax(networks.outputs),1)
network_accu = tf.reduce_mean(tf.cast(tf.equal(tf.cast(network_y_op, tf.int32), y), tf.float32))
network_spp_output = networks.inputs
network_fc_output = networks.outputs
  
sess.run(tf.initialize_all_variables())
#sess.run(tf.global_variables_initializer())
# RGB
training_datalist = '/home/tingkuei/IsoGD_dataset/IsoGD_Image/train_rgb_list.txt'
X_train,y_train = data.load_video_list(training_datalist)
X_tridx = np.asarray(np.arange(0, len(y_train)), dtype=np.int32)
y_train  = np.asarray(y_train, dtype=np.int32)
rgb_prediction = np.zeros((len(y_train),num_classes), dtype=np.float32)
rgb_fc_output = np.zeros((len(y_train),num_classes), dtype=np.float32)
rgb_spp_output = np.zeros((len(y_train), 26880), dtype=np.float32)
load_params = tl.files.load_npz(name='/home/tingkuei/benchmark/IsoGD_pretrain_model/isogr_rgb_model_strategy_3.npz')


np.save('./extracted_feature/train_y.npy', y_train)

tl.files.assign_params(sess, load_params, networks)
networks.print_params(True)
average_accuracy = 0.0
train_iterations = 0

print '%s: rgb training' % datetime.now()
for X_indices, y_label_t in tl.iterate.minibatches(X_tridx, 
                                                   y_train, 
                                                   batch_size, 
                                                   shuffle=False):
  # Read data for each batch      
  image_path = []
  image_fcnt = []
  image_olen = []
  is_training = []
  for data_a in range(batch_size):
    X_index_a = X_indices[data_a]
    key_str = '%06d' % X_index_a
    image_path.append(X_train[key_str]['videopath'])
    image_fcnt.append(X_train[key_str]['framecnt'])
    image_olen.append(seq_len)
    is_training.append(False) # Testing
    image_info = zip(image_path,image_fcnt,image_olen,is_training)
  X_data_t = tl.prepro.threading_data([_ for _ in image_info], 
                                      data.prepare_isogr_rgb_data)
  feed_dict = {x: X_data_t, y: y_label_t}
  dp_dict = tl.utils.dict_to_one(networks.all_drop)
  feed_dict.update(dp_dict)
  predict_value,accu_value = sess.run([network_pred, network_accu], feed_dict=feed_dict)
  spp_output = sess.run(network_spp_output, feed_dict=feed_dict)
  fc_output = sess.run(network_fc_output, feed_dict=feed_dict)
  rgb_spp_output[train_iterations*batch_size:(train_iterations+1)*batch_size,:]=spp_output
  average_accuracy = average_accuracy + accu_value
  train_iterations = train_iterations + 1
average_accuracy = average_accuracy / train_iterations
format_str = ('%s: rgb average_accuracy = %.6f')
print (format_str % (datetime.now(), average_accuracy))
print rgb_prediction.shape
np.save('./extracted_feature/rgb_train_spp_output.npy', rgb_spp_output)

# Depth
training_datalist = '/home/tingkuei/IsoGD_dataset/IsoGD_Image/train_depth_list.txt'
X_train,y_train = data.load_video_list(training_datalist)
X_tridx = np.asarray(np.arange(0, len(y_train)), dtype=np.int32)
y_train  = np.asarray(y_train, dtype=np.int32)
depth_spp_output = np.zeros((len(y_train),26880), dtype=np.float32)
load_params = tl.files.load_npz(name='/home/tingkuei/benchmark/IsoGD_pretrain_model/isogr_depth_model_strategy_3.npz')
tl.files.assign_params(sess, load_params, networks)
networks.print_params(True)
average_accuracy = 0.0
train_iterations = 0
print '%s: depth training' % datetime.now()
for X_indices, y_label_t in tl.iterate.minibatches(X_tridx, 
                                                   y_train, 
                                                   batch_size, 
                                                   shuffle=False):
  # Read data for each batch      
  image_path = []
  image_fcnt = []
  image_olen = []
  is_training = []
  for data_a in range(batch_size):
    X_index_a = X_indices[data_a]
    key_str = '%06d' % X_index_a
    image_path.append(X_train[key_str]['videopath'])
    image_fcnt.append(X_train[key_str]['framecnt'])
    image_olen.append(seq_len)
    is_training.append(False) # Testing
    image_info = zip(image_path,image_fcnt,image_olen,is_training)
  X_data_t = tl.prepro.threading_data([_ for _ in image_info], 
                                      data.prepare_isogr_depth_data)
  feed_dict = {x: X_data_t, y: y_label_t}
  dp_dict = tl.utils.dict_to_one(networks.all_drop)
  feed_dict.update(dp_dict)
  predict_value,accu_value = sess.run([network_pred, network_accu], feed_dict=feed_dict)
  spp_output = sess.run(network_spp_output, feed_dict=feed_dict)
  fc_output = sess.run(network_fc_output, feed_dict=feed_dict)
  depth_prediction[train_iterations*batch_size:(train_iterations+1)*batch_size,:]=predict_value
  depth_spp_output[train_iterations*batch_size:(train_iterations+1)*batch_size,:]=spp_output
  depth_fc_output[train_iterations*batch_size:(train_iterations+1)*batch_size,:]=fc_output
  average_accuracy = average_accuracy + accu_value
  train_iterations = train_iterations + 1
average_accuracy = average_accuracy / train_iterations
format_str = ('%s: depth average_accuracy = %.6f')
print (format_str % (datetime.now(), average_accuracy))
np.save('./extracted_feature/depth_train_spp_output.npy', depth_spp_output)



fusion_prediction = rgb_prediction + depth_prediction
prediction_values = tf.argmax(fusion_prediction, 1)
final_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(prediction_values, tf.int32), y_train), tf.float32))
print final_accuracy.eval()
# In the end, close TensorFlow session.
sess.close()
