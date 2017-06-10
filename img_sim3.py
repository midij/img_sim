try:
    import cPickle as pickle
    from urllib2 import urlopen
except ImportError:
    import pickle
    from urllib.request import urlopen

import sys
import os
import random
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.contrib.losses.python.losses import loss_ops
import logging
import os.path

IMG_SIZE = 128

def denseToOneHot(labels_dense, num_classes):
    #Convert class labels from scalars to one-hot vectors.
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


# @deprecated
def loadFileLists():
    dummylist = ['./data/test.jpg']
    return dummylist
	

def loadFeatures(files):
    data = np.ndarray((len(files), IMG_SIZE * IMG_SIZE * 3))
    for n, f in enumerate(files):
        logging.debug('loading file #%d' % n)
        img = cv2.imread(f)
        #print(img.shape)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("orig", img)
        h, w, _ = img.shape
	#print h
	#print w
	
        if w > h:
            diff = w - h
            img = img[:, diff / 2: diff / 2 + h]
        elif w < h:
            diff = h - w
            img = img[diff / 2: diff / 2 + w, :]
	#print(img.shape) 
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data[n] = img.ravel()
        # cv2.imshow("res", img)
        # cv2.waitKey(0)
    return data


def load_dataset_list(datasetconf):
	x1_list = []
	x2_list = []
	label_list =[]
	#in the file: image1 "\t" image2 "\t" label = [0|1]
	with open(datasetconf, 'r') as f:
		for line in f:
			line = line.strip()
			if line=='': continue
			cols = line.split('\t')
			if len(cols) <  3:
				logging.debug('input fields less than 3')
				continue
			if cols[0].strip()== "": 
				continue
			if cols[1].strip()== "": 
				continue
			if cols[2].strip()== "": 
				continue
			#string label to int label
			try:
				val_y = int(cols[2])
			except:
				continue	
			x1_list.append(cols[0])
			x2_list.append(cols[1])
			label_list.append(val_y)

	#print x1_list
	#print x2_list
	#print label_list
	return x1_list, x2_list, label_list
	
def load_dataset(trainlistconf = None, testlistconf = None):
	#load train list:
	#trainlistconf = "train_set_conf.txt"
	#trainlistconf = "train_set_conf_1k.txt"
	if trainlistconf == None:	
		trainlistconf = "train_set_conf_500.txt"
	print trainlistconf 
	x1_list, x2_list, label_list = load_dataset_list(trainlistconf)

	#load test list:
	#testlistconf = "test_set_conf.txt"
	#testlistconf = "test_set_conf_300.txt"
	if testlistconf == None:	
		testlistconf = "test_set_conf_1000.txt"
	print testlistconf
	t_x1_list, t_x2_list, t_label_list = load_dataset_list(testlistconf)
	
	#load features
	x1 = loadFeatures(x1_list)
	x2 = loadFeatures(x2_list)
	#label_list = denseToOneHot(np.array(label_list),2)
	label_list = np.reshape(label_list,(-1,1))

	t_x1 = loadFeatures(t_x1_list)
	t_x2 = loadFeatures(t_x2_list)
	#t_label_list = denseToOneHot(np.array(t_label_list),2)
	t_label_list = np.reshape(t_label_list,(-1,1))

	#print label_list
	#print t_label_list
	return x1, x2, label_list, t_x1, t_x2, t_label_list

def test_load_data():
	logging.info('start loading')
	load_dataset()


##let's define a simple graph here
# net related functions
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	inital = tf.constant(0.1, shape= shape)
	return tf.Variable(initial)

def conv2d(x, W):
	# for strdies[0] = strides[3] = 1, strides width = strides height = 2
	return tf.nn.conv2d(x, W, strides = [1,2,2,1], padding = 'SAME')
	#return tf.nn.conv2d(x, W, strides = [1,2,2,1], padding = 'VALID')
	
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')


def get_accuracy(in_x1, in_x2, ideal_y):
	global y_conv
	y_pre = sess.run(y_conv, feed_dict = {x1:in_x1, x2:in_x2, keep_prob:1})
	correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(ideal_y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	result = sess.run(accuracy, feed_dict={x1: in_x1, x2:in_x2, y:ideal_y, keep_prob:1})
	return result


def get_accuracy_old(in_x1, in_x2, ideal_y):
	global y_conv
	y_pre = sess.run(y_conv, feed_dict = {x1:in_x1, x2:in_x2, keep_prob:1})
	correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(ideal_y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	result = sess.run(accuracy, feed_dict={x1: in_x1, x2:in_x2, y:ideal_y, keep_prob:1})
	return result



#define place holders for inputs
with tf.name_scope("inputs"):
	x1 = tf.placeholder(tf.float32, [None, IMG_SIZE * IMG_SIZE * 3], name ="x1_input")
	x2 = tf.placeholder(tf.float32,  [None, IMG_SIZE * IMG_SIZE * 3], name = "x2_input")
	#y = tf.placeholder(tf.float32, [None, 2])
	y = tf.placeholder(tf.float32,[None,1], name = "y_input")	
	keep_prob = tf.placeholder(tf.float32, name = "keep_prob")


#----------------set up the net of the left side
#define the first conv layer

with tf.name_scope("leftlayers"):
	x1_image = tf.reshape(x1, [-1, IMG_SIZE, IMG_SIZE, 3])
	#W_conv1 = tf.get_variable("W_conv1", [3,3,3,32]) #patch = 3*3, input chanel =3, output chanel = 32
	W_conv1 = weight_variable([3,3,3,32])
	#b_conv1 = tf.get_variable("b_conv1", [32]) 
	b_conv1 = weight_variable([32]) 
	h_conv1 = tf.nn.relu(conv2d(x1_image, W_conv1) + b_conv1) #128 -> 64
	h_pool1 = max_pool_2x2(h_conv1) #64 ->32


	#W_conv2 = tf.get_variable("W_conv2", [3,3,32,64]) #patch = 3*3, input chanel =3, output chanel = 32
	W_conv2 = weight_variable([3,3,32,64])
#b_conv2 = tf.get_variable("b_conv2", [64]) 
	b_conv2 = weight_variable([64])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) #32->16
	h_pool2 = max_pool_2x2(h_conv2)	#16->8

	h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
	h_pool2_flat = tf.Print(h_pool2_flat,[h_pool2_flat], "h_pool2_flat:")

#----------------set up the net of the right side
with tf.name_scope("rightlayers"):
	x2_image = tf.reshape(x2, [-1, IMG_SIZE, IMG_SIZE, 3])

	#define the first conv layer
	#W2_conv1 = tf.get_variable("W2_conv1", [3,3,3,32]) #patch = 3*3, input chanel =3, output chanel = 32
	W2_conv1 = weight_variable([5,5,3,32])
	b2_conv1 = tf.get_variable("b2_conv1", [32]) 
	h2_conv1 = tf.nn.relu(conv2d(x2_image, W2_conv1) + b2_conv1) #128 -> 64
	h2_pool1 = max_pool_2x2(h2_conv1) #64 ->32


	W2_conv2 = tf.get_variable("W2_conv2", [3,3,32,64]) #patch = 3*3, input chanel =3, output chanel = 32
	b2_conv2 = tf.get_variable("b2_conv2", [64]) 
	h2_conv2 = tf.nn.relu(conv2d(h2_pool1, W2_conv2) + b2_conv2) #32->16
	h2_pool2 = max_pool_2x2(h2_conv2)	#16->8
	h2_pool2_flat = tf.reshape(h2_pool2, [-1, 8*8*64])
	h2_pool2_flat = tf.Print(h2_pool2_flat,[h2_pool2_flat], "h2_pool2_flat:")

#---------------set up the combinationlayer
#fc1 layer
'''
W_fc1 = tf.get_variable("W_fc1", [8*8*64,512])
W2_fc1 = tf.get_variable("W2_fc1", [8*8*64,512])
b_fc1 = tf.get_variable("b_fc1", [512])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + tf.matmul(h2_pool2_flat, W2_fc1)+ b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#------------------- test one side output ----------------------
#fc1 layer:
#W_fc1 =tf.get_variable("W_fc1", [8*8*64, 512])
#b_fc1 = tf.get_variable("b_fc1", [512])
#h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#fc2 layer
W_fc2 = tf.get_variable("W_fc2", [512, 2])
b_fc2 = tf.get_variable("b_fc2", [2])
#b_fc2 = tf.Print(b_fc2, [b_fc2], "b_fc2:")
logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
logits = tf.Print(logits, [logits],"logits:")
y_conv = tf.nn.softmax(logits)
y_conv = tf.Print(y_conv, [y_conv], "Y_Conv:")

#---------------- set up the train_step
#cross_entropy = -tf.reduce_sum(y * tf.log(y_conv+1e-50), reduction_indices=[1])
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_conv+1e-50), reduction_indices=[1]))
#cross_entropy = loss_ops.softmax_cross_entropy(logits, y)
#cross_entropy = tf.Print(cross_entropy, [cross_entropy], "cost") #print to the console 
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
'''

#---------------set up the train step 2
l2diff = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(h_pool2_flat,h2_pool2_flat)), reduction_indices = 1))
l2diff = tf.Print(l2diff, [l2diff], "l2diff:")
margin = tf.to_float(1.)
labels = tf.to_float(y)
match_loss = tf.square(l2diff, 'match_term')	
mismatch_loss = tf.maximum(0., tf.sub(margin, tf.square(l2diff)),'mismatch_term')
loss = tf.add(tf.mul(labels, match_loss), tf.mul((1-labels),mismatch_loss), 'loss_add')
loss_mean = tf.reduce_mean(loss)


train_step = tf.train.AdamOptimizer(1e-4).minimize(loss_mean)
#init session
sess = tf.Session()
writer=tf.train.SummaryWriter('logs/', sess.graph)

init = tf.global_variables_initializer()
sess.run(init)


def print_usage():
	print "img_sim.py train_data_conf test_data_conf"


if __name__ == '__main__':
	print_usage()
	train_set_name = None 
	test_set_name = None	
	if len(sys.argv)==3:
		train_set_name = sys.argv[1]
		test_set_name = sys.argv[2]	
	in_x1, in_x2, in_y, t_x1, t_x2, t_y = load_dataset(train_set_name, test_set_name)

	for i in range(200):
	#for i in range(200):
		sess.run(train_step,  feed_dict = {x1:in_x1, x2:in_x2, y:in_y, keep_prob:0.9})
		
		#tmp_y_conv = sess.run(y_conv,  feed_dict = {x1:in_x1, x2:in_x2, y:in_y, keep_prob:0.9})
		#print tmp_y_conv 
		#print(get_accuracy(t_x1, t_x2, t_y))
		if i % 50 ==0:	
			print "%d th turn", i
			#print(get_accuracy(t_x1, t_x2, t_y))
			left_emb = sess.run(h_pool2_flat, feed_dict={x1:t_x1})
			#print "left:"
			#print left_emb

			right_emb = sess.run(h2_pool2_flat, feed_dict ={x2:t_x2}) 
			#print "right:"
			#print right_emb

			tmploss = sess.run(loss_mean,  feed_dict = {x1:t_x1, x2:t_x2, y:t_y, keep_prob:1})
			print tmploss
		

