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
import dataloader
from time import gmtime, strftime

IMG_SIZE = 128

def denseToOneHot(labels_dense, num_classes):
    #Convert class labels from scalars to one-hot vectors.
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


	

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
	

#input: labbel_type = onehot or scalar
# it will return you : [0,1] or 1, value for a label
def list_to_data(inlist, label_type = "onehot"):
	x1_list = []
	x2_list = []
	label_list = []
	#split line into lists
	for line in inlist:
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

		#print cols[0]
		#print cols[1]
		#print (val_y)
	x1 = loadFeatures(x1_list)
	x2 = loadFeatures(x2_list)
	if label_type == "onehot":
		label_list = denseToOneHot(np.array(label_list),2)
	elif label_type == "scalar":
		label_list = np.reshape(label_list,(-1,1))
	
	return x1, x2, label_list

def list_to_predict_data(inlist):
	x1_list = []
	x2_list = []
	for line in inlist:
		line = line.strip()
		if line == "": continue
		cols = line.split('\t')
		if len(cols) < 3:
			logging.debug('input fields less than 2')
			continue
		if cols[0].strip() == "":
			continue
		if cols[1].strip() == "":
			continue
		x1_list.append(cols[0])
		x2_list.append(cols[1])

	x1 = loadFeatures(x1_list)
	x2 = loadFeatures(x2_list)
	return x1, x2	
	


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
	global logits 
	y_pre = sess.run(logits, feed_dict = {x1:in_x1, x2:in_x2, keep_prob:1})
	correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(ideal_y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	result = sess.run(accuracy, feed_dict={x1: in_x1, x2:in_x2, y:ideal_y, keep_prob:1})
	return result




#define place holders for inputs
with tf.name_scope("inputs"):
	x1 = tf.placeholder(tf.float32, [None, IMG_SIZE * IMG_SIZE * 3], name ="x1_input")
	x2 = tf.placeholder(tf.float32,  [None, IMG_SIZE * IMG_SIZE * 3], name = "x2_input")
	#y = tf.placeholder(tf.float32, [None, 2])
	#y = tf.placeholder(tf.float32,[None,1], name = "y_input")	
	y = tf.placeholder(tf.float32,[None,2], name = "y_input")	
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
	#h_pool2_flat = tf.Print(h_pool2_flat,[h_pool2_flat], "h_pool2_flat:")
	#fc1 layer:
	W_fc1 =tf.get_variable("W_fc1", [8*8*64, 512])
	b_fc1 = tf.get_variable("b_fc1", [512])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


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
	#h2_pool2_flat = tf.Print(h2_pool2_flat,[h2_pool2_flat], "h2_pool2_flat:")

	#fc1  layer
	W2_fc1 = tf.get_variable("W_fc2", [8*8*64, 512])
	b2_fc1 = tf.get_variable("b_fc2", [512])
	h2_fc1 = tf.nn.relu(tf.matmul(h2_pool2_flat, W2_fc1) + b2_fc1)
	h2_fc1_drop = tf.nn.dropout(h2_fc1, keep_prob)

#---------------set up the combinationlayer
# combine layer
with tf.name_scope("combinationlayer"):
	W_cb1 = tf.get_variable("W_cb1", [8*8*64,512])
	W2_cb1 = tf.get_variable("W2_cb1", [8*8*64,512])
	b_cb1 = tf.get_variable("b_cb1", [512])
	h_cb1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_cb1) + tf.matmul(h2_pool2_flat, W2_cb1)+ b_cb1)
	h_cb1_drop = tf.nn.dropout(h_cb1, keep_prob)


#fc2 layer
with tf.name_scope("softmaxlayer"):
	W_sf1 = tf.get_variable("W_sf1", [512, 2])
	b_sf1 = tf.get_variable("b_sf1", [2])
	logits = tf.matmul(h_cb1_drop, W_sf1) + b_sf1
	#logits = tf.Print(logits, [logits],"logits:")
	#y_conv = tf.nn.softmax(logits)
	#y_conv = tf.Print(y_conv, [y_conv], "y_Conv:")



#---------------- set up the train_step
#cross_entropy = -tf.reduce_sum(y_conv * tf.log(y), reduction_indices=[1])
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_conv+1e-50), reduction_indices=[1]))
cross_entropy = loss_ops.softmax_cross_entropy(logits, y)
cross_entropy = tf.Print(cross_entropy, [cross_entropy], "cost:") #print to the console 
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#---------------set up the train step 2, loss type 1: L2 distance
'''
l2diff = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(h_pool2_flat,h2_pool2_flat)), reduction_indices = 1))
l2diff = tf.Print(l2diff, [l2diff], "l2diff:")
margin = tf.to_float(1.)
labels = tf.to_float(y)
match_loss = tf.square(l2diff, 'match_term')	
mismatch_loss = tf.maximum(0., tf.sub(margin, tf.square(l2diff)),'mismatch_term')
loss = tf.add(tf.mul(labels, match_loss), tf.mul((1-labels),mismatch_loss), 'loss_add')
loss_mean = tf.reduce_mean(loss)


train_step = tf.train.AdamOptimizer(1e-4).minimize(loss_mean)
'''
#---------------- set up the train_step, loss type 2: cross entropy of two vectores.



#init session
sess = tf.Session()
writer=tf.train.SummaryWriter('logs/', sess.graph)

init = tf.global_variables_initializer()
sess.run(init)


def print_usage():
	print "%s train"%sys.argv[0]
	print "%s predict"&sys.argv[0]
	sys.exit(42)

def train_with_epoch():
	#loader = dataloader.DataLoader("dataset_conf_path.txt")
	#loader = dataloader.DataLoader("face_dataset_conf_path.txt")

	img_path= "./data/image_face_v0/images_face/"
	loader = dataloader.DataLoader("image_face_v0_list.txt",img_path)
	loader.load_list()
	epoch_num = 200 
	iter_per_epoch = 10

	test_list = loader.next_epoch_list(500,500)
	t_x1, t_x2, t_y = list_to_data(test_list, label_type = "onehot")

	for i in range(epoch_num):
		print "epoch %d"%i
		train_list = loader.next_epoch_list(100,100)
		print loader.pos_idx, loader.neg_idx

		in_x1, in_x2, in_y = list_to_data(train_list, label_type = "onehot")

		if i % 1 == 0:
			print "report on %d th turn"%i
			print"on trained before:", (get_accuracy(in_x1, in_x2, in_y))
		# train the same epoch for 20 times
		#for j in range(15):	
		for j in range(iter_per_epoch):
			sess.run(train_step,  feed_dict = {x1:in_x1, x2:in_x2, y:in_y, keep_prob:0.5})
		if i %1 == 0:
			print"on test:", (get_accuracy(t_x1, t_x2, t_y))
			print"on trained after:", (get_accuracy(in_x1, in_x2, in_y))
	# test do prediction here
	predict(test_list)

def save_model(model_file_str=None):
	saver = tf.train.Saver()
	tt = strftime("%Y-%m-%d_%H_%M_%S", gmtime())
	filename = "nets/save_net_"+ tt +".ckpt"
	if model_file_str !=  None:
		filename = model_file_str
	save_path = saver.save(sess,filename) 
	print("trained model saved to:", save_path)

def load_model(netfilestr):
	saver = tf.train.Saver()
	saver.restore(sess, netfilestr)
	print("weights:", sess.run(W_conv1))
	print("biases:", sess.run(b_conv1))

def predict(pairlist):	
	for pair in pairlist:
		pre_x1, pre_x2 = list_to_predict_data([pair])
		print pair
		print ("predict results:",  sess.run(logits, feed_dict = {x1:pre_x1, x2:pre_x2, keep_prob:1}))
	

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print_usage()
	mode = sys.argv[1]
	if mode == "train":
		train_with_epoch()
		save_model()
	elif mode == "predict":
		load_model("nets/save_net_2017-06-24_19_30_32.ckpt")	
		#predict()	
	sys.exit()
	
