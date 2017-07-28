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
import time
from tensorflow.python.saved_model import builder as saved_model_builder

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
	

#input: labbel_type = onehot or scalar or scalar_revert
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
	elif label_type == "scalar_revert":
		#label_list[:] = [1-x for x in label_list]
		#label_list = np.reshape(label_list,(-1,1))
		pass
	return x1, x2, label_list

def list_to_predict_data(inlist):
	x1_list = []
	x2_list = []
	for line in inlist:
		line = line.strip()
		if line == "": continue
		cols = line.split('\t')
		if len(cols) < 2:
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

def weight_get_variable(name, shape):
	initer = tf.truncated_normal_initializer(stddev=0.01)
	return tf.get_variable(name, dtype = tf.float32, shape=shape, initializer = initer)
	#return tf.get_variable(initial)


def bias_variable(shape):
	inital = tf.constant(0.1, shape= shape)
	return tf.Variable(initial)

def bias_get_variable(name, shape):
	return tf.get_variable(name, dtype=tf.float32, initializer=tf.constant(0.01, shape=shape, dtype=tf.float32))
	

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

def get_siamese_accuracy(eucd, labels):
	eucd_label = [0 if x > 0.5 else 1 for x in eucd]
	correct = [1 if x == y else 0 for x, y in zip(eucd_label,labels)]
	return np.mean(correct)



def conv_relu_maxpool(input, kernel_shape, bias_shape, name):
	W = weight_get_variable(name + "_W",kernel_shape)
	b = bias_get_variable(name + "_b", bias_shape) 
	h = tf.nn.relu(conv2d(input, W) + b) 
	#pool = max_pool_2x2(h) 
	pool = h # remove max pooling
	return pool

def fc_layer(input, weight_shape, bias_shape, name):
	W = weight_get_variable(name + "_W", weight_shape)
	b = bias_get_variable(name + "_b", bias_shape)
	fc = tf.matmul(input, W) + b
	return fc	

#define place holders for inputs
with tf.name_scope("inputs"):
	x1 = tf.placeholder(tf.float32, [None, IMG_SIZE * IMG_SIZE * 3], name ="x1_input")
	x2 = tf.placeholder(tf.float32,  [None, IMG_SIZE * IMG_SIZE * 3], name = "x2_input")

	#y = tf.placeholder(tf.float32,[None,1], name = "y_input")	
	y = tf.placeholder(tf.float32,[None], name = "y_input")	
	#y = tf.placeholder(tf.float32,[None,2], name = "y_input")	
	keep_prob = tf.placeholder(tf.float32, name = "keep_prob")


#----------------set up the net of the left side
#define the first conv layer

with tf.name_scope("leftlayers"):
	x1_image = tf.reshape(x1, [-1, IMG_SIZE, IMG_SIZE, 3])
	#kernel = 3*3, input chanel =3, output chanel = 32
	#stride = 2,2
	h_pool1 = conv_relu_maxpool(x1_image, [3,3,3,32],[32], "conv11") #128 -> 64
	h_pool2 = conv_relu_maxpool(h_pool1, [3,3,32,64],[64], "conv12") # 64-> 32
	h_pool2_flat = tf.reshape(h_pool2, [-1, 32*32*64])
	#h_pool2_flat = tf.Print(h_pool2_flat,[h_pool2_flat], "h_pool2_flat:")

	#fc1 layer:
	h_fc1 = fc_layer(h_pool2_flat, [32*32*64, 512],[512], "fc11")
	h_fc1 = tf.nn.relu(h_fc1)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#----------------set up the net of the right side
with tf.name_scope("rightlayers"):
	x2_image = tf.reshape(x2, [-1, IMG_SIZE, IMG_SIZE, 3])

	h2_pool1 = conv_relu_maxpool(x2_image, [3,3,3,32],[32], "conv21") #128 -> 64
	h2_pool2 = conv_relu_maxpool(h2_pool1, [3,3,32,64],[64], "conv22") # 64-> 32

	h2_pool2_flat = tf.reshape(h2_pool2, [-1, 32*32*64])
	#h_pool2_flat = tf.Print(h_pool2_flat,[h_pool2_flat], "h_pool2_flat:")
	#fc1 layer:
	h2_fc1 = fc_layer(h2_pool2_flat, [32*32*64, 512],[512], "fc21")
	h2_fc1 = tf.nn.relu(h2_fc1)
	h2_fc1_drop = tf.nn.dropout(h2_fc1, keep_prob)


'''
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
'''


'''
#---------------- set up the train_step
#---------------- loss functio is entropy 
# this loss is saved as the branch_for_entropy_loss branch
# can be deprecated
#cross_entropy = -tf.reduce_sum(y_conv * tf.log(y), reduction_indices=[1])
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_conv+1e-50), reduction_indices=[1]))
cross_entropy = loss_ops.softmax_cross_entropy(logits, y)
cross_entropy = tf.Print(cross_entropy, [cross_entropy], "cost:") #print to the console 
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
'''


#-------------- set up the train_step
#-------------- using the siamese net loss
# ------------- define: Euclidean distance: eucd = sqrt(||cnn_embed(x1)-cnn_embed(x2)||^2)
#-------------- loss = label * eucd(x1,x2)^2 + (1 - label) * max (0, (c-eucd(x1,x2)))^2
#-------------- input label = 0, if x1 and x2 is same, label = 1 if x1 and x2 is not same

with tf.name_scope("loss"):
	margin = 1.0  # define the C. could be 5.0 or 1.0
	# y should be in the format of 0 or 1, not onehot.
	#y_t = y
	#y_f = tf.sub(1.0, y_t, name = "1-y_t")
	#y_f = tf.Print(y_f, [y_f], "print_1-y_t")
	eucd2 = None
	if int((tf.__version__).split('.')[1]) <12 and int((tf.__version__).split('.')[0]) <1: #tensorflow version <0.12
		#eucd2 = tf.pow(tf.sub(h_pool2_flat, h2_pool2_flat),2) #should try dropout next time
		eucd2 = tf.pow(tf.sub(h_fc1, h2_fc1),2) #should try dropout next time
	else:
		#eucd2 = tf.pow(tf.subtract(h_pool2_flat, h2_pool2_flat),2) #should try dropout next time
		eucd2 = tf.pow(tf.subtract(h_fc1, h2_fc1),2) #should try dropout next time

	eucd2 = tf.reduce_sum(eucd2, 1)
	#eucd2 = tf.Print(eucd2,[eucd2], "print_eucd2")
	eucd = tf.sqrt(eucd2 + 1e-6, name = "eucd")
	
	C = tf.constant(margin, name = "C")
	#pos = tf.mul(1-y_t, eucd2, name = "yi_x_eucd2") # the first half of the loss
	#neg = tf.mul(y_t, tf.pow(tf.maximum(tf.sub(C, eucd),0),2), name = "Nyi_x_C-eucd_xx_2") # the second half of the loss
	
	losses = y * eucd2 + (1-y) * tf.square(tf.maximum(0., margin - eucd))
	#losses = y * eucd2 + (1-y) * tf.maximum(0., margin - eucd2)
	# follow the paper, the function is not symmetrical
	#pos = tf.mul(y_f, eucd2, name = "Nyi_x_eucd2") # the first half of the loss
	#neg = tf.mul(y_t, tf.pow(tf.maximum(tf.sub(C, eucd),0),2), name = "yi_x_C-eucd_xx_2") # the second half of the loss
	#losses = tf.add(pos, neg, name= "losses")
	#losses = (1-y) * eucd2 + y * tf.square(tf.maximum(0., margin - eucd))
	loss = tf.reduce_mean(losses, name = "loss")
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)


#init session
sess = tf.Session()
if int((tf.__version__).split('.')[1]) <12 and int((tf.__version__).split('.')[0]) <1: #tensorflow version <0.12
	writer=tf.train.SummaryWriter('logs/', sess.graph)
else: # tensorflow version >=0.12
	writer=tf.summary.FileWriter('logs/', sess.graph)

init = tf.global_variables_initializer()
sess.run(init)


def print_usage():
	print "%s train"%sys.argv[0]
	print "%s predict [modelname]"%sys.argv[0]
	sys.exit(42)


def train():
	#loader = dataloader.DataLoader("dataset_conf_path.txt")
	#loader = dataloader.DataLoader("face_dataset_conf_path.txt")
	img_path= "./data/image_face_v0/images_face/"
	loader = dataloader.DataLoader("image_face_v0_list.txt",img_path)
	loader.load_list()

	epoch_num = 30000
	#epoch_num = 1
	iter_per_epoch = 1
	# get an untouched  data test for final test
	# load from list and remove them 
	valid_list = loader.get_test_set(1000,1000)
	with open("untouched_test_list.txt", "w") as of:
		for line in valid_list:
			of.writelines(line+"\n")

	#get the validation dataset
	# load from list and remove them
	test_list = loader.get_test_set(300,300)
	#t_x1, t_x2, t_y = list_to_data(test_list, label_type = "onehot")
	t_x1, t_x2, t_y = list_to_data(test_list, label_type = "scalar_revert")
	# do training.
	# without using test data loaded before.
	for i in range(epoch_num):
		print "epoch %d"%i
		train_list = loader.next_epoch_list(100,100)
		print loader.pos_idx, loader.neg_idx
		#in_x1, in_x2, in_y = list_to_data(train_list, label_type = "onehot")
		in_x1, in_x2, in_y = list_to_data(train_list, label_type = "scalar_revert")

			
		loss_before =-1.0
		loss_after = -1.0
		loss_test = -1.0
		starttime = None
		endtime = None
		if i % 50  == 0:
			#print"on trained before:", (get_accuracy(in_x1, in_x2, in_y))
			starttime = time.time()
			loss_before = sess.run(loss, feed_dict={x1: in_x1, x2:in_x2, y:in_y, keep_prob:1})

		for j in range(iter_per_epoch):
			sess.run(train_step,  feed_dict = {x1:in_x1, x2:in_x2, y:in_y, keep_prob:0.5})
		if i % 50 == 0:
			endtimet= time.time()
			#print"on test:", (get_accuracy(t_x1, t_x2, t_y))
			loss_test, dist, test_y = sess.run([loss,eucd,y], feed_dict={x1: t_x1, x2:t_x2, y:t_y, keep_prob:1})
			accu_test = get_siamese_accuracy(dist, t_y)			
			#print dist
			#print"on trained after:", (get_accuracy(in_x1, in_x2, in_y))
			loss_after = sess.run(loss, feed_dict={x1: in_x1, x2:in_x2, y:in_y, keep_prob:1})
			print "report on %d th turn, trained %d iterations, on train: %f -> %f; on test: loss= %f; accu = %f"%(i, iter_per_epoch, loss_before, loss_after, loss_test, accu_test)

			print "============ start debug session =============================="
			#print "tensor shape h_pool2_flat", h_pool2_flat.get_shape().as_list()
			print "tensor shape h_pool2_flat", sess.run(h_pool2_flat, feed_dict={x1: t_x1, x2:t_x2, y:t_y, keep_prob:1}).shape
			print "tensor shape h_fc1", sess.run(h_fc1, feed_dict={x1: t_x1, x2:t_x2, y:t_y, keep_prob:1}).shape
			#print "tensor shape h2_pool2_flat", h2_pool2_flat.get_shape().as_list()
			print "tensor shape h2_pool2_flat", sess.run(h2_pool2_flat, feed_dict={x1: t_x1, x2:t_x2, y:t_y, keep_prob:1}).shape
			print "tensor shape h2_fc1", sess.run(h2_fc1, feed_dict={x1: t_x1, x2:t_x2, y:t_y, keep_prob:1}).shape
			#print "tensor shape eucd2:", eucd2.get_shape().as_list()
			#print "tensor shape eucd:",  eucd.get_shape().as_list()
			print " eucd dist:", np.shape(dist)
			print dist
			print "test_y:", np.shape(test_y)
			print test_y
			print "============ end debug session =============================="


		# save tmp model
		if (i% 5000 == 0) and (i > 1):
			save_model("tmp.ckpt")
			print "trained model on %d th minibatch saved to %s:"%(i,"tmp.ckpt")
			
	# test do prediction here
	#predict(valid_list)
	#predict_siamese_sim(test_list)

def save_model(model_file_str=None):
	saver = tf.train.Saver()
	tt = strftime("%Y-%m-%d_%H_%M_%S", gmtime())
	filename = "nets/save_net_"+ tt +".ckpt"
	if model_file_str !=  None:
		filename = model_file_str
	save_path = saver.save(sess,filename) 
	print("trained model saved to:", save_path)

def load_model(netfilestr):
	print "load model %s"%netfilestr
	saver = tf.train.Saver()
	saver.restore(sess, netfilestr)
	#print("weights:", sess.run(W_conv1))
	#print("biases:", sess.run(b_conv1))

def predict(pairlist):	
	for pair in pairlist:
		pre_x1, pre_x2 = list_to_predict_data([pair])
		#print ("predict results:",  sess.run(logits, feed_dict = {x1:pre_x1, x2:pre_x2, keep_prob:1}))
		y_pre = sess.run(logits, feed_dict = {x1:pre_x1, x2:pre_x2, keep_prob:1})
		label = np.argmax(y_pre) 
		print "predict: " + pair + "\t"+ str(label)
		print "details:", y_pre
	
def predict_siamese_sim(pairlist):
	for pair in pairlist:
		pre_x1, pre_x2 = list_to_predict_data([pair])
		#print ("predict results:",  sess.run(logits, feed_dict = {x1:pre_x1, x2:pre_x2, keep_prob:1}))
		dist = sess.run(eucd, feed_dict = {x1: pre_x1, x2: pre_x2, keep_prob:1})
		print "predict: " + pair + "\t"+ str(dist)



def export_model_for_serving():
	# load an existing model	
	#modelname = "nets/save_net_2017-07-22_06_23_56.ckpt"
	modelname= "nets/save_net_2017-07-22_00_05_22.ckpt"
	load_model(modelname)

	#export to serving format	
	saved_model_dir = "./exported_model/1/"
	#builder = saved_model_builder.SavedModelBuilder(saved_model_dir)
	builder = tf.saved_model.builder.SavedModelBuilder(saved_model_dir)
	inputs = {"input_x1": tf.saved_model.utils.build_tensor_info(x1),
		"input_x2": tf.saved_model.utils.build_tensor_info(x2),
		"keep_prob": tf.saved_model.utils.build_tensor_info(keep_prob)}
	outputs = {"output": tf.saved_model.utils.build_tensor_info(y)}
	signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs, "test_sig_name")	

	builder.add_meta_graph_and_variables(
		sess, 
		#['imgsim_test_export_1'], 
		[tf.saved_model.tag_constants.SERVING], #it seems you have to use this tag if you want to serve
		{"test_signature":signature}
	)
	builder.save()
	print "Done Exporting!"

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print_usage()
	mode = sys.argv[1]
	if mode == "train":
		train()
		save_model()
	elif mode == "predict":
		#load_model("nets/save_net_2017-06-24_19_30_32.ckpt")	
		#load_model("nets/save_net_2017-06-29_12_45_02.ckpt")
		#pred_filestr = "predict_list.txt"	
		modelname = "nets/save_net_2017-07-12_11_21_10.ckpt"
		pred_filestr = "untouched_test_list.txt"	
                if len(sys.argv) >= 3:
                        modelname = sys.argv[2]
                        print modelname

		load_model(modelname)
		pred_list = []
		with open(pred_filestr, "r") as f:
			for line in f:
				line = line.strip()
				pred_list.append(line)
		
		print "do prediction ..."
		#predict(pred_list)	
		predict_siamese_sim(pred_list)
	elif mode == "export": #export the model for serving
		export_model_for_serving()	
	else:
		print_usage()
	sys.exit()
	
