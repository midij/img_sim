import sys
import random
import os
import cv2
import numpy as np
import logging

class DataLoader(object):
	infile  = None
	pos_data = []
	neg_data = []
	pos_idx = 0
	neg_idx = 0
	path = None  #the path for images
	def __init__(self, datalistfile, path=None):
		self.path = path
		self.infile = datalistfile
	
	def do_print_shortlist(self, alist):
		for i in range(min(10, len(alist))):
			print alist[i]

	def do_print(self):
		print "pos:"
		for i in range(min(10, len(self.pos_data))):
			print self.pos_data[i]

		print "neg:"
		for i in range(min(10, len(self.neg_data))):
			print self.neg_data[i]

	#load the full list information from a file
	def load_list(self):
		if self.infile == None:
			return 
		cnt_nofile = 0
		cnt_line = 0
		with open(self.infile, "r") as f:
			for line in f:
				cnt_line +=1
				line = line.strip()
				cols = line.split("\t")	
				if len(cols) <3:
					continue
				else:
					img1path = cols[0]
					img2path = cols[1]
					if not self.path is None:
						img1path = self.path + cols[0]
						img2path = self.path + cols[1]
					newline = "\t".join([img1path, img2path, cols[2]])
					if not os.path.exists(img1path):
						cnt_nofile += 1
						continue
					if not os.path.exists(img2path):
						cnt_nofile+=1
						continue
					
					if cols[2] == "1":
						self.pos_data.append(newline)
					else:
						self.neg_data.append(newline)
					
				if cnt_line % 500 == 0:
					print "cnt_line %d"%cnt_line
					print "cnt_nofile %d"%cnt_nofile
					print "cnt_pos %d"%len(self.pos_data)
					print "cnt_neg %d"%len(self.neg_data)
		random.shuffle(self.pos_data)	
		random.shuffle(self.neg_data)

	# this function should always be called before the nex_epoch_lis function is called
	# get test dat from list
	# then remove those test data from the list
	def get_test_set(self, posnum, negnum):
		print "load test epoch from: ", len(self.pos_data), len(self.neg_data)
		pos_list = []
		neg_list = []
		merged_list = []
		if (posnum > len(self.pos_data)):
			print "test set pos num > pos num"
			posnum = len(self.pos_data)
		if (negnum > len(self.neg_data)):
			print "test set neg num > neg num"
			negnum = len(self.neg_data)
		pos_list = self.pos_data[:posnum]
		neg_list = self.neg_data[:negnum] 
		merged_list.extend(pos_list)
		merged_list.extend(neg_list)
		random.shuffle(merged_list)
		
		self.pos_data = self.pos_data[posnum:]
		self.neg_data = self.neg_data[negnum:]
		
		self.remove_test_points_from_list(merged_list)

		return merged_list	
	

	# test list: a line inthe form of: point1 \t  point2 \t label
	# remove all the lines in the pos/neg list that contains point1 or point2 
	def remove_test_points_from_list(self, test_list):
		print "start cleaning list" 
		test_point_dic = {}
		for line in test_list:
			line = line.strip()
			cols = line.split('\t')
			try:
				p1 = cols[0]
				p2 = cols[1]
				test_point_dic[p1] = 1
				test_point_dic[p2] = 1	
			except:
				print "skip test pair: %s"%line	
			
		pos_list = []
		neg_list = []
		#deal with pos data
		for line in self.pos_data:
			line = line.strip()
			cols = line.split('\t')
			p1 = cols[0]
			p2 = cols[0]
			if p1 in test_point_dic:
				continue
			if p2 in test_point_dic: 
				continue
			pos_list.append(line)

		#del with neg data
		for line in self.neg_data:	
			line = line.strip()
			cols = line.split('\t')
			p1 = cols[0]
			p2 = cols[0]
			if p1 in test_point_dic:
				continue
			if p2 in test_point_dic: 
				continue
			neg_list.append(line)

		self.pos_data = pos_list
		self.neg_data = neg_list
		print "after test data filter, [pos_data_len, neg_data_len] = [%d,%d]"%(len(self.pos_data), len(self.neg_data))

	# get a mini batch from the loaded list
	def next_epoch_list(self, posnum, negnum):
		print "load next epoch, [pos_data_len, neg_data_len]= [%d, %d]"%(len(self.pos_data), len(self.neg_data))
		pos_list = []			
		neg_list = []
		merged_list = []

		while ( posnum > 0):
			#print "pos:", len(self.pos_data)
			end = min((self.pos_idx + posnum), len(self.pos_data))
			#print end
			pos_list.extend(self.pos_data[self.pos_idx: end])					
			posnum = posnum - (end - self.pos_idx) 
			self.pos_idx = end % (len(self.pos_data))
			#print posnum

		while ( negnum > 0):
			#print "neg:", len(self.neg_data)
			end = min((self.neg_idx + negnum), len(self.neg_data))
			neg_list.extend(self.neg_data[self.neg_idx: end])					
			negnum = negnum - (end - self.neg_idx) 
			self.neg_idx = end % (len(self.neg_data))
			#print negnum
		
		#print len(pos_list)
		merged_list.extend(pos_list)
		merged_list.extend(neg_list)
		random.shuffle(merged_list)
			
		return merged_list


class ImageLoader(object):
	
	IMG_SIZE = 128
	def denseToOneHot(self, labels_dense, num_classes):
   		#Convert class labels from scalars to one-hot vectors.
		num_labels = labels_dense.shape[0]
		index_offset = np.arange(num_labels) * num_classes
		labels_one_hot = np.zeros((num_labels, num_classes))
		labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
		return labels_one_hot


	

	def loadFeatures(self, files):
		data = np.ndarray((len(files), self.IMG_SIZE * self.IMG_SIZE * 3))
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
			img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
			data[n] = img.ravel()
			# cv2.imshow("res", img)
			# cv2.waitKey(0)
		return data

	#input: labbel_type = onehot or scalar or scalar_revert
	# it will return you : [0,1] or 1, value for a label
	def list_to_data(self, inlist, label_type = "onehot"):
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
		x1 = self.loadFeatures(x1_list)
		x2 = self.loadFeatures(x2_list)
		if label_type == "onehot":
			label_list = self.denseToOneHot(np.array(label_list),2)
		elif label_type == "scalar":
			label_list = np.reshape(label_list,(-1,1))
		elif label_type == "scalar_revert":
			#label_list[:] = [1-x for x in label_list]
			#label_list = np.reshape(label_list,(-1,1))
			pass
		return x1, x2, label_list

	def list_to_predict_data(self, inlist):
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

		x1 = self.loadFeatures(x1_list)
		x2 = self.loadFeatures(x2_list)
		return x1, x2	
		


