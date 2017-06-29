import sys
import random
import os

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

		return merged_list	
	

	# get a mini batch from the loaded list
	def next_epoch_list(self, posnum, negnum):
		print "load next epoch from: ", len(self.pos_data), len(self.neg_data)
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



