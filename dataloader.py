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

	def next_epoch_list(self, posnum, negnum):
		print "load next epoch"
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


#oncf format:
# image1 \t image2 \ label =(0/1)
def load_conf(conf_str):
	train_pos_num = 20
	train_neg_num = 180
	
	train_dic_pos={}
	train_dic_neg = {}

	test_pos_num = 150
	test_neg_num = 150
	test_dic_pos = {}
	test_dic_neg = {}
	
	
	with open(conf_str,"r") as f:
		for line in f:
			line = line.strip()
			cols = line.split('\t')
			if cols[2] == "1":
				print line
				print len(train_dic_pos.keys())
				print len(test_dic_pos.keys())
				if len(train_dic_pos.keys()) < train_pos_num:
					train_dic_pos[line] = 1
				elif len(test_dic_pos.keys()) < test_pos_num:
					test_dic_pos[line] =1
			else:
				if len(train_dic_neg.keys()) < train_neg_num:
					train_dic_neg[line] = 1
				elif len(test_dic_neg.keys()) < test_neg_num:
					test_dic_neg[line] =1
		#output:
		train_conf = "train_conf.txt"
		test_conf = "test_conf.txt"
		
		train_list = train_dic_pos.keys()	
		train_list.extend(train_dic_neg.keys())
		random.shuffle(train_list)
		with open(train_conf,"w") as f:
			for key in train_list:
				f.write(key+'\n')
		
		test_list = test_dic_pos.keys()	
		test_list.extend(test_dic_neg.keys())
		random.shuffle(test_list)
	
		with open(test_conf, "w") as f:
			for key in test_list:
				f.write(key+'\n')

'''

if __name__ == "__main__":
	#load_conf("dataset_conf_path.txt")
	loader = DataLoader("dataset_conf.txt")
	loader.load_list()
	loader.do_print()
	test_set = loader.next_epoch_list(150,150)
	print loader.pos_idx
	print loader.neg_idx
	loader.do_print_shortlist(test_set)
	train_epoch = loader.next_epoch_list(100,100)
	loader.do_print_shortlist(train_epoch)
	print loader.pos_idx
	print loader.neg_idx

	

'''
