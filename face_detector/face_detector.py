import cv2
import sys

cascPath = "haarcascade_frontalface_default.xml"

	
def do_corp(infilestr):
	ofilestr = None
	# Creatge the haar cascade
	faceCascade= cv2.CascadeClassifier(cascPath)
	# read teh image
	image = cv2.imread(infilestr)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	# Detect faces in the images
	faces = faceCascade.detectMultiScale(
		gray, 
		scaleFactor =1.1,
		minNeighbors = 5,
		minSize = (30,30),
		flags = cv2.cv.CV_HAAR_SCALE_IMAGE
	)
		
	for (x, y, w, h) in faces:
		pad = 20
			
		img = image[x:w, y:h]
		img = image[y:y+h, x:x+w]
		#cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0),2)

		idx = infilestr.rfind('.')
		if idx == -1: break
		name = infilestr[:idx]
		suffix = infilestr[idx+1:]	
		ofilestr = name+"_face." + suffix
		cv2.imwrite(ofilestr, img)
		break	

	return ofilestr
	

def gen_corpedface(dataconf,newdataconf):
	ofile = open(newdataconf, "w")
	with open(dataconf, "r") as indata:
		for line in indata:
			line =line.strip()
			cols = line.split("\t")
			if (len(cols)!=3):
				continue
			f1str = cols[0]
			f2str = cols[1]
			label = cols[2]		

			ofstr1 = do_corp(f1str)
			print "corped %s!"%ofstr1
			if ofstr1 is None:
				continue
	
			ofstr2 = do_corp(f2str)
			print "corped %s!"%ofstr2
			if ofstr2 is None:
				continue
			
			ofile.write("\t".join([ofstr1, ofstr2, label])+"\n")				
	ofile.close()
			
if __name__ == "__main__":
	# A0003382e9eb11ce83484a46e1c412837.jpg
	gen_corpedface("dataset_conf_path.txt", "face_dataset_conf_path.txt")
