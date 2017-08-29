import os
import sys
import base64


from gevent import monkey
monkey.patch_all()
from flask import Flask, request, Response, jsonify, send_file, render_template
from gevent import wsgi
import tensorflow as tf

sys.path.append("../")
#self defined modules
import img_sim3

sys.path.append("../face_detector/")
import face_detector


'''
a = tf.placeholder(tf.int32, shape=(), name = "input")
asquare = tf.multiply(a, a, name = "output")
sess = tf.Session()
'''

app = Flask(__name__, static_url_path='')


@app.route('/')
def index():
	# some how this folder has to be under /static folder
    	return app.send_static_file('index.html')
	#return render_template("render_index.html")

@app.route('/hello')
#@app.route('/hello', methods = ['POST'])
def response_request():
	pred_filestr = "predict_list.txt"
	pred_list = []
	with open(pred_filestr, "r") as f:
		for line in f:
			line = line.strip()
			pred_list.append(line)
	

	results = img_sim3.predict_siamese_sim(pred_list) #only x1 and x2
	return jsonify({"prediction": results})
	#return "<br>".join(results)

@app.route('/sim', methods = ["POST"])
def calculate_sim():
	# get images and save
	#if 'image1' not in request.files:
	
	content = request.get_json(silent=True)
	if content is None:
		return Response('no data is submitted', 400)
	if "image1" not in content:
		return Response('Missing image1 paramter', 400)
	if "image2" not in content:
		return Response('Missing image2 paramter', 400)
	if "type" not in content:
		return Response('Missing type paramter', 400)

	
	
	option = content['type']

	#write image1 to disk
	img1_data = content['image1']
	idx = img1_data.find(',')
	img1_data = img1_data[idx+1:] 
	img1_name = "./request_images/test1.jpg"	
	with open(img1_name, 'wb') as f:
		f.write(base64.decodestring(img1_data))
	#cor face 1	
	img1face1_name = face_detector.do_corp(img1_name)
	if img1face1_name is None:
		print("no face detected in %s"% img1_name)
		img1face1_name = img1_name
	
	
	#write image2 to disk
	img2_data = content['image2']
	idx = img2_data.find(',')
	img2_data = img2_data[idx+1:] 
	img2_name = "./request_images/test2.jpg"	
	with open(img2_name, 'wb') as f:
		f.write(base64.decodestring(img2_data))
	#corp face2	
	img2face1_name = face_detector.do_corp(img2_name)
	if img2face1_name is None:
		print("no face detected in %s"% img2_name)
		img2face1_name = img2_name

	
	# calculate the similarities	
	line = img1_name + "\t" + img2_name
	if option == "face":
		line = img1face1_name + "\t" + img2face1_name
	results = img_sim3.predict_siamese_sim([line]) #only x1 and x2
	#return jsonify({"prediction": results})
	#return render_template("render_index.html", result = results)
	results = [1-float(x) for x in results]
	return render_template("show_results_ajax.html", result = results)


# this function works with index_v2.html
# tobe deprecated
@app.route('/sim2', methods = ["POST"])
def calculate_sim2():
	# get images and save
	#if 'image1' not in request.files:
	if 'image1' not in request.files:
		return Response('Missing image1 paramter', 400)
	if 'image2' not in request.files:
		return Response('Missing image2 paramter', 400)

	option = request.form['type']
	#write image1 to disk
	img1_name = './request_images/'+request.files['image1'].filename
    	#with open(img1_tmp_name, 'wb') as f:
        #	f.write(request.files['image1'].read())	
	with open(img1_name, 'wb') as f:
        	f.write(request.files['image1'].read())	
	#img1face1_name = "./request_images/img1_face1.jgp"
	img1face1_name = face_detector.do_corp(img1_name)
	if img1face1_name is None:
		print("no face detected in %s"% img1_name)
		img1face1_name = img1_name
		
	
	
	#write image2 to disk	
	img2_name = './request_images/'+request.files['image2'].filename
    	#with open(img2_tmp_name, 'wb') as f:
        #	f.write(request.files['image2'].read())	
	with open(img2_name, 'wb') as f:
        	f.write(request.files['image2'].read())	
	
	img2face1_name = face_detector.do_corp(img2_name)
	if img2face1_name is None:
		print("no face detected in %s"% img2_name)
		img2face1_name = img2_name
	
	
	# calculate the similarities	
	line = img1_name + "\t" + img2_name
	if option == "face":
		line = img1face1_name + "\t" + img2face1_name
	results = img_sim3.predict_siamese_sim([line]) #only x1 and x2
	#return jsonify({"prediction": results})
	#return render_template("render_index.html", result = results)
	results = [1-float(x) for x in results]
	return render_template("show_results2.html", result = results)

'''
def response_request():
	num = request.args.get('num')
	for i in range (100):
		ret = sess.run([asquare], feed_dict={a:num})
	return str(ret)
'''

if __name__ == "__main__":

	img_sim3.load_model_for_flask_serving("../nets/save_net_2017-08-10_07_55_18.ckpt")
	#server = wsgi.WSGIServer(('127.0.0.1', 19877), app)
	server = wsgi.WSGIServer(('0.0.0.0', 19877), app)

	#server = wsgi.WSGIServer(('0.0.0.0', 8080), app)
	#server = wsgi.WSGIServer(('127.0.0.1', 19877), app)
	server.serve_forever()
