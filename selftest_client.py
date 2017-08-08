from grpc.beta import implementations
import numpy as np
import tensorflow as tf
import dataloader


from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
tf.app.flags.DEFINE_float("request_timeout", 10.0, "Timeout of gRPC request")
FLAGS = tf.app.flags.FLAGS

def main(_):
	host, port = FLAGS.server.split(':')
	channel = implementations.insecure_channel(host, int(port))
	stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
	
	request = predict_pb2.PredictRequest()
	request.model_spec.name = "exported"
	request.model_spec.signature_name = "serving"
	# open file 1
	# open file 2
			
	#generate test data
	image_loader = dataloader.ImageLoader()	
	pred_filestr = "/home/midi/Data/predict_list_short.txt"
	pred_list = []
	with open(pred_filestr, "r") as f:
		for line in f:
			line = line.strip()
			print line
			pred_list.append(line)

        image_loader = dataloader.ImageLoader()
    	pre_x1, pre_x2 = image_loader.list_to_predict_data(pred_list)
	keep_prob = float(1.0)

	request = predict_pb2.PredictRequest()
	request.model_spec.name = 'exported'	 
	request.model_spec.signature_name = 'test_signature'	 
	request.inputs['keep_prob'].CopyFrom(
	        tf.contrib.util.make_tensor_proto(keep_prob, dtype = tf.float32, shape=[1]))	


	for x1, x2 in zip(pre_x1, pre_x2):
		request.inputs['input_x1'].CopyFrom(
	        	tf.contrib.util.make_tensor_proto(x1, dtype = tf.float32, shape=[1, x1.size]))	
		request.inputs['input_x2'].CopyFrom(
	       		tf.contrib.util.make_tensor_proto(x2, dtype = tf.float32, shape=[1, x2.size]))

		result = stub.Predict(request, FLAGS.request_timeout)	

		print result
	#result = stub.Predict(request, 10.0) # 10 secs timeout

if __name__ ==  '__main__':
	tf.app.run()
