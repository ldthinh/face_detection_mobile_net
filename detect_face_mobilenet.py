import tensorflow as tf
import cv2
import time
import numpy as np
import os

detection_graph = tf.Graph()
model_path = os.getcwd() + "/model/frozen_inference_graph_face.pb"

with detection_graph.as_default():
	od_graph_def = tf.compat.v1.GraphDef()
	with tf.io.gfile.GFile(model_path, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')
		config = tf.compat.v1.ConfigProto()
	config.gpu_options.allow_growth = True
	sess=tf.compat.v1.Session(graph=detection_graph, config=config)
	image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
	boxes_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')    
	scores_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
	num_detections = detection_graph.get_tensor_by_name('num_detections:0')

def get_mobilenet_face(image):
	start = time.time()
	global boxes,scores,num_detections
	(im_height,im_width) = image.shape[:-1]
	imgs = np.array([image])
	(boxes, scores) = sess.run(
		[boxes_tensor, scores_tensor],
		feed_dict={image_tensor: imgs})
	# change to detect multi-face in picture, because it chooses max scorces of face , so just choose the only one
	# print("scores", scores)
	max_= np.where(scores == scores.max())[0][0]
	box = boxes[0][max_]
	ymin, xmin, ymax, xmax = box
	(left, right, top, bottom) = (xmin * im_width, xmax * im_width,
								ymin * im_height, ymax * im_height)
	left, right, top, bottom = int(left), int(right), int(top), int(bottom)
	return time.time()-start,(left, right, top, bottom)

def histogram_equalization(img_in):
# segregate color streams
	b,g,r = cv2.split(img_in)
	h_b, bin_b = np.histogram(b.flatten(), 256, [0, 256])
	h_g, bin_g = np.histogram(g.flatten(), 256, [0, 256])
	h_r, bin_r = np.histogram(r.flatten(), 256, [0, 256])
# calculate cdf    
	cdf_b = np.cumsum(h_b)  
	cdf_g = np.cumsum(h_g)
	cdf_r = np.cumsum(h_r)
	
# mask all pixels with value=0 and replace it with mean of the pixel values 
	cdf_m_b = np.ma.masked_equal(cdf_b,0)
	cdf_m_b = (cdf_m_b - cdf_m_b.min())*255/(cdf_m_b.max()-cdf_m_b.min())
	cdf_final_b = np.ma.filled(cdf_m_b,0).astype('uint8')
  
	cdf_m_g = np.ma.masked_equal(cdf_g,0)
	cdf_m_g = (cdf_m_g - cdf_m_g.min())*255/(cdf_m_g.max()-cdf_m_g.min())
	cdf_final_g = np.ma.filled(cdf_m_g,0).astype('uint8')
	cdf_m_r = np.ma.masked_equal(cdf_r,0)
	cdf_m_r = (cdf_m_r - cdf_m_r.min())*255/(cdf_m_r.max()-cdf_m_r.min())
	cdf_final_r = np.ma.filled(cdf_m_r,0).astype('uint8')
# merge the images in the three channels
	img_b = cdf_final_b[b]
	img_g = cdf_final_g[g]
	img_r = cdf_final_r[r]
  
	img_out = cv2.merge((img_b, img_g, img_r))

	return img_out
	
def annotate_image(frame,bbox,color):
	if bbox==[]:
		return frame
	frame=frame.copy()
	return cv2.rectangle(frame,(bbox[0],bbox[2]),(bbox[1],bbox[3]),color,2)

while True:

	img = cv2.imread("preview_5.jpg")
	# img = histogram_equalization(img)
	mobilenet_time, mobilenet_bboxes = get_mobilenet_face(img)
	print("Mobilenet Detection Time:"+str(mobilenet_time))
	frame = annotate_image(img, mobilenet_bboxes, (255,0,0))
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	cv2.imshow("Frame",frame)
