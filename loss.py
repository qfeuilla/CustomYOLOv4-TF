import dataset as data
import tensorflow as tf

def YOLO_loss(dataset):
	if not isinstance(dataset, data.Dataset):
		raise Exception("please input a valid dataset")
	def loss(y_true, y_pred):
		bbox = dataset.yolo_output_to_bbox(y_pred)
		
		return tf.reduce_mean(y_true)
	return loss 