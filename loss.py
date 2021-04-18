import dataset as data
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

epsilon = 1e-7

def bbox_iou(boxes1, boxes2):
	boxes1_area = boxes1[..., 2] * boxes1[..., 3]
	boxes2_area = boxes2[..., 2] * boxes2[..., 3]

	boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
						boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
	boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
						boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

	left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
	right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

	inter_section = np.maximum(right_down - left_up, 0.0)
	inter_area = inter_section[..., 0] * inter_section[..., 1]
	union_area = boxes1_area + boxes2_area - inter_area
	iou = 1.0 * inter_area / union_area
	return iou

def bbox_ciou(boxes1, boxes2):
	boxes1_x0y0x1y1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
								 boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
	boxes2_x0y0x1y1 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
								 boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

	boxes1_x0y0x1y1 = np.concatenate([np.minimum(boxes1_x0y0x1y1[..., :2], boxes1_x0y0x1y1[..., 2:]),
								 np.maximum(boxes1_x0y0x1y1[..., :2], boxes1_x0y0x1y1[..., 2:])], axis=-1)
	boxes2_x0y0x1y1 = np.concatenate([np.minimum(boxes2_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., 2:]),
								 np.maximum(boxes2_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., 2:])], axis=-1)

	boxes1_area = (boxes1_x0y0x1y1[..., 2] - boxes1_x0y0x1y1[..., 0]) * (
				boxes1_x0y0x1y1[..., 3] - boxes1_x0y0x1y1[..., 1])
	boxes2_area = (boxes2_x0y0x1y1[..., 2] - boxes2_x0y0x1y1[..., 0]) * (
				boxes2_x0y0x1y1[..., 3] - boxes2_x0y0x1y1[..., 1])

	left_up = np.maximum(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
	right_down = np.minimum(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])

	inter_section = np.maximum(right_down - left_up, 0.0)
	inter_area = inter_section[..., 0] * inter_section[..., 1]
	union_area = boxes1_area + boxes2_area - inter_area
	iou = inter_area / (union_area + epsilon)

	enclose_left_up = np.minimum(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
	enclose_right_down = np.maximum(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])

	enclose_wh = enclose_right_down - enclose_left_up
	enclose_c2 = np.power(enclose_wh[..., 0], 2) + np.power(enclose_wh[..., 1], 2) + epsilon

	p2 = np.power(boxes1[..., 0] - boxes2[..., 0], 2) + np.power(boxes1[..., 1] - boxes2[..., 1], 2)

	atan1 = np.arctan(boxes1[..., 2] / (boxes1[..., 3] + epsilon))
	atan2 = np.arctan(boxes2[..., 2] / (boxes2[..., 3] + epsilon))
	v = 4.0 * np.power(atan1 - atan2, 2) / (np.pi ** 2)
	a = v / (1 - iou + v)

	ciou = iou - 1.0 * p2 / enclose_c2 - 1.0 * a * v
	return ciou


def YOLO_loss(dataset, output_1, output_2, confidence_tresh=0.4, iou_treshold=0.6):
	if not isinstance(dataset, data.Dataset):
		raise Exception("please input a valid dataset")
	def loss(y_true, y_pred):
		print("output_1 shape :", output_1.shape)
		print("output_2 shape :", output_2.shape)
		print("pred (AKA output_3) shape :", y_pred.shape)
		y_pred = np.array([output_1, output_2, y_pred])
		pred_bbox = dataset.yolo_output_to_bbox(y_pred)
		print(pred_bbox.shape)
		label_bbox = np.array([dataset.loss_preprocess_y(y) for y in y_true])
		print(label_bbox.shape)
		exit(1)
		# cleaning confidense treshold
		conf_mask = np.expand_dims(pred_bbox[:, :, 0] > confidence_tresh, -1)
		pred_bbox = pred_bbox * conf_mask
		
		#keep only max class for nms
		tmp = pred_bbox[:, :, 5:-1]
		abs_classes = np.argmax(tmp, axis=-1)

		# NMS (removing boxes that are overlaping to much with each others)
		for ind in range(dataset.batch_size):
			image_pred = np.array(pred_bbox[ind])
			image_true = label_bbox[ind]
			image_pred_classes = abs_classes[ind]
			unique_class = np.unique(image_pred_classes)
			for c in unique_class:
				class_mask = (image_pred_classes == c)
				class_bboxes = image_pred[class_mask]
				for i, box in enumerate(class_bboxes[:-1]):
					iou = bbox_iou(box[1:5], class_bboxes[i+1:,1:5])
					if max(iou) > iou_treshold:
						box_index = int(box[-1])
						pred_bbox[ind][box_index] = np.zeros((dataset.nclass + 5)).tolist() + [box_index]
		respond_bbox = label_bbox[..., 0]

		# ciou
		ciou = bbox_ciou(pred_bbox[..., 1:5], label_bbox[..., 1:5])
		ciou_loss_scale = (2.0 - 1.0 * label_bbox[..., 3:4] * label_bbox[..., 4:5] / (dataset.shape[0] ** 2))[..., 0]
		ciou_loss = respond_bbox * ciou_loss_scale * (1 - ciou)
		# print('ciou loss :', tf.reduce_mean(tf.reduce_sum(ciou_loss, axis=1)))

		# probabilistic loss
		prob_loss = respond_bbox * tf.losses.sparse_categorical_crossentropy(label_bbox[..., 5], tmp)
		# print('prob loss :', tf.reduce_mean(tf.reduce_sum(prob_loss, axis=1)))

		# confidence loss
		conf_loss = tf.losses.binary_crossentropy(label_bbox[..., 0:1], pred_bbox[..., 0:1])
		# print('conf loss :', tf.reduce_mean(tf.reduce_sum(conf_loss, axis=1)))

		total_loss = tf.reduce_mean(tf.reduce_sum(ciou_loss, axis=1)) + tf.reduce_mean(tf.reduce_sum(prob_loss, axis=1)) + tf.reduce_mean(tf.reduce_sum(conf_loss, axis=1))
		return total_loss
	return loss

''' old numpy loss 
def YOLO_loss(dataset, confidence_tresh=0.4, iou_treshold=0.6):
	if not isinstance(dataset, data.Dataset):
		raise Exception("please input a valid dataset")
	def loss(y_true, y_pred):
		pred_bbox = dataset.yolo_output_to_bbox(y_pred)
		label_bbox = np.array([dataset.loss_preprocess_y(y) for y in y_true])

		# cleaning confidense treshold
		conf_mask = np.expand_dims(pred_bbox[:, :, 0] > confidence_tresh, -1)
		pred_bbox = pred_bbox * conf_mask
		
		#keep only max class for nms
		tmp = pred_bbox[:, :, 5:-1]
		abs_classes = np.argmax(tmp, axis=-1)

		# NMS (removing boxes that are overlaping to much with each others)
		for ind in range(dataset.batch_size):
			image_pred = np.array(pred_bbox[ind])
			image_true = label_bbox[ind]
			image_pred_classes = abs_classes[ind]
			unique_class = np.unique(image_pred_classes)
			for c in unique_class:
				class_mask = (image_pred_classes == c)
				class_bboxes = image_pred[class_mask]
				for i, box in enumerate(class_bboxes[:-1]):
					iou = bbox_iou(box[1:5], class_bboxes[i+1:,1:5])
					if max(iou) > iou_treshold:
						box_index = int(box[-1])
						pred_bbox[ind][box_index] = np.zeros((dataset.nclass + 5)).tolist() + [box_index]
		respond_bbox = label_bbox[..., 0]

		# ciou
		ciou = bbox_ciou(pred_bbox[..., 1:5], label_bbox[..., 1:5])
		ciou_loss_scale = (2.0 - 1.0 * label_bbox[..., 3:4] * label_bbox[..., 4:5] / (dataset.shape[0] ** 2))[..., 0]
		ciou_loss = respond_bbox * ciou_loss_scale * (1 - ciou)
		print('ciou loss :', tf.reduce_mean(tf.reduce_sum(ciou_loss, axis=1)))

		# probabilistic loss
		prob_loss = respond_bbox * tf.losses.sparse_categorical_crossentropy(label_bbox[..., 5], tmp)
		print('prob loss :', tf.reduce_mean(tf.reduce_sum(prob_loss, axis=1)))

		# confidence loss
		conf_loss = tf.losses.binary_crossentropy(label_bbox[..., 0:1], pred_bbox[..., 0:1])
		print('conf loss :', tf.reduce_mean(tf.reduce_sum(conf_loss, axis=1)))

		total_loss = tf.reduce_mean(tf.reduce_sum(ciou_loss, axis=1)) + tf.reduce_mean(tf.reduce_sum(prob_loss, axis=1)) + tf.reduce_mean(tf.reduce_sum(conf_loss, axis=1))
		return total_loss
	return loss 
'''