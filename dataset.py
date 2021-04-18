import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import cv2
from tqdm import tqdm
import tensorflow as tf
import glob
import os
from tqdm import tqdm

def euclidian_dist(a, b):
	return tf.norm(a - b)

class Dataset():
	def __init__(self, batch_size, head_shape, augmention_strength="None", image_directory="data/", annotation_directory="data_getter_and_preprocess/", anchor_bbox_file="anchor_bbox.npy", annotation_pattern="cleaned_{}_bbox.csv", shape=(608, 608), classes=['Human face', 'Man', 'Woman', 'Clothing', 'Car', 'Wheel', 'Food', 'Vehicle', 'Boy', 'Girl'], bbox_per_cell=2):
		"""
		Parameters
		batch_size : int 
		augmention_strength : str
			either None, soft or strong
		image_directory : str
			images should be on test train and validation folder in the directory pointed by this parameter
		annotation_directory : str
			where is the annotation saved
		anchor_bbox_file : str
			k mean clustered anchor box
		annotation_pattern : str
			patern with one {} that will be formated with test train and validation and match the annotation files
		shape : tuple of int
			shape of the resized image
		*"""
		self.batch_size = batch_size
		self.aug_st = augmention_strength
		self.img_dir = image_directory
		self.annot_dir = annotation_directory
		self.anchors = np.load(anchor_bbox_file)
		# soft augment data 10 times and strong 20 times
		epochs_mult = (["None", "soft", "strong"].index(self.aug_st) * 10) + 1
		self.shape = shape
		self.classes = classes
		self.nclass = len(self.classes)
		self.bbox_per_cell = bbox_per_cell
		self.bbox_size = self.nclass + 5
		self.head_shape = head_shape
		self.annotation_pattern = annotation_pattern

		'''
		def show_current_batch(self):
			for i in range(self.batch_size):
				current_data = self.train_data[self.current_index + i]
				im = cv2.imread(self.img_dir + "train/" + current_data[0] + ".jpg")
				im = cv2.resize(im, (608, 608), interpolation = cv2.INTER_AREA)
				sy, sx, _ = im.shape
				for b in current_data[1].values.tolist():
					col = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
					lab, cx, cy, w, h = b[1:]
					start, end = (int((cx - (w / 2)) * sx), int((cy - (h / 2)) * sy)), (int((cx + (w / 2)) * sx), int((cy + (h / 2)) * sy))
					im = cv2.rectangle(im.copy(), start, end, col, 2)
					cv2.putText(im, lab, (int((cx - (w / 4)) * sx), int(cy * sy)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
				cv2.imshow('boxed', im)
				cv2.waitKey(0)
				cv2.destroyAllWindows()
		'''
	@tf.function
	def read_image(self, image_path):
		image = tf.io.read_file(image_path)
		# tensorflow provides quite a lot of apis for io
		image = tf.image.decode_image(image, channels = 3, dtype = tf.float32)
		return image

	@tf.function
	def normalize(self, image):
		image = (image - tf.reduce_min(image))/(tf.reduce_max(image) - tf.reduce_min(image))
		image = (2 * image) - 1
		return image

	@tf.function
	def augment(self, image):
		crop_size = int(self.shape[0] / 1.5)
		image = tf.image.random_crop(image, (crop_size, crop_size, 3))
		image = tf.image.resize(image, self.shape)
		image = tf.image.random_saturation(image, 0.5, 2.0)
		image = tf.image.random_brightness(image, 0.5)
		return image

	def get_label_data(self, _type="train"):
		if ["train", "test", "validation"].count(_type) == 0:
			raise Exception("type data should be train, test or validation")
		if (_type == "train"):
			return [self.preprocess_y(np.array(i[1].values.tolist())[:, 1:]) for i in tqdm(pd.read_csv(self.annot_dir + self.annotation_pattern.format("train"), dtype=str).groupby("ImageID"))]
		elif (_type == "test"):
			return [self.preprocess_y(np.array(i[1].values.tolist())[:, 1:]) for i in pd.read_csv(self.annot_dir + self.annotation_pattern.format("test"), dtype=str).groupby("ImageID")]
		return [self.preprocess_y(np.array(i[1].values.tolist())[:, 1:]) for i in pd.read_csv(self.annot_dir + self.annotation_pattern.format("validation"), dtype=str).groupby("ImageID")]

	def get_dir(self, _type="train"):
		if ["train", "test", "validation"].count(_type) == 0:
			raise Exception("type data should be train, test or validation")
		return sorted(glob.glob(os.path.join(self.img_dir, _type, '*.jpg')))
	
	@tf.function
	def preprocess_X_y(self, image_path, label):
		"""
		No augmentation for the moment
		"""
		X = self.read_image(image_path)
		X = self.augment(X)
		X = self.normalize(X)
		
		label = label[:, 1:]
		label = tf.strings.to_number(label[:, 1:], tf.float32)
		label = tf.concat([class_id, label], 1)
		return X, label

		'''
		def preprocess_X_y(self, progress=True, type_data="train"):
			"""
			No augmentation for the moment
			"""
			if type_data=="train":
				current_data = self.train_data[self.current_index]
				self.current_index += 1
			elif type_data == "test":
				current_data = self.test_data[self.current_index_test]
				self.current_index_test += 1
			else:
				raise Exception("type data should be train or test")
			X = cv2.resize(cv2.imread(self.img_dir + type_data + "/" + current_data[0] + ".jpg"), 
						self.shape, interpolation = cv2.INTER_LINEAR)
			return np.array(X), np.array(current_data[1].iloc[:, 1:].values.tolist())
		'''

	def preprocess_y(self, boxes):
		num_data = np.array(boxes[:, 1:5], dtype=np.float)
		final = np.zeros((np.sum([sh[1] * sh[2] for sh in self.head_shape]) * self.bbox_per_cell, 6))
		for j, d in enumerate(num_data):
			closest_clust = 0
			dist = float('inf')
			for i, c in enumerate(self.anchors):
				tmp = euclidian_dist(d[2:4], c)
				if tmp < dist:
					dist = tmp
					closest_clust = i
			head = int(np.trunc(closest_clust / 2))
			conf = 1
			cell = 1
			class_index = self.classes.index(boxes[j][0])
			cx, cy = np.array(num_data[j][:2])
			head_w, head_h = self.head_shape[head][1:3]
			x, y = int(np.trunc(cx / (1 / head_w))), int(np.trunc(cy / (1 / head_h)))
			indice = int(np.sum([self.head_shape[hm][1] * self.head_shape[hm][2] * self.bbox_per_cell for hm in range(head)]) + y * self.head_shape[head][1] + x)
			while final[indice][0] == 1 and cell < self.bbox_per_cell:
				indice += self.head_shape[head][1] * self.head_shape[head][2]
				cell += 1
			if final[indice][0] != 1:
				final[indice] = [conf] + list(d) + [class_index]
		print(final.shape)
		return final

	def yolo_output_to_bbox(self, preds):
		boxes = []
		for b in range(self.batch_size):
			bbox_all = []
			for i in range(3):
				h, w = preds[i].shape[1:3]
				ah, aw = 1 / h, 1 / w
				bh, bw = ah / 2, aw / 2
				for c in range(self.bbox_per_cell):
					raw_data = tf.concat([
							preds[i][b][:, :, c : c + 1], 
							preds[i][b][:, :, self.bbox_per_cell + c * 4 : self.bbox_per_cell + c * 4 + 4], 
							preds[i][b][:, :, self.bbox_per_cell + 4 * self.bbox_per_cell + c * self.nclass : self.bbox_per_cell + 4 * self.bbox_per_cell + c * self.nclass + self.nclass]
						], axis=-1
					)
					conf = raw_data[:, :, 0:1]
					col, row = tf.meshgrid(tf.range(w), tf.range(h))
					col = tf.expand_dims(tf.cast(col, dtype=tf.float32), -1)
					row = tf.expand_dims(tf.cast(row, dtype=tf.float32), -1)
					mid_x = col * aw + bw
					mid_y = row * ah + bh
					_x = mid_x + raw_data[:, :, 1:2]
					_y = mid_x + raw_data[:, :, 2:3]
					_w = self.anchors[i + c][0] + raw_data[:, :, 3:4]
					_h = self.anchors[i + c][1] + raw_data[:, :, 4:5]
					labs = raw_data[:, :, 5:]
					box_indexes = row * w + col
					result = tf.concat([conf, _x, _y, _w, _h, labs, box_indexes], -1)
					result = tf.reshape(result, [-1, result.shape[-1]])
					bbox_all.append(result)
			bbox_all = tf.expand_dims(tf.concat(bbox_all, 0), 0)
			boxes.append(bbox_all)
		final = tf.concat(boxes, 0)
		return final
