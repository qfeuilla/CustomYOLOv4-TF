import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import cv2
from tqdm import tqdm
import tensorflow as tf

def euclidian_dist(a, b):
    return np.linalg.norm(a - b)

class Dataset():
    def __init__(self, batch_size, augmention_strength="None", image_directory="data/", annotation_directory="data_getter_and_preprocess/", anchor_bbox_file="anchor_bbox.npy", annotation_pattern="cleaned_{}_bbox.csv", shape=(608, 608), classes=['Human face', 'Man', 'Woman', 'Clothing', 'Car', 'Wheel', 'Food', 'Vehicle', 'Boy', 'Girl'], bbox_per_cell=2):
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
        anchors = np.load(anchor_bbox_file)
        self.anchors = anchors
        self.current_index = 0
        self.train_data = [i for i in pd.read_csv(self.annot_dir + annotation_pattern.format("test")).groupby("ImageID")]
        self.test_data = [i for i in pd.read_csv(self.annot_dir + annotation_pattern.format("test")).groupby("ImageID")]
        self.validation_data = [i for i in pd.read_csv(self.annot_dir + annotation_pattern.format("validation")).groupby("ImageID")]
        # soft augment data 10 times and strong 20 times
        epochs_mult = (["None", "soft", "strong"].index(self.aug_st) * 10) + 1
        self.epochs_length = len(self.train_data) * epochs_mult
        self.epochs_length -= self.epochs_length % self.batch_size
        self.shape = shape
        self.classes = classes
        self.nclass = len(self.classes)
        self.bbox_per_cell = bbox_per_cell
        self.bbox_size = self.nclass + 5
        self.reset_dataset()

    def reset_dataset(self):
        np.random.shuffle(self.train_data)
        self.current_index = 0

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
    
    def get_X_y(self, progress=True):
        """
        No augmentation for the moment
        """
        current_data = self.train_data[self.current_index]
        self.current_index += 1
        X = cv2.resize(cv2.imread(self.img_dir + "train/" + current_data[0] + ".jpg"), 
                    self.shape, interpolation = cv2.INTER_LINEAR)
        return X, current_data[1].iloc[:, 1:].values.tolist()

    def get_batch(self):
        return [self.get_X_y() for _ in range(self.batch_size)]
    
    def epoch_tqdm(self):
        return tqdm(range(0, self.epochs_length, self.batch_size))

    def yolo_output_to_bbox(self, preds, conf_discard_treshold=0.2):
        boxes = []
        for b in range(preds[0].shape[0]):
            bbox_all = []
            for i in range(3):
                h, w = preds[i].shape[1:3]
                ah, aw = 1 / h, 1 / w
                bh, bw = ah / 2, aw / 2
                for c in range(self.bbox_per_cell):
                    raw_data = np.concatenate((
                            np.array(preds[i][b][:, :, c : c + 1]), 
                            np.array(preds[i][b][:, :, self.bbox_per_cell + c * 4 : self.bbox_per_cell + c * 4 + 4]), 
                            np.array(preds[i][b][:, :, self.bbox_per_cell + 4 * self.bbox_per_cell + c * self.nclass : self.bbox_per_cell + 4 * self.bbox_per_cell + c * self.nclass + self.nclass])
                        ), axis=-1
                    )
                    for y in range(h):
                        for x in range(w):
                            cell_data = raw_data[y][x]
                            conf = cell_data[0]
                            if (1 >= conf_discard_treshold):
                                mid = [aw * x + bw, ah * y + bh]
                                _x, _y = [mid[i] + cell_data[i + 1] for i in range(2)]
                                _w, _h = [self.anchors[c][i] + cell_data[i + 3] for i in range(2)]
                                labs = cell_data[5:]
                                bbox_all.append([conf, _x, _y, _w, _h] + [labs])
            boxes.append(bbox_all)
        return boxes
