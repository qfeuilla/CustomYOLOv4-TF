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
        self.current_index = 0
        self.current_index_test = 0
        self.train_data = [i for i in pd.read_csv(self.annot_dir + annotation_pattern.format("test")).groupby("ImageID")] # DONT FORGET TO PUT train BACK !!!!!
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
        self.head_shape = head_shape
        self.reset_dataset()

    def reset_dataset(self):
        np.random.shuffle(self.train_data)
        np.random.shuffle(self.test_data)
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
    
    def get_X_y(self, progress=True, type_data="train"):
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

    def get_batch(self, type_data="train"):
        return [self.get_X_y(type_data=type_data) for _ in range(self.batch_size)]
    
    def epoch_tqdm(self):
        return tqdm(range(0, self.epochs_length, self.batch_size))

    def preprocess_y(self, boxes):
        final = []
        w_h = np.array(boxes[:, 3: 5], dtype=np.float)
        num_data = np.array(boxes[:, 1: 5], dtype=np.float)
        for j, d in enumerate(num_data):
            one = []
            closest_clust = 0
            dist = float('inf')
            for i, c in enumerate(self.anchors):
                tmp = euclidian_dist(d[2:4], c)
                if tmp < dist:
                    dist = tmp
                    closest_clust = i
            head = np.trunc(closest_clust / 2)
            conf = 1
            class_index = self.classes.index(boxes[j][0])
            cx, cy = np.array(num_data[j][:2])
            head_w, head_h = self.head_shape[int(head)][1:3]
            x, y = np.trunc(cx / (1 / head_w)), np.trunc(cy / (1 / head_h))
            indice = int(y * head_w + x)
            one = [conf] + list(d) + [class_index] + [(int(head), indice)]
            final.append(one)
        final = np.array(final)
        print(final)
        return final

    def yolo_output_to_bbox(self, preds, conf_discard_treshold=0):
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
                            if (conf >= conf_discard_treshold):
                                mid = [aw * x + bw, ah * y + bh]
                                _x, _y = [mid[_i] + cell_data[_i + 1] for _i in range(2)]
                                _w, _h = [self.anchors[c][_i] + cell_data[i + 3] for _i in range(2)]
                                labs = cell_data[5:]
                                loc = (i, y * h + x) # (which_head, grid_index)
                                bbox_all.append([conf, _x, _y, _w, _h] + [labs] + [loc])
            boxes.append(bbox_all)
        return np.array(boxes, dtype=object)
