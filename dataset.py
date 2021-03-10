import matplotlib as plt
import numpy as np
from PIL import Image
import pandas as pd

class Dataset():
    def __init__(self, batch_size, augmention_strength="None", image_directory="data/", annotation_directory="data_getter_and_preprocess/", anchor_bbox_file="anchor_bbox.npy", annotation_pattern="cleaned_{}_bbox.csv"):
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
        """
        self.batch_size = batch_size
        self.aug_st = augmention_strength
        self.img_dir = image_directory
        self.annot_dir = annotation_directory
        anchors = np.load(anchor_bbox_file)
        self.sm_bbox = anchors[:3]
        self.md_bbox = anchors[3:6]
        self.lg_bbox = anchors[6:]
        self.train_data = [i for i in pd.read_csv(self.annot_dir + annotation_pattern.format("train")).groupby("ImageID")]
        self.test_data = [i for i in pd.read_csv(self.annot_dir + annotation_pattern.format("test")).groupby("ImageID")]
        self.validation_data = [i for i in pd.read_csv(self.annot_dir + annotation_pattern.format("validation")).groupby("ImageID")]
        # soft augment data 10 times and strong 20 times
        epochs_mult = (["None", "soft", "strong"].index(self.aug_st) * 10) + 1
        self.epochs_length = len(self.train_data) * epochs_mult
        np.random.shuffle(self.train_data)
        print(self.train_data[0])

Dataset(4)