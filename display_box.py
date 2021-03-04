DIR = "../data/train/"
NUM = "004242"

import json
import cv2
from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt

def display_box(image_num):
    boxes = []
    with open(DIR + "annos/" + image_num + ".json") as file:
        data = json.load(file)
        for k in data.keys():
            try:
                boxes.append(data[k]['bounding_box'])
            except: 
                continue
    print(boxes)
    img = np.array(Image.open(DIR + "image/" + image_num + ".jpg"))
    for b in boxes:
        x1, y1, x2, y2 = b
        cv2.rectangle(img,(x1,y1),(x2,y2),(random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)),6)
    plt.imshow(img)
    plt.show()

for _ in range(10):
    i = random.randint(0, 10000)
    print(i)
    display_box(str(i).rjust(6, '0'))
