import os
import cv2
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import shutil

data_root = 'data/'
save_path = 'annotation/'


def FileList_Generation():
    # Update your data path below
    data_name_list = [
        # e.g., (path_to_dataset, label)
        ('AIDraw_1122/Fake/MidjourneyV5', 1),  # Fake Dataset
        ('AIDraw_1122/Real/AhmedKaram', 0),  # Real Dataset
    ]

    img_list = []
    for data_name, label in data_name_list:
        if label == 0:
            continue
        path = '%s/' % data_name
        flist = sorted(os.listdir(data_root + path))
        for file in tqdm(flist):
            img_list.append((path + file, label))
    img_list2 = []
    for data_name, label in data_name_list:
        if label == 1:
            continue
        path = '%s/' % data_name
        flist = sorted(os.listdir(data_root + path))
        np.random.shuffle(flist)
        flist = flist[:len(img_list)]
        for file in tqdm(flist):
            img_list2.append((path + file, label))
    np.random.shuffle(img_list2)
    img_list2 = img_list2[:len(img_list)]
    img_list = img_list + img_list2

    print('#Images: %d' % len(img_list))
    textfile = open(save_path + 'Test_num%d.txt' % (len(img_list)), 'w')
    for item in img_list:
        textfile.write('%s %s\n' % (item[0], item[1]))
    textfile.close()


if __name__ == '__main__':
    # generate file list for training/testing
    # E.g., Train_VISION.txt contains [[image_path_1, image_label_1], [image_path_2, image_label_2], ...]
    FileList_Generation()
