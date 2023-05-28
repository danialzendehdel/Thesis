import os
import glob
import natsort
import math

import numpy as np


def dataloader(filepath, filepath2=None):
    upper_path = "upper_cam/"
    lower_path = "lower_cam/"
    disp_path = "lidar/Range Images/"

    print(filepath + upper_path)

    all_files = [
        img for idx, img in enumerate(os.listdir(filepath + upper_path)) if img.find('.jpg') > -1 and idx % 10 == 0
    ]

    all_files_2 = [
        img for idx, img in enumerate(os.listdir(filepath2 + upper_path)) if img.find('.jpg') > -1 and idx % 10 == 0
    ]

    all_files = natsort.natsorted(all_files)
    all_files_2 = natsort.natsorted(all_files_2)

    all_files = all_files + all_files_2
    print("#total:", len(all_files))

    border = int(math.floor(0.8 * len(all_files)))
    print("border", border)

    train = all_files[:border]
    val = all_files[border:-10]
    print("all", len(all_files), "train", len(train), "val", len(val))

    up_train = [filepath + upper_path + img for img in train]
    down_train = [filepath + lower_path + 'lower' + img[5:] for img in train]

    disp_train = []
    for img in train:
        if len(img[7:-6]) == 4:
            disp_train.append(filepath + disp_path + "frame00" + img[7:-6] + '.png')
        elif len(img[7:-6]) == 3:
            disp_train.append(filepath + disp_path + "frame000" + img[7:-6] + '.png')
        elif len(img[7:-6]) == 2:
            disp_train.append(filepath + disp_path + "frame0000" + img[7:-6] + '.png')
        else:
            disp_train.append(filepath + disp_path + "frame00000" + img[7:-6] + '.png')

    # Validation set
    up_val = [filepath + upper_path + img for img in val]
    down_val = [filepath + lower_path + 'lower' + img[5:] for img in val]
    # with this assumption that the first item has a number more than 1000
    disp_val = [filepath + disp_path + 'frame00' + img[7:-6] + '.png' for img in val]

    print("Train:", "up", len(up_train), "down", len(down_train), "disp", len(disp_train))
    print("Val:", "up", len(up_val), "down", len(down_val), "disp", len(disp_val))

    return up_train, down_train, disp_train, up_val, down_val, disp_val

