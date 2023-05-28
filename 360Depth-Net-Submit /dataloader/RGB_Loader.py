import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import random
from PIL import Image
import numpy as np
import dataloader.preprocess
from dataloader import preprocess
import cv2
import imutils

# ---------------- Read our images --------------------------------
# Distortion removal operator
def transf(nb_pixels ,arr):
    new_arr = np.zeros(arr.shape)
    new_arr[:, nb_pixels:len(arr[0]), :] = arr[:, 0:len(arr[0])-nb_pixels, :]
    new_arr[:,0:nb_pixels,:] = arr[:, len(arr[0])-nb_pixels:len(arr[0]),:]
    return new_arr.astype(int)

# RGB images
def default_loader(path, u_d):
    image = cv2.imread(path)
    if u_d == "up":
        # Translation to make Camera and LiDAR have the same "Center"
        upper_image = transf(2340, image).astype('uint8')
        # Croping higher and lower area of the image that can't be labeled by the LiDAR
        upper_image = cv2.resize(upper_image, (1024, 512))
        # Isolation the target vertical area
        result = upper_image[97:353, :, :]
        # Resizing final image as LiDAR image resolution
        # result = cv2.resize(upper_image, (1024, 64))

    elif u_d == "down":
        lower_image = transf(2360, image).astype('uint8')
        lower_image = cv2.resize(lower_image, (1024, 512))
        result = lower_image[87:343, :, :]
        # result = cv2.resize(lower_image, (1024, 64))

    return result

# ---------------- Read our Depth images --------------------------------
def disparity_loader(path):
    img = (cv2.imread(path, cv2.IMREAD_UNCHANGED))
    if path.find("new street") <0 :
        img = imutils.rotate(img, 180)
    # print(type(img))
    # print(img.dtype)
    return img
# --------------------------------------------------------------------



class myImageFolder(data.Dataset):
    def __init__(self,
                 equi_infos,
                 up,
                 down,
                 up_disparity,
                 training,
                 loader=default_loader,
                 dploader=disparity_loader):

        self.up = up
        self.down = down
        self.disp_name = up_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training
        self.equi_infos = equi_infos

    def __getitem__(self, index):
        up = self.up[index]
        down = self.down[index]
        disp_name = self.disp_name[index]
        equi_info = self.equi_infos

        # ------------------------ Ours
        up_img = self.loader(up, "up")
        down_img = self.loader(down, "down")
        # ----------------------------
        disp = self.dploader(disp_name)
        # Wrap images with polar angle, "early fusion"
        up_img = np.concatenate([np.array(up_img), equi_info], 2)
        down_img = np.concatenate([np.array(down_img), equi_info], 2)

        if self.training:
            h, w = up_img.shape[0], up_img.shape[1]
            th, tw = 256, 256 # target input size

            # vertical remaining cropping
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            up_img = up_img[y1:y1 + th, x1:x1 + tw, :]
            down_img = down_img[y1:y1 + th, x1:x1 + tw, :]
            disp = np.ascontiguousarray(disp, dtype=np.float32)
            # Convert Depth pixel values to meter
            disp = disp[y1:y1 + th, x1:x1 + tw] * 4 / 1000

            # preprocessing
            processed = preprocess.get_transform(augment=False)
            up_img = processed(up_img)
            down_img = processed(down_img)

            return up_img, down_img, disp
        else:
            disp = np.ascontiguousarray(disp, dtype=np.float32) * 4 / 1000

            processed = preprocess.get_transform(augment=False)
            up_img = processed(up_img)
            down_img = processed(down_img)

            return up_img, down_img, disp

    def __len__(self):
        return len(self.up)
