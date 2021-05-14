import cv2
import tifffile as tiff
import numpy as np
import torch
import matplotlib.pyplot as plt


original = tiff.imread("./k_fold/fold_1/train_x/aaa6a05cc.tiff")
def crop_image(origin_image, patch_size=256, overlap=0.5):

    origin_patch = origin_image[:256, :256, :] / 255.0
    next_patch = origin_image[7500:7500+256, 2500:2500+256, :] / 255.0
    plt.imshow(origin_patch)
    plt.show()
    plt.imshow(next_patch)
    plt.show()
    img1 = torch.from_numpy(origin_patch).permute(2, 0, 1).unsqueeze(0).cuda()
    img2 = torch.from_numpy(next_patch).permute(2, 0, 1).unsqueeze(0).cuda()
    print(ssim(img1, img2, window_size=16))




crop_image(original)
