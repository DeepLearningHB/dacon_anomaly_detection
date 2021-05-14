import torch
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import glob
import shutil
from sklearn.model_selection import KFold
from environments import *
import tifffile as tiff
import cv2
import ssim

from itertools import product


def makedirs(basepath, k_fold=4):
    os.makedirs(basepath, exist_ok=True)
    fold_name = "fold"
    fold_name_list = [fold_name+"_"+str(i) for i in range(k_fold)]
    sub_folder_list = ["train_x", "train_y", "valid_x", "valid_y"]
    for i in range(k_fold):
        folder_name = os.path.join (basepath, fold_name + "_" + str (i + 1))
        os.makedirs(folder_name, exist_ok=True)
        for sub in sub_folder_list:
            os.makedirs(os.path.join(folder_name, sub), exist_ok=True)
    print("Successfully made folders.")
    return fold_name_list


def copy_dataset_to_hb(to_base, train_x, train_y):
    X = np.array(train_x)
    y = np.array(train_y)
    kf = KFold(n_splits=k_fold)
    kf.get_n_splits(X)
    print('Making cross validation set')
    for idx, (train_index, test_index) in enumerate(kf.split(X)):
        sub_train_X, sub_train_Y = X[train_index], y[train_index]
        sub_test_X, sub_test_Y = X[test_index], y[test_index]
        for sub_x, sub_y in zip(sub_train_X, sub_train_Y):
            shutil.copy(sub_x, os.path.join(to_base, "fold_"+str(idx+1)+"/train_x"))
            shutil.copy(sub_y, os.path.join(to_base, "fold_"+str(idx+1)+"/train_y"))

        for sub_x, sub_y in zip(sub_test_X, sub_test_Y):
            shutil.copy(sub_x, os.path.join(to_base, "fold_"+str(idx+1)+"/valid_x"))
            shutil.copy(sub_y, os.path.join(to_base, "fold_"+str(idx+1)+"/valid_y"))


def cal_ssim(img1, img2):
    img1 = img1 / 255.0
    img2 = img2 / 255.0
    img1 = torch.from_numpy (img1).permute (2, 0, 1).unsqueeze (0).cuda ()
    img2 = torch.from_numpy (img2).permute (2, 0, 1).unsqueeze (0).cuda ()
    return ssim.ssim(img1, img2, window_size=16)


def read_tiff(image_file):
    """
    *data size*
    e.g.) (3, w, h) or (1,1,3,w,h) or (w, h, 3)  --> transform --> (w, h, 3)
    """
    image = tiff.imread(image_file)
    if image.shape[0] == 1:
        image = image[0][0]
        image = image.transpose(1, 2, 0)
        image = np.ascontiguousarray(image)
    elif image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
        image = np.ascontiguousarray(image)
    return image


def blackout_coordinates(image):
    blackout_list = []

    index_upper = -1
    for i in range(len(image)):
        if np.sum(image[i]) == 0:
            index_upper = i
        else:
            break
    blackout_list.append(index_upper+1)

    index_left = -1
    for i in range(len(image[0])):
        if np.sum(image[:, i, :]) == 0:
            index_left = i
        else:
            break
    blackout_list.append(index_left+1)

    index_lower = len(image) - 1
    for i in reversed(range(len(image))):
        if np.sum(image[i]) == 0:
            index_lower = i
        else:
            break
    blackout_list.append(index_lower)

    index_right = len(image[0]) - 1
    for i in reversed(range(len(image[0]))):
        if np.sum (image[:, i, :]) == 0:
            index_right = i
        else:
            break
    blackout_list.append(index_right)
    return blackout_list # up, left, down, right

def crop_image(origin_image, origin_filename, seg_image, seg_filename, patch_size=480, overlap=0.5, scale_factor=0.25, threshold=0.85):
    blackout = blackout_coordinates(origin_image)
    origin_image = origin_image[blackout[0]:blackout[2], blackout[1]:blackout[3], :]
    seg_image = seg_image[blackout[0]:blackout[2], blackout[1]:blackout[3], :]
    origin_image = cv2.resize(origin_image, (int(origin_image.shape[1] * scale_factor), int(origin_image.shape[0] * scale_factor)), interpolation=cv2.INTER_AREA)
    seg_image = cv2.resize(seg_image, (int(seg_image.shape[1] * scale_factor), int(seg_image.shape[0] * scale_factor)), interpolation=cv2.INTER_AREA)
    seg_image = np.where(seg_image != 0, 255, 0)
    offset = 35
    basis_patch = origin_image[offset:patch_size+offset, offset:patch_size+offset, :]

    base_path_x = "/".join(origin_filename.split("/")[:-1])
    base_path_y = "/".join(seg_filename.split("/")[:-1])
    filename_only = origin_filename.split("/")[-1].split(".")[0]
    x_list = [i for i in range(0, len(origin_image[0]) - patch_size, int(patch_size * (1 - overlap)))]
    y_list = [i for i in range(0, len(origin_image) - patch_size, int(patch_size * (1 - overlap)))]
    count = 0

    for idx_x, x in enumerate(x_list):
        for idx_y, y in enumerate(y_list):
            patch_image = origin_image[y:y+patch_size, x:x+patch_size, :]
            ssim_value = cal_ssim(patch_image, basis_patch)
            if ssim_value < threshold and np.count_nonzero(patch_image==0) < 7680:
                count += 1
                target_patch = seg_image[y:y+patch_size, x:x+patch_size, :]
                folder_name = "p_"+str(patch_size)+"_s_"+str(int(scale_factor*100))+"_o_"+str(int(overlap*100))+"_new"
                os.makedirs(os.path.join(base_path_x, folder_name), exist_ok=True)
                os.makedirs(os.path.join(base_path_y, folder_name), exist_ok=True)
                tiff.imwrite(os.path.join(base_path_x, folder_name, filename_only+"_"+"%.2f"%(ssim_value)+"_"+str(count)+".tiff"), patch_image)
                cv2.imwrite(os.path.join(base_path_y, folder_name, filename_only+"_"+"%.2f"%(ssim_value)+"_"+str(count)+".png"), target_patch)

def crop_image_for_valid(origin_image, origin_filename, seg_image, seg_filename, patch_size=480, overlap=0.5,
               scale_factor=0.25, threshold=0.85):
    origin_image = cv2.resize (origin_image, (
    int (origin_image.shape[1] * scale_factor), int (origin_image.shape[0] * scale_factor)),
                               interpolation=cv2.INTER_AREA)
    seg_image = cv2.resize (seg_image,
                            (int (seg_image.shape[1] * scale_factor), int (seg_image.shape[0] * scale_factor)),
                            interpolation=cv2.INTER_AREA)
    seg_image = np.where (seg_image != 0, 255, 0)
    h, w, _ = origin_image.shape
    base_path_x = "/".join (origin_filename.split ("/")[:-1])
    base_path_y = "/".join (seg_filename.split ("/")[:-1])
    filename_only = origin_filename.split ("/")[-1].split (".")[0]
    x_list = [i for i in range (0, len (origin_image[0]), int (patch_size * (1 - overlap)))]
    y_list = [i for i in range (0, len (origin_image), int (patch_size * (1 - overlap)))]
    count = 0

    for idx_x, x in enumerate (x_list):
        for idx_y, y in enumerate (y_list):
            patch_image = origin_image[y:y + patch_size, x:x + patch_size, :]
            target_patch = seg_image[y:y + patch_size, x:x + patch_size, :]

            if patch_image.shape[0] != patch_size: # 480 - 240
                remain_size = patch_size - patch_image.shape[0]
                additional_patch = np.zeros((remain_size, patch_image.shape[1], 3), dtype=np.uint8)
                patch_image = np.vstack([patch_image, additional_patch])
                target_patch = np.vstack([target_patch, additional_patch])
            if patch_image.shape[1] != patch_size:
                remain_size = patch_size - patch_image.shape[1]
                additional_patch = np.zeros((patch_image.shape[0], remain_size, 3), dtype=np.uint8)
                patch_image = np.hstack([patch_image, additional_patch])
                target_patch = np.hstack ([target_patch, additional_patch])
            folder_name = "p_" + str (patch_size) + "_s_" + str (int (scale_factor * 100)) + "_o_" + str (
                    int (overlap * 100))+"_new"
            os.makedirs (os.path.join (base_path_x, folder_name), exist_ok=True)
            os.makedirs (os.path.join (base_path_y, folder_name), exist_ok=True)
            tiff.imwrite (os.path.join (base_path_x, folder_name,
                                        filename_only + "_" + str (idx_y+1) + "_" + str(len(y_list)) + "_" + str(idx_x+1)
                                        + "_" + str(len(x_list)) + "_" + str(h)+"_" + str(w) +".tiff"), patch_image)

            cv2.imwrite (os.path.join (base_path_y, folder_name,
                                        filename_only + "_" + str (idx_y+1) + "_" + str(len(y_list)) + "_" + str(idx_x+1)
                                        + "_" + str(len(x_list)) + "_" + str(h)+"_" + str(w) + ".png"), target_patch)

def crop_image_for_test(origin_image, origin_filename, patch_size=640, overlap=0.5,
                         scale_factor=0.25, threshold=0.85):
    origin_image = cv2.resize (origin_image, (
        int (origin_image.shape[1] * scale_factor), int (origin_image.shape[0] * scale_factor)),
                               interpolation=cv2.INTER_AREA)
    # seg_image = cv2.resize (seg_image,
    #                         (int (seg_image.shape[1] * scale_factor), int (seg_image.shape[0] * scale_factor)),
    #                         interpolation=cv2.INTER_AREA)
    # seg_image = np.where (seg_image != 0, 255, 0)
    h, w, _ = origin_image.shape
    base_path_x = "/".join (origin_filename.split ("/")[:-1])
    #base_path_y = "/".join (seg_filename.split ("/")[:-1])
    filename_only = origin_filename.split ("/")[-1].split (".")[0]
    x_list = [i for i in range (0, len (origin_image[0]), int (patch_size * (1 - overlap)))]
    y_list = [i for i in range (0, len (origin_image), int (patch_size * (1 - overlap)))]
    count = 0

    for idx_x, x in enumerate (x_list):
        for idx_y, y in enumerate (y_list):
            patch_image = origin_image[y:y + patch_size, x:x + patch_size, :]
           #  target_patch = seg_image[y:y + patch_size, x:x + patch_size, :]

            if patch_image.shape[0] != patch_size:  # 480 - 240
                remain_size = patch_size - patch_image.shape[0]
                additional_patch = np.zeros ((remain_size, patch_image.shape[1], 3), dtype=np.uint8)
                patch_image = np.vstack ([patch_image, additional_patch])
                # target_patch = np.vstack ([target_patch, additional_patch])
            if patch_image.shape[1] != patch_size:
                remain_size = patch_size - patch_image.shape[1]
                additional_patch = np.zeros ((patch_image.shape[0], remain_size, 3), dtype=np.uint8)
                patch_image = np.hstack ([patch_image, additional_patch])
               #  target_patch = np.hstack ([target_patch, additional_patch])
            folder_name = "p_" + str (patch_size) + "_s_" + str (int (scale_factor * 100)) + "_o_" + str (
                int (overlap * 100)) + "_new"
            os.makedirs (os.path.join (base_path_x, folder_name), exist_ok=True)
            #os.makedirs (os.path.join (base_path_y, folder_name), exist_ok=True)
            tiff.imwrite (os.path.join (base_path_x, folder_name,
                                        filename_only + "_" + str (idx_y + 1) + "_" + str (
                                            len (y_list)) + "_" + str (idx_x + 1)
                                        + "_" + str (len (x_list)) + "_" + str (h) + "_" + str (w) + ".tiff"),
                          patch_image)

            # cv2.imwrite (os.path.join (base_path_y, folder_name,
            #                            filename_only + "_" + str (idx_y + 1) + "_" + str (
            #                                len (y_list)) + "_" + str (idx_x + 1)
            #                            + "_" + str (len (x_list)) + "_" + str (h) + "_" + str (w) + ".png"),
            #              target_patch)

    # next_patch = origin_image[256:256+256, 256:256+256, :]
    # img1 = torch.from_numpy(origin_patch).permute(0, 3, 1, 2).cuda()
    # img2 = torch.from_numpy(next_patch).permute(0, 3, 1, 2).cuda()
    # print(pytorch_ssim.ssim(img1, img2, window_size=16))


def main1():
    origin_dataset_path = '../k_fold/origin'
    assert os.path.exists(origin_dataset_path), 'Cannot find directory, check again'
    makedirs('k_fold', k_fold=5)
    train_tiff_filelist = glob.glob(os.path.join(origin_dataset_path, "data/*.tiff"))
    #train_mask_filelist = [os.path.join("/".join(a.split("/")[:-1]), a.split("/")[-1].split(".")[0]+".mask.png") for a in train_tiff_filelist]
    train_mask_filelist = glob.glob (os.path.join (origin_dataset_path, "label/*.mask.png"))
    copy_dataset_to_hb('../k_fold', train_tiff_filelist, train_mask_filelist)


def test_main():
    base_path = sorted(glob.glob('../k_fold/*_*'))
    print(base_path)
    for path in base_path:
        train_x = sorted(glob.glob(os.path.join(path, "train_x/*.*")))
        train_y = sorted(glob.glob(os.path.join(path, "train_y/*.*")))
        count = 0

        for x, y in zip(train_x, train_y):

            tiff_ = tiff.imread(x).squeeze()
            png_ = cv2.imread(y)
            print (x, "complete")
            t_h, t_w, t_c = tiff_.shape
            h, w, c = png_.shape
            t_shape_list = [t_h, t_w, t_c]
            png_shape_list = [h, w, c]
            shape_revision = []
            for i, t in enumerate(png_shape_list):
                for j, s in enumerate(t_shape_list):
                    if t == s:
                        shape_revision.append(j)
            tiff_ = tiff_.transpose((shape_revision[0], shape_revision[1], shape_revision[2]))

            crop_image(tiff_, x, png_, y)

        valid_x = sorted(glob.glob(os.path.join(path, "valid_x/*.*")))
        valid_y = sorted(glob.glob(os.path.join(path, "valid_y/*.*")))

        for x, y in zip(valid_x, valid_y):
            tiff_ = tiff.imread(x).squeeze()
            png_ = cv2.imread(y)
            print(x, 'loaded')
            t_h, t_w, t_c = tiff_.shape
            h, w, c = png_.shape
            t_shape_list = [t_h, t_w, t_c]
            png_shape_list = [h, w, c]
            shape_revision = []
            for i, t in enumerate(png_shape_list):
                for j, s in enumerate(t_shape_list):
                    if t == s:
                        shape_revision.append(j)
            tiff_ = tiff_.transpose((shape_revision[0], shape_revision[1], shape_revision[2]))

            crop_image_for_valid(tiff_, x, png_, y, overlap=0)

       # if "fold_1" not in path:



def test_main_2():
    base_path = sorted(glob.glob('../hubmap_test/*'))
    print(base_path)
    for path in base_path:
        # if path.split("/")[-1] != "fold_1":
        #     continue
        # train_x = sorted(glob.glob(os.path.join(path, "train_x/*.*")))
        # train_y = sorted(glob.glob(os.path.join(path, "train_y/*.*")))
        # count = 0
        # for x, y in zip(train_x, train_y):
        #
        #     tiff_ = tiff.imread(x).squeeze()
        #     png_ = cv2.imread(y)
        #     print (x, "complete")
        #     t_h, t_w, t_c = tiff_.shape
        #     h, w, c = png_.shape
        #     t_shape_list = [t_h, t_w, t_c]
        #     png_shape_list = [h, w, c]
        #     shape_revision = []
        #     for i, t in enumerate(png_shape_list):
        #         for j, s in enumerate(t_shape_list):
        #             if t == s:
        #                 shape_revision.append(j)
        #     tiff_ = tiff_.transpose((shape_revision[0], shape_revision[1], shape_revision[2]))
        #
        #     crop_image(tiff_, x, png_, y)

        valid_x = sorted(glob.glob(os.path.join('../hubmap_test', "*.tiff")))
        # valid_y = sorted(glob.glob(os.path.join(path, "valid_y/*.*")))
        print(valid_x)
        for x in valid_x:
            print(x)
            tiff_ = read_tiff(x)
            #png_ = cv2.imread(y)
            print(x, 'loaded')
            t_h, t_w, t_c = tiff_.shape
            # h, w, c = png_.shape
            t_shape_list = [t_h, t_w, t_c]
            #png_shape_list = [h, w, c]
            shape_revision = []
            # for i, t in enumerate(png_shape_list):
            #     for j, s in enumerate(t_shape_list):
            #         if t == s:
            #             shape_revision.append(j)
            # tiff_ = tiff_.transpose((shape_revision[0], shape_revision[1], shape_revision[2]))

            crop_image_for_test(tiff_, x, overlap=0)
        exit()

if __name__ == "__main__":
    test_main_2()
