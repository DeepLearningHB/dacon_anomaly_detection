import shutil
import os
import glob

base_path = './k_fold'
target_dataset = 'p_480_s_25_o_50'

folder_list = glob.glob(os.path.join(base_path, "*", "*", target_dataset))

print(folder_list)
for folder in folder_list:
    #print(folder)
    if folder.split("/")[-3] == "fold_1":
        continue
    shutil.rmtree(folder)
    print(folder)