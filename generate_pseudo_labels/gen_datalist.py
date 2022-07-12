from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import glob
import os
import numpy as np


def gentxt(data_root, outfile, outsider="", mask="" , aug="", porpotion=[-1, -1, 0.1]):         # generate data file via traveling the target dataset 
    '''
    Use ImageFolder method to travel the target dataset 
    Save to two files including ".label", ".labelpath", ".labelfeature"
    porpotion = [mask, aug, outsider] | -1 = add all if available
    '''
    # output file1
    outfile1 = open(outfile, 'w') # names of images
    # output file2
    outfile2 = open(outfile+'path', 'w') # path to images + their class
    outfile3 = open(outfile+'feature', 'w') # path to features file + their class
    data = ImageFolder(data_root)
    outsider_data = []
    aug_data = {} # {class: [list of images]}
    mask_data = {} # {class: [list of images]}

    if len(outsider)>0:
        outsider_data = list(glob.iglob(os.path.join(outsider, "*.jpg")))
    if len(aug)>0:
        list_classes = os.listdir(aug)
        for cls in list_classes:
            aug_data[cls] = list(glob.iglob(os.path.join(os.path.join(aug, cls), "*.jpg")))
    if len(mask)>0:
        list_classes = os.listdir(mask)
        for cls in list_classes:
            mask_data[cls] = list(glob.iglob(os.path.join(os.path.join(mask, cls), "*.jpg")))

    last_class = -1
    last_class_num = 0
    count = 0
    tqdm(data.imgs)
    # travel the target dataset
    for index, value in enumerate(data.imgs):
        this_class = value[0].split("/")[-2]
        if this_class != last_class:
            _last_class = str(last_class)
            # add outsider
            if len(outsider_data)>0:
                out_add_num = int(np.ceil(last_class_num*porpotion[2]))
                for _ in range(out_add_num):
                    if len(outsider_data)<=0:
                        break
                    fake_img = '/' + f"{last_class}" + '/' + str(outsider_data[0].split('/')[-1:][0]) # fake /class/name.jpg for later people name identifying
                    outfile1.write(fake_img + '\n')
                    outfile2.write(outsider_data[0] + '\t' + str(last_class) + '\n')
                    outfile3.write(outsider_data[0].replace(outsider_data[0].split("/")[-2], outsider_data[0].split("/")[-2]+"_feature").replace(".jpg", ".npy") + '\t' + str(last_class) + '\n')
                    del outsider_data[0]
                    count +=1
            # add mask
            if mask_data.get(_last_class, None) != None:
                mask_add_num = min(int(np.ceil(last_class_num*porpotion[0])), len(mask_data[_last_class])) if porpotion[0] >0 else len(mask_data[_last_class])
                for _ in range(mask_add_num):
                    if len(mask_data[_last_class])<=0:
                        break
                    fake_img = '/' + f"{last_class}" + '/' + str(mask_data[_last_class][0].split('/')[-1:][0]) # fake /class/name.jpg for later people name identifying
                    outfile1.write(fake_img + '\n')
                    outfile2.write(mask_data[_last_class][0] + '\t' + str(last_class) + '\n')
                    outfile3.write(mask_data[_last_class][0].replace(mask_data[_last_class][0].split("/")[-3], mask_data[_last_class][0].split("/")[-3]+"_feature").replace(".jpg", ".npy") + '\t' + str(last_class) + '\n')
                    del mask_data[_last_class][0]
                    count +=1
            # add aug
            if aug_data.get(_last_class, None) != None:
                aug_add_num = min(int(np.ceil(last_class_num*porpotion[1])), len(aug_data[_last_class])) if porpotion[0] >0 else len(aug_data[_last_class])
                for _ in range(aug_add_num):
                    if len(aug_data[_last_class])<=0:
                        break
                    fake_img = '/' + f"{last_class}" + '/' + str(aug_data[_last_class][0].split('/')[-1:][0]) # fake /class/name.jpg for later people name identifying
                    outfile1.write(fake_img + '\n')
                    outfile2.write(aug_data[_last_class][0] + '\t' + str(last_class) + '\n')
                    outfile3.write(aug_data[_last_class][0].replace(aug_data[_last_class][0].split("/")[-3], aug_data[_last_class][0].split("/")[-3]+"_feature").replace(".jpg", ".npy") + '\t' + str(last_class) + '\n')
                    del aug_data[_last_class][0]
                    count +=1

            last_class = this_class
            last_class_num = 0

        count += 1
        last_class_num += 1
        img = '/'+'/'.join(value[0].split('/')[-2:])
        outfile1.write(img + '\n')
        outfile2.write(value[0] + '\t' + str(value[1]) + '\n')
        outfile3.write(value[0].replace(data_root.split("/")[-1], data_root.split("/")[-1]+"_feature").replace(".jpg", ".npy") + '\t' + str(this_class) + '\n')

    return count
    
if __name__ == "__main__":                  # obtain data list
    '''
    This method is to obtain data list from dataset
    and save to txt files
    '''
    outfile = './DATA.label'
    data_root = '/Data/DeepGlint_asian'
    aug_root = '/Data/DeepGlint_asian_aug'
    mask_root = '/Data/DeepGlint_asian_mask'
    outsider_root = '/Data/outsider_save'
    print(gentxt(data_root, outfile, outsider_root, mask_root, aug_root))
