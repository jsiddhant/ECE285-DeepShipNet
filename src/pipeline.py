import numpy as np
import pandas as pd

from skimage.data import imread
from skimage.morphology import label

def rle_to_mask(rle_list, SHAPE):
    '''
    Translate labeled pixels to the mask in the image
    '''
    tmp_flat = np.zeros(SHAPE[0]*SHAPE[1])
    if len(rle_list) == 1:
        mask = np.reshape(tmp_flat, SHAPE).T
    else:
        strt = rle_list[::2]
        length = rle_list[1::2]
        for i,v in zip(strt,length):
            tmp_flat[(int(i)-1):(int(i)-1)+int(v)] = 255
        mask = np.reshape(tmp_flat, SHAPE).T
    return mask

def DataPipeline(isship, nanship, batchsize, cap, train_img_dir, train_df):
    k = 0
    nanship_names, isship_names = nanship[:cap], isship[:cap]
    while True:
        if k+batchsize//2 >= cap:
            k = 0
        batch_nanship_names = nanship_names[k:k+batchsize//2]
        batch_isship_names = isship_names[k:k+batchsize//2]
        batch_img, batch_mask = [], []
        
        for name in batch_nanship_names:
            tmp_img = imread(train_img_dir + name)
            batch_img.append(tmp_img)
            mask_list = train_df['EncodedPixels'][train_df['ImageId'] == name].tolist()
            one_mask = np.zeros((768, 768, 1))
            for item in mask_list:
                rle_list = str(item).split()
                tmp_mask = rle_to_mask(rle_list, (768, 768))
                one_mask[:,:,0] += tmp_mask
            batch_mask.append(one_mask)
            
        for name in batch_isship_names:
            tmp_img = imread(train_img_dir + name)
            batch_img.append(tmp_img)
            mask_list = train_df['EncodedPixels'][train_df['ImageId'] == name].tolist()
            one_mask = np.zeros((768, 768, 1))
            for item in mask_list:
                rle_list = str(item).split()
                tmp_mask = rle_to_mask(rle_list, (768, 768))
                one_mask[:,:,0] += tmp_mask
            batch_mask.append(one_mask)
            
        img = np.stack(batch_img, axis=0)
        mask = np.stack(batch_mask, axis=0)
        img = img / 255.0
        mask = mask / 255.0
        k += batchsize//2
        yield img, mask