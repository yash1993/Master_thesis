import sys
import os
import cv2

folderpath = "/home/yash/Desktop/Master_Thesis/Thesis_data-set/ROCF_Charite"
images = os.listdir(folderpath)
images = [file for file in images if file.endswith('png') ]

#print(images)

for i in images:
    img = cv2.imread(folderpath+'/'+i)
    h, w, c = img.shape
    delta_w = 512 - w
    delta_h = 512 - h
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [255, 255, 255]
    new_im = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    cv2.imwrite(folderpath+'/'+i,new_im)

mask_folders = os.listdir(folderpath)
mask_folders = [file for file in mask_folders if file.endswith('masks') ]
# print(mask_folders)

for i in mask_folders:
    mask_imgs = os.listdir(folderpath+'/'+i)

    for j in mask_imgs:
        mask = cv2.imread(folderpath+'/'+i+'/'+j)
        h, w, c = mask.shape
        delta_w = 512 - w
        delta_h = 512 - h
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        color = [255, 255, 255]
        new_im = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)
        cv2.imwrite(folderpath+'/'+i+'/'+j,new_im)


