import numpy as np
import cv2
import os

y = np.zeros((512,512,3),dtype=np.uint8)
color_list = [[  0 , 0, 255],
 [  0, 255, 0],
 [  255, 0, 0],
 [  0, 255, 255],
 [  255, 0, 255],
 [  255, 255, 0],
 [  0, 0, 128],
 [  0, 128, 0],
 [ 128, 0, 0],
 [ 0, 128, 128],
 [128, 0, 128],
 [128, 128,   0],
 [255, 128, 0],
 [252, 3, 115],
 [83, 55, 122],
 [255, 128, 128],
 [129, 112, 102],
 [244, 200, 0]]

xu = np.ones((512,512)) 
for i in range(1,19):
    x = cv2.imread('/home/yash/Desktop/Master_Thesis/Thesis_data-set/ROCF_Charite/Charite ROCF (28)_masks/Charite ROCF (28)_mask_'+str(i)+'.png')[:,:,0]
    xu = np.where(x==0,0,xu)
    y[x==0] = color_list[i-1]


#im_color = cv2.applyColorMap(y, cv2.COLORMAP_HSV)
y[xu==1] = 255
cv2.imwrite('/home/yash/Desktop/Test_ouputs/fixed_color.png', y[:,:,[2,1,0]])
    