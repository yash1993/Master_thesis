import numpy as np
import cv2
import os

y = np.zeros((512,512,3),dtype=np.uint8)
color_list = [[  0 , 60, 255],
 [  0, 120, 255],
 [  0, 180, 255],
 [  0, 240, 255],
 [  0, 255,  30],
 [  0, 255,  90],
 [  0, 255, 150],
 [  0, 255, 210],
 [ 30, 255,   0],
 [ 90, 255,   0],
 [150, 255,   0],
 [210, 255,   0],
 [255,   0,   0],
 [255,   0,  60],
 [255,  60,   0],
 [255, 120,   0],
 [255, 180,   0],
 [255, 240,   0]]

xu = np.ones((512,512)) 
for i in range(18):
    x = cv2.imread('/home/yash/Desktop/Test_ouputs/mask_img_55-CF-None-1611675518618-0_Unet_'+str(i)+'.png')[:,:,0]
    xu = np.where(x==0,0,xu)
    y[x==0] = color_list[abs(17-i)]


#im_color = cv2.applyColorMap(y, cv2.COLORMAP_HSV)
y[xu==1] = 255
cv2.imwrite('/home/yash/Desktop/Test_ouputs/fixed_color.png', y[:,:,[2,1,0]])
    