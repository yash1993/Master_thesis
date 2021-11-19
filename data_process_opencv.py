import json
import sys
import os
import matplotlib.pyplot as plt
from digital_ink_library.sketch import Sketch
from digital_ink_library.serialization.json import JsonDataSerialization
import numpy as np
import cv2
# fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
# ax.plot([0,1,2], [10,20,3])
# fig.savefig('/home/yash/Desktop/Master_Thesis/to4.png')   # save the figure to file
# plt.close(fig)

with open('/home/yash/Desktop/Master_Thesis/Thesis_data-set/Intera-KT_Website_Export_ALL_2020-05-26/cdt/B01_20191023_16_12_CDT.json') as f:
    data = json.load(f)

#print(type(data['strokes'][40]))
x_cord = []
y_cord = []
fig, ax = plt.subplots( nrows=1, ncols=1 )
#for keys in data['strokes'][40]:
 #   print(keys)
#fig, ax = plt.subplots( nrows=1, ncols=1 ) 
# ax.plot([0,1,2], [10,20,3])
# fig.savefig('blah.png')
# plt.close(fig)
#ax.set_xlim([0,86])
#ax.set_ylim([0,96])
for elements in data['strokes']:

    print(type(elements['meta']['labels'][0]))
    label_check = list(map(int,elements['meta']['labels'][0]))
    print(label_check)
    if sum(label_check) == 0:
        continue
    x_cord.extend(elements['x'])
    y_cord.extend(elements['y'])


x_cord = np.floor(np.array(x_cord*1000)).astype(np.int32)
y_cord = np.floor(np.array(y_cord*1000)).astype(np.int32)
blan_im = np.ones((np.max(y_cord) +1, np.max(x_cord)+1))
blan_im[y_cord,x_cord] = 255


plt.axis('off')
plt.imshow(blan_im)
fig.savefig('/home/yash/Desktop/Master_Thesis/blah2.png', bbox_inches = 'tight',pad_inches = 0,dpi=300)
plt.show()