#%%
'''Script to pre-process json files from Charite containing coordinates into images'''
import json
import sys
import os
import matplotlib.pyplot as plt
from digital_ink_library.sketch import Sketch
from digital_ink_library.serialization.json import JsonDataSerialization
import numpy as np
import PIL
from PIL import Image, ImageOps
import cv2

folderpath = "/home/yash/Desktop/Master_Thesis/Thesis_data-set/ROCF_Charite"
images = os.listdir(folderpath)
images = [file for file in images if file.split('.')[1] == 'json' ]
#print(type(images))
max_x = 0
max_y = 0
min_x = 0
min_y = 0
scores = []
for img in images:
    
    filename = img.split('.')[0]
    with open(folderpath+'/'+filename+'.json') as f:
        data = json.load(f)

    fig, ax = plt.subplots( nrows=1, ncols=1, figsize=(5,5))
    
    
    for elements in data['strokes']:

        new_elements_x = []
        new_elements_y = []
        for i, coord_labels in enumerate(elements['meta']['labels']):
            label_check = list(map(int,coord_labels))
            #print(label_check)
            if sum(label_check) == 0:
                continue
            new_elements_x.append(elements['x'][i])
            new_elements_y.append(elements['y'][i])
        
        ax.plot(new_elements_x,new_elements_y,'k',linewidth=1.5)
    
    ax.set_aspect('equal')
    ax=plt.gca()                            # get the axis
    ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
    ax.xaxis.tick_top()                     # and move the X-Axis
    ax.yaxis.tick_left()                    # remove right y-Ticks

    
    ax.axis('off')
    fig.savefig('/home/yash/Desktop/Master_Thesis/Thesis_data-set/ROCF_Charite/'+filename+'.png', bbox_inches = 'tight',pad_inches = 0,dpi=100)
    plt.close(fig)


# %%
