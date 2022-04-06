#%%
import json
import sys
import os
import matplotlib.pyplot as plt
from digital_ink_library.sketch import Sketch
from digital_ink_library.serialization.json import JsonDataSerialization
import numpy as np

folderpath = "/home/yash/Desktop/Master_Thesis/Thesis_data-set/UPD_Bern_DFKI_additional_data_Dropbox_09.08.2021/DFKI"
images = os.listdir(folderpath)
images = [file for file in images if file.split('.')[1] == 'json' ]
#print(type(images))
max_x = 0
max_y = 0
scores = []
png_filename = set()
for img in images:
    #print(type(img))
    filename = img.split('.')[0]
    print(filename)
    with open(folderpath+'/'+filename+'.json') as f:
        data = json.load(f)

    
    x_cord = []
    y_cord = []
    
    fig, ax = plt.subplots( nrows=1, ncols=1, figsize=(5,5)) 
    
    for elements in data['lines']:
        x_cord = []
        y_cord = []

        for nested_elements in elements['points']:
            label_check = list(map(int,nested_elements['labels']))
            # #print(label_check)
            if sum(label_check) == 0:
                continue
            x_cord.append(nested_elements['x'])
            y_cord.append(nested_elements['y'])
        
        ax.plot(x_cord,y_cord,'k',linewidth=1.5)    

        
        
        
        
    
    
    ax.set_aspect('equal')
    ax=plt.gca()                            # get the axis
    ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
    ax.xaxis.tick_top()                     # and move the X-Axis
    ax.yaxis.tick_left()                    # remove right y-Ticks

    ax.axis('off')
    fig.savefig('/home/yash/Desktop/Master_Thesis/Thesis_data-set/UPD_Bern_DFKI_additional_data_Dropbox_09.08.2021/DFKI/'+filename+'.png', bbox_inches = 'tight',pad_inches = 0,dpi=100)
    plt.close(fig)
    


# %%
