#%%
import json
import sys
import os
import matplotlib.pyplot as plt
from digital_ink_library.sketch import Sketch
from digital_ink_library.serialization.json import JsonDataSerialization
import numpy as np

folderpath = "/home/yash/Desktop/Master_Thesis/Thesis_data-set/data"

# images = os.listdir(folderpath)
# images = [file for file in images if file.endswith('json') ]
images = ['labeled-1781-CF-R','labeled-2143-CF-R','labeled-2241-CF-R','labeled-2242-CF-E','labeled-2269-CF-R']

mask_dict_x = {'1':[], '2':[], '3':[], '4':[], '5':[], '6':[], '7':[], '8':[], '9':[], '10':[], '11':[], '12':[], '13':[], '14':[], '15':[], '16':[], '17':[], '18':[]}
mask_dict_y = {'1':[], '2':[], '3':[], '4':[], '5':[], '6':[], '7':[], '8':[], '9':[], '10':[], '11':[], '12':[], '13':[], '14':[], '15':[], '16':[], '17':[], '18':[]}

for img in images:
    print(img)
    filename = img.split('.')[0]
    

    x_cord = []
    y_cord = []

    with open(folderpath+'/'+filename+'.json') as f:
        data = json.load(f)

    if not os.path.exists(folderpath+'/'+filename+'_masks'):
        os.mkdir(folderpath+'/'+filename+'_masks')
    
    #Loop for creating white plots to maintain original image resolution among all mask images
    for keys in mask_dict_x:

        fig, ax = plt.subplots( nrows=1, ncols=1 ,figsize=(5,5))
        for elements in data['lines']:
            x_cord = []
            y_cord = []

            for nested_elements in elements['points']:
                label_check = list(map(int,nested_elements['labels']))
                #print(label_check)
                if sum(label_check) == 0:
                    continue
                x_cord.append(nested_elements['x'])
                y_cord.append(nested_elements['y'])
            
            ax.plot(x_cord,y_cord,'w',linewidth=1.5) 

        #Mask plotting loop
        for elements in data['lines']:
            
            for nested_elements in elements['points']:
                #label_check = list(map(int,nested_elements['labels']))
                for label in nested_elements['labels']:
                    
                    if label == '0':
                        continue
                    if label == keys:
                        mask_dict_x[label].append(nested_elements['x'])
                        mask_dict_y[label].append(nested_elements['y'])
                    # Check if distance between consecutive coordinates is too large to avoid unwanted plots
                    if len(mask_dict_x[keys]) >= 2 and (mask_dict_x[keys][-2] - mask_dict_x[keys][-1])**2 + (mask_dict_y[keys][-2] - mask_dict_y[keys][-1])**2 > 225:
                        mask_dict_x[keys].pop()
                        mask_dict_y[keys].pop()
                        ax.plot(mask_dict_x[keys],mask_dict_y[keys],'k',linewidth=1.5)
                        for keys_ in mask_dict_x:
                            mask_dict_x[keys_] = []
                        for keys_ in mask_dict_y:
                            mask_dict_y[keys_] = []
            ax.plot(mask_dict_x[keys],mask_dict_y[keys],'k',linewidth=1.5)
            
            
            for keys_ in mask_dict_x:
                mask_dict_x[keys_] = []
            for keys_ in mask_dict_y:
                mask_dict_y[keys_] = []
            
            
        
        ax.set_aspect('equal')
        ax=plt.gca()                            # get the axis
        ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
        ax.xaxis.tick_top()                     # and move the X-Axis
        ax.yaxis.tick_left()                    # remove right y-Ticks

        ax.axis('off')
        #plt.draw()

        fig.savefig(folderpath+'/'+filename+'_masks/'+filename+'_mask_'+keys+'.png', bbox_inches = 'tight',pad_inches = 0,dpi=100)
        plt.close(fig)
    
    

# %%
