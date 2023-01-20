'''Script to generate ground truth masks for ROCF-test images'''
import json
import sys
import os
import matplotlib.pyplot as plt
from digital_ink_library.sketch import Sketch
from digital_ink_library.serialization.json import JsonDataSerialization
import numpy as np

# Folder path for input json files
folderpath = "/home/yash/Desktop/Master_Thesis/Thesis_data-set/ROCF_Charite"
images = os.listdir(folderpath)
images = [file for file in images if file.endswith('json') ]
# images = ['Charite ROCF (121)']
max_x = 0
max_y = 0
mask_dict_x = {'1':[], '2':[], '3':[], '4':[], '5':[], '6':[], '7':[], '8':[], '9':[], '10':[], '11':[], '12':[], '13':[], '14':[], '15':[], '16':[], '17':[], '18':[]}
mask_dict_y = {'1':[], '2':[], '3':[], '4':[], '5':[], '6':[], '7':[], '8':[], '9':[], '10':[], '11':[], '12':[], '13':[], '14':[], '15':[], '16':[], '17':[], '18':[]}


for img in images:
    #print(type(img))
    filename = img.split('.')[0]
    #print(filename)
    with open(folderpath+'/'+filename+'.json') as f:
        data = json.load(f)
    if not os.path.exists(folderpath+'/'+filename+'_masks'):
        os.mkdir(folderpath+'/'+filename+'_masks')

    # Iterate through all the 18 labels and find coordinates for them
    for keys in mask_dict_x:
        
        # Dummy plot loop
        fig, ax = plt.subplots( nrows=1, ncols=1 ,figsize=(5,5))
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
        
            ax.plot(new_elements_x,new_elements_y,'w',linewidth=1.5) 

        # Main loop to find coordinates with corresponding label
        for elements in data['strokes']:
            
            for i, label_list in enumerate(elements['meta']['labels']):
                
                for label in label_list:
                    if label == '0':
                        continue
                    if label == keys:
                        mask_dict_x[label].append(elements['x'][i])
                        mask_dict_y[label].append(elements['y'][i])

                    # Condition to check for distance. This is checked to avoid unwanted lines in the output masks
                    # The value of 50 for distance limit is determined empirically
                    if len(mask_dict_x[keys]) >= 2 and (mask_dict_x[keys][-2] - mask_dict_x[keys][-1])**2 + (mask_dict_y[keys][-2] - mask_dict_y[keys][-1])**2 > 50:
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
        
        fig.savefig(folderpath+'/'+filename+'_masks/'+filename+'_mask_'+keys+'.png', bbox_inches = 'tight',pad_inches = 0,dpi=100)
        plt.close(fig)
    
    

# %%
