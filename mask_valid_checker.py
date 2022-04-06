#%%
import json
import sys
import os
import matplotlib.pyplot as plt
from digital_ink_library.sketch import Sketch
from digital_ink_library.serialization.json import JsonDataSerialization
import numpy as np
# fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
# ax.plot([0,1,2], [10,20,3])
# fig.savefig('/home/yash/Desktop/Master_Thesis/to4.png')   # save the figure to file
# plt.close(fig)
folderpath = "/home/yash/Desktop/Master_Thesis/Thesis_data-set/UPD_Bern_DFKI_additional_data_Dropbox_09.08.2021/DFKI"
filename = "33-CF-None-1605509750025-0"
#images = os.listdir(folderpath)
#images = [file for file in images if file.split('.')[1] == 'json' ]
#print(type(images))
max_x = 0
max_y = 0
mask_dict_x = {'1':[], '2':[], '3':[], '4':[], '5':[], '6':[], '7':[], '8':[], '9':[], '10':[], '11':[], '12':[], '13':[], '14':[], '15':[], '16':[], '17':[], '18':[]}
mask_dict_y = {'1':[], '2':[], '3':[], '4':[], '5':[], '6':[], '7':[], '8':[], '9':[], '10':[], '11':[], '12':[], '13':[], '14':[], '15':[], '16':[], '17':[], '18':[]}

#for img in images:
    #print(type(img))
    #filename = img.split('.')[0]
    #print(filename)


    #print(type(data['strokes'][40]))
x_cord = []
y_cord = []
#for keys in data['strokes'][40]:
#   print(keys)


# fig.savefig('blah.png')
# plt.close(fig)
#ax.set_xlim([0,80])
#ax.set_ylim([0,107])
#print(data['meta']['total_score'])
#for img in images:
    #print(type(img))
#filename = img.split('.')[0]
#print(filename)
with open(folderpath+'/'+filename+'.json') as f:
    data = json.load(f)

#os.mkdir(folderpath+'/'+filename+'_masks')

for keys in mask_dict_x:

    
         


    for i,elements in enumerate(data['lines']):
        fig, ax = plt.subplots( nrows=1, ncols=1 ,figsize=(5,5))
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
        for nested_elements in elements['points']:
            #label_check = list(map(int,nested_elements['labels']))
            for label in nested_elements['labels']:
                
                if label == '0':
                    continue
                if label == keys:
                    mask_dict_x[label].append(nested_elements['x'])
                    mask_dict_y[label].append(nested_elements['y'])

                if len(mask_dict_x[keys]) >= 2 and i == 23 and keys == '2':
                    print((mask_dict_x[keys][-2] - mask_dict_x[keys][-1])**2 + (mask_dict_y[keys][-2] - mask_dict_y[keys][-1])**2)
        
        if i == 23 and keys == '2':
            print(mask_dict_x[keys])
            print(mask_dict_y[keys])
                # if mask_dict_x[label] and mask_dict_y[label] and ((nested_elements['x'] - mask_dict_x[label][-1])**2 + (nested_elements['y'] - mask_dict_y[label][-1])**2) > 0.000001:
                #     mask_dict_x[label].pop()
                #     mask_dict_y[label].pop()

        ax.plot(mask_dict_x[keys],mask_dict_y[keys],linewidth=1.5)
        ax.set_aspect('equal')
        ax=plt.gca()                            # get the axis
        ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
        ax.xaxis.tick_top()                     # and move the X-Axis
        ax.yaxis.tick_left()                    # remove right y-Ticks

        ax.axis('off')
        #plt.draw()

        fig.savefig(folderpath+'/'+filename+'_masks/'+filename+'_mask_'+keys+'_stroke_'+str(i)+'.png', bbox_inches = 'tight',pad_inches = 0,dpi=100)
        plt.close(fig)
        # if mask_dict_x[keys] and mask_dict_y[keys]:
        #     #print('boooobies')
        #     plt.annotate('first_point',(mask_dict_x[keys][0],mask_dict_y[keys][0]))
        #     plt.annotate('last_point',(mask_dict_x[keys][-1],mask_dict_y[keys][-1]))
        
        for keys_ in mask_dict_x:
            mask_dict_x[keys_] = []
        for keys_ in mask_dict_y:
            mask_dict_y[keys_] = []
        #label_check = list(map(int,elements['meta']['labels'][0]))
        #print(label_check)
        #if not elements['meta']['labels'][0]:
        #    continue
        #x_cord.extend(elements['x'])
        #y_cord.extend(elements['y'])
        
    # a = max(x_cord)
    # b = max(y_cord)
    # if a > max_x:
    #     max_x = a
    # if b > max_y:
    #     max_y = b
    # print(max_x,max_y)
    #scores = np.append(scores,data['meta']['total_score'])
    
    
    # import numpy as np
    # x_cord = np.floor(np.array(x_cord*1000)).astype(np.int32)
    # y_cord = np.floor(np.array(y_cord*1000)).astype(np.int32)
    # blan_im = np.ones((np.max(y_cord) +1, np.max(x_cord)+1))
    # blan_im[y_cord,x_cord] = 255
    #import cv2
    #plt.imshow(blan_im)

    #plt.show()


    #plt.savefig("/home/yash/Desktop/Master_Thesis/filename.png", bbox_inches = 'tight',pad_inches = 0)
    #plt.show(blan_im)
    #print(data['strokes'])
    #plt.scatter(x_cord,y_cord)
    #ax=plt.gca()                            # get the axis
    #ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
    #ax.xaxis.tick_top()                     # and move the X-Axis
    #ax.yaxis.tick_left()                    # remove right y-Ticks
    #plt.show()
#print(scores)

# unique, counts = np.unique(scores, return_counts=True)
# score_count = dict(zip(unique, counts))
# print(score_count)
# score_x = []
# score_y = []
# for keys,values in score_count.items():
#     score_x.append(keys)
#     score_y.append(values)

# # %%
# print(score_y)
# score_x = list(map(str,score_x))
# print(score_x)
# #fig1 = plt.figure()
# #ax1 = fig1.add_axes([0,0,1,1])
# #ax1.axis('on')
# plt.bar(score_x,score_y)
# #print(os.getcwd())
# plt.savefig('/home/yash/Desktop/Master_Thesis/CDT_images/CDT_stats.png')

# %%
