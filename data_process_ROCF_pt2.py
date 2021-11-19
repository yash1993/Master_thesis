#%%
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
# fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
# ax.plot([0,1,2], [10,20,3])
# fig.savefig('/home/yash/Desktop/Master_Thesis/to4.png')   # save the figure to file
# plt.close(fig)
folderpath = "/home/yash/Desktop/Master_Thesis/Thesis_data-set/annotated"
images = os.listdir(folderpath)
images = [file for file in images if file.split('.')[1] == 'json' ]
#print(type(images))
max_x = 0
max_y = 0
min_x = 0
min_y = 0
scores = []
for img in images:
    #print(type(img))
    filename = img.split('.')[0]
    print(filename)
    with open(folderpath+'/'+filename+'.json') as f:
        data = json.load(f)

    #print(type(data['strokes'][40]))
    x_cord = []
    y_cord = []
    #for keys in data['strokes'][40]:
    #   print(keys)
    fig, ax = plt.subplots( nrows=1, ncols=1, figsize=(5,5))
    
    # ax.plot([0,1,2], [10,20,3])
    # fig.savefig('blah.png')
    # plt.close(fig)
    #ax.set_xlim([-109,116])
    #ax.set_ylim([-109,116])
    #print(data['meta']['total_score'])
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
        # x_cord.extend(elements['x'])
        # y_cord.extend(elements['y'])
        ax.plot(new_elements_x,new_elements_y,'k',linewidth=1.5)
    # print(len(x_cord))
    # a = max(x_cord)
    # b = max(y_cord)
    # c = min(x_cord)
    # d = min(y_cord)
    # if a > max_x:
    #     max_x = a
    # if b > max_y:
    #     max_y = b
    # if c < min_x:
    #     min_x = c
    # if d < min_y:
    #     min_y = d
    # print(min_x,min_y)
    #scores = np.append(scores,data['meta']['total_score'])
    ax.set_aspect('equal')
    ax=plt.gca()                            # get the axis
    ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
    ax.xaxis.tick_top()                     # and move the X-Axis
    ax.yaxis.tick_left()                    # remove right y-Ticks

    
    ax.axis('off')
    fig.savefig('/home/yash/Desktop/Master_Thesis/Thesis_data-set/annotated/'+filename+'.png', bbox_inches = 'tight',pad_inches = 0,dpi=100)
    plt.close(fig)
    

#padding algorithm

# img = cv2.imread('/home/yash/ass.png')
# h, w, c = img.shape
# delta_w = 1200 - w
# delta_h = 1200 - h
# top, bottom = delta_h//2, delta_h-(delta_h//2)
# left, right = delta_w//2, delta_w-(delta_w//2)
# color = [255, 255, 255]
# new_im = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
#     value=color)
# cv2.imwrite('resized.png',new_im)

#padding algorithm end
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
