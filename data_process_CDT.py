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
folderpath = "/home/yash/Desktop/Master_Thesis/Thesis_data-set/Intera-KT_Website_Export_ALL_2020-05-26/cdt"
images = os.listdir(folderpath)
images = [file for file in images if file.split('.')[1] == 'json' ]
#print(type(images))
max_x = 0
max_y = 0
scores = []
for img in images:
    #print(type(img))
    filename = img.split('.')[0]
    #print(filename)
    with open(folderpath+'/'+filename+'.json') as f:
        data = json.load(f)

    #print(type(data['strokes'][40]))
    x_cord = []
    y_cord = []
    #for keys in data['strokes'][40]:
    #   print(keys)
    fig, ax = plt.subplots( nrows=1, ncols=1 ) 
    # ax.plot([0,1,2], [10,20,3])
    # fig.savefig('blah.png')
    # plt.close(fig)
    ax.set_xlim([0,80])
    ax.set_ylim([0,107])
    #print(data['meta']['total_score'])
    for elements in data['strokes']:

        #print(type(elements['meta']['labels'][0]))
        label_check = list(map(int,elements['meta']['labels'][0]))
        #print(label_check)
        if sum(label_check) == 0:
            continue
        x_cord.extend(elements['x'])
        y_cord.extend(elements['y'])
        ax.plot(elements['x'],elements['y'],'k',linewidth=0.7)
    # a = max(x_cord)
    # b = max(y_cord)
    # if a > max_x:
    #     max_x = a
    # if b > max_y:
    #     max_y = b
    # print(max_x,max_y)
    scores = np.append(scores,data['meta']['total_score'])
    ax.set_aspect('equal')
    ax=plt.gca()                            # get the axis
    ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
    ax.xaxis.tick_top()                     # and move the X-Axis
    ax.yaxis.tick_left()                    # remove right y-Ticks

    ax.axis('off')
    fig.savefig('/home/yash/Desktop/Master_Thesis/CDT_images/'+filename+'.png', bbox_inches = 'tight',pad_inches = 0,dpi=300)
    plt.close(fig)
    
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

unique, counts = np.unique(scores, return_counts=True)
score_count = dict(zip(unique, counts))
print(score_count)
score_x = []
score_y = []
for keys,values in score_count.items():
    score_x.append(keys)
    score_y.append(values)

# %%
print(score_y)
score_x = list(map(str,score_x))
print(score_x)
fig, ax = plt.subplots()
rec1 = ax.bar(score_x,score_y,width = 0.3)
ax.set_ylabel('Number of figures')
ax.set_xlabel('CDT Total Mendez score')
ax.set_title('CDT stats')
ax.bar_label(rec1)
#fig1 = plt.figure()
#ax1 = fig1.add_axes([0,0,1,1])
#ax1.axis('on')
#plt.bar(score_x,score_y)
#print(os.getcwd())
plt.savefig('/home/yash/Desktop/Master_Thesis/CDT_images/CDT_stats.png')

# %%
