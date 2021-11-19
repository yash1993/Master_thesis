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
folderpath = "/home/yash/Desktop/Master_Thesis/Thesis_data-set/ROCF_Charite"

with open(folderpath+'/Charite ROCF (7).json') as f:
        data = json.load(f)

x_cord = []
y_cord = []
count = 0
#for keys in data['strokes'][40]:
#   print(keys)
fig, ax = plt.subplots( nrows=1, ncols=1 ) 
# ax.plot([0,1,2], [10,20,3])
# fig.savefig('blah.png')
# plt.close(fig)
#ax.set_xlim([0,121])
#ax.set_ylim([0,121])
print(len(data['strokes']))
for elements in data['strokes']:

    
    label_check = list(map(int,elements['meta']['labels'][0]))
    print(label_check)
    if sum(label_check) == 0:
        continue
    x_cord.extend(elements['x'])
    y_cord.extend(elements['y'])
    count+=1
    ax.plot(elements['x'],elements['y'],'k',linewidth=0.7)
# a = max(x_cord)
# b = max(y_cord)
# if a > max_x:
#     max_x = a
# if b > max_y:
#     max_y = b
# print(max_x,max_y)
#scores = np.append(scores,data['meta']['total_score'])
print(count)
ax.set_aspect('equal')
ax=plt.gca()                            # get the axis
ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
ax.xaxis.tick_top()                     # and move the X-Axis
ax.yaxis.tick_left()                    # remove right y-Ticks

ax.axis('off')
fig.savefig('/home/yash/Desktop/Charite ROCF (7).png', bbox_inches = 'tight',pad_inches = 0,dpi=300)
plt.close(fig)