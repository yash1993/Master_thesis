from pandas_ods_reader import read_ods
from collections import Counter
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
path = "/home/yash/Desktop/Master_Thesis/Thesis_data-set/UPD_Bern_scores.ods"

# load a sheet based on its index (1 based)
sheet_idx = 1
df = read_ods(path, sheet_idx)

# load a sheet based on its name
sheet_name = "Sheet1"
df = read_ods(path, sheet_name)
print(type(df['Total Score']))
score_list = list(df['Total Score'])

score_list.remove(score_list[0])
#print(score_list)
#print(score_list)
unique, counts = np.unique(score_list, return_counts=True)
score_count = dict(zip(unique, counts))
score_x = []
score_y = []
for keys,values in score_count.items():
    score_x.append(keys)
    score_y.append(values)
#score_x = Counter(score_list).keys()
#score_y = Counter(score_list).values()
#score_x = list(score_x)
#print(type(score_x))
print(score_x)
score_x.sort()
print(score_x)
score_x = list(map(str,score_x))
sex_lund = dict(zip(score_x,score_y))
print(sex_lund)
fig, ax = plt.subplots(figsize=(40,20))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
rec1 = ax.bar(score_x,score_y,width = 0.5)
ax.set_ylabel('Number of figures',fontsize=25)
ax.set_xlabel('ROCF Total score',fontsize=25)
ax.set_title('UPD Bern stats',fontsize=25)
ax.bar_label(rec1,fontsize=20)
#print(os.getcwd())
plt.savefig('/home/yash/Desktop/Master_Thesis/Thesis_data-set/2021-03-24_ROCF_UPD_Uni_Bern/ROCF UPD Uni Bern/UPD_Bern_stats.png',dpi=300,bbox_inches='tight')

# load a file that does not contain a header row
# if no columns are provided, they will be numbered
#df = read_ods(path, 1, headers=False)

# load a file and provide custom column names
# if headers is True (the default), the header row will be overwritten
#df = read_ods(path, 1, columns=["A", "B", "C"])