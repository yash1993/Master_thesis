
import pandas as pd

score_path = "/home/yash/Desktop/Master_Thesis/Thesis_data-set/20210424_ROCF-Scoring_DFKI.xlsx"

df = pd.concat(pd.read_excel(score_path, sheet_name = None,skiprows = 8), ignore_index=True)
df.rename(columns = {'Unnamed: 0':'filename'}, inplace=True)
df.set_index('filename', inplace=True)
#print(df.columns)
print(list(df.loc['Charite ROCF (69)']))
