import pandas as pd
import numpy as np
from functools import reduce

data1 = pd.read_csv('./submissions/test_submission_xgb_stacker_LB_1497.csv')
data2 = pd.read_csv('./submissions/test_submission_xgb_stacker_LB_1522.csv')
data3 = pd.read_csv('./submissions/test_submission_xgb_stacker_LB_1635.csv')
data4 = pd.read_csv('./submissions/test_submission_xgb_stacker_5.csv')
#data5 = pd.read_csv('./submissions/test_submission_xgb_stacker_LB_1773.csv')
dfs = [data1, data2, data3, data4]#, data5]

df_merged = reduce( lambda  left,right: pd.merge(left,right,on=['id'],
                                            how='outer'), dfs )
idx = df_merged['id']
df_merged.drop( ['id'], inplace = True, axis = 1 )

weights = [4, 3, 2, 1]
df_merged['weighted_avg'] = np.average(df_merged, axis = 1, weights = weights)
print(df_merged.head())

avg_df = pd.DataFrame(data = {'id': idx, 'is_iceberg': df_merged['weighted_avg']})
avg_df.to_csv( './submissions/weighted_average.csv', index = False )