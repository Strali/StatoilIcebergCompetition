import pandas as pd
from functools import reduce
'''
test1 = pd.read_csv( './submissions/test_submission_0.csv' )
test2 = pd.read_csv( './submissions/test_submission_1.csv' )
test3 = pd.read_csv( './submissions/test_submission_2.csv' )
test4 = pd.read_csv( './submissions/test_submission_3.csv' )
'''
test5 = pd.read_csv( './submissions/test_submission_4.csv' )
'''
test6 = pd.read_csv( './submissions/test_submission_5.csv' )
test7 = pd.read_csv( './submissions/test_submission_6.csv' )
test8 = pd.read_csv( './submissions/test_submission_7.csv' )
test9 = pd.read_csv( './submissions/test_submission_8.csv' )
'''
test10 = pd.read_csv( './submissions/test_submission_9.csv' )

test11 = pd.read_csv( './submissions/test_submission_twostage_resnet_avg_pool.csv' )
test12 = pd.read_csv( './submissions/test_submission_twostage_resnet_prelu_avg_pool.csv' )
test13 = pd.read_csv( './submissions/test_submission_twostage_resnet_prelu_avg_pool_higher_lr.csv' )
#test14 = pd.read_csv( './submissions/test_submission_stack.csv' )

#dfs = [ test1, test2, test3, test4, test5, test6, test7, test8, test9, test10, test11, test12, test13 ]
dfs = [test5, test10, test11, test12, test13]

df_merged = reduce( lambda  left,right: pd.merge(left,right,on=['id'],
                                            how='outer'), dfs )

idx = df_merged['id']
print( df_merged.head() )
df_merged.drop( ['id'], inplace = True, axis = 1 )

#df_merged['avg'] = df_merged[['is_iceberg_x', 'is_iceberg_y', 'is_iceberg_x', 'is_iceberg_y', 'is_iceberg']].mean(axis = 1)
df_merged['avg'] = df_merged.mean(axis = 1)
print( df_merged.head(10) )

avg_df = pd.DataFrame( data = {'id': idx, 'is_iceberg': df_merged['avg']} )
#avg_df.rename( columns = {'id': 'id', 'avg': 'is_iceberg'}, inplace = True )
print(avg_df.head())
avg_df.to_csv( './submissions/avg_test_scores_5_with_stack.csv', index = False )