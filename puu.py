import numpy as np
import pandas as pd
labels = ['a','b','c']
my_data=[10,20,30]
arr = np.array(my_data)
d = {'a':10,'b':20,'c':30}
print(labels)
print(my_data)
print(d)
print(arr)
print(pd.Series(data=my_data))
print(pd.Series(data=my_data,index=labels))

### what type of vlues can hold
"""
numerical
text
functions
dict

"""

#INDEXING AND SLICING 
ser1 = pd.Series([1,2,3,4],['a','b','c','d'])
print(ser1)
ser2= pd.Series([1,2,5,4],['e','f','g','h'])
print(ser2)
print(ser1['c'])#  index
print("oo",ser1+ser2)

#creating and accessing dataframe

np.random.seed(101)
row_labels=['a','b','c','d','e']
col_labels=['f','g','h']
df = pd.DataFrame(data=,index=row_labels,columns=col_labels)
print(df)
# by using + we can add two dfs or cols and by using df.drop we can deleye a col . "by using inplace = True and axis =1" we have to drop.
# for selecting rows "loc "  is used and for indexing rows "iloc " is used .
  # SUBSETTING DATAFRAME
#booldf = df>0  in the +output if num is greater than 0 it prints true else false
# for dropping rows axis =0
# For filling null values df.filna(value to be filled )
# groupby  df.groupby 
#describe()  COUNT,MEAN,STD,MIN,25%.......
# CONCATENATE OF DFS pd.concat([df1,df2,df3],axis=0) rows,    [df1,df2,df3],axis=1 cols , 
# MERGE pd.merge(x,y,how = 'inner',on='keys)
# head(), unique(),nunique(),value_count()a     ---------------------------------------------------------------aaaa








#### LINEAR REGRESSION IN MULTI VARIABLE
# y = m1x1 + m2x2 + m3x3 + b   for example there are 2 variables where m1 ,m2.m3 are the coefficients and b is the intercept 
# removing nan values and fil it with median value or any mean 
