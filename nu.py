import numpy as np
a = np.array([1,2,3,4])
print(a)
twod=np.array([[1,2,3,4],
          [3,4,5,6]])
print(twod)
print(twod.ndim) # count no of dimensions
print(twod.shape)# no of rows and cols
print(twod.dtype) # data type
print(twod.itemsize) # size of array
print(twod.nbytes) # size of bytes

# how to access in  arrays 
b = np.array([[1,2,3,4,5,6,7],[8,7,6,5,4,3,2]])
print(b)
print(b[1,5]) #(r,c)
print(b[1,:]) #full row 
print(b[:,4]) # seperate col indexing from 0 
# we can do in 3d alsoo([[[]]]) we can check by using .ndim
s = np.zeros((2,3))
print(s)
o = np.ones((2,3,2)) #(rows,cols,no of matrices )
print(o)
d = np.full((2,3),100) # whole matrix with same number 
print(d)
f = np.random.rand(4,2) # random values
print(f) 
id = np.identity(5) # identiti matrix with 5*5
print(id)

# copying  
a = np.array([1,2,3,4])
b = a.copy() # if we change in b also it wont change 
print(b)
print(a+2) # we can doo math operations power of nums
