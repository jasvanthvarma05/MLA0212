#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'plusMinus' function below.
#
# The function accepts INTEGER_ARRAY arr as parameter.
#

def plusMinus(arr):
    # Write your code here
    for num in arr:
        count = 0
        if num>0:
            count+=1
            avg = count/len(arr)
            print(avg)
        elif(num<0):
            count = 0
        if num>0:
            count+=1
            avg1= count/len(arr)
            print(avg1)
            
if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    plusMinus(arr)
