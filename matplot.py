import matplotlib.pyplot as plt
x = [1,2,3,4]
y=[12,12,12,43]
plt.xlabel('me')
plt.ylabel('u')
plt.title('u and me')
plt.plot(x,y,color='green',linewidth=3,linestyle='dashdot') #dashdot ,dotted
plt.show()

#plt.plot(x,y,'g+') this means instead of line it visualize a +symbol with green colour(g)'rD means diamond red .
#plt.plot(x,y,rD) == plt.plot(x,y,color='red',marker='D',linestyle='')
#can specify marker size also 
#alphe means transparency can specify 0-1
# wenn can specify three lines in a single chart and specify individual lables . BY specifying individually plt.lagend() to clearly visualize the name of the lines. 
# can modify place of legend by using "loc", plt.legend(loc="upper right"),    plt.legend(loc="best")  , can change fosize of legend.
# # can use gridlines to exactlty figure out the data plt.grid() 


#plt.bar(x,y) for bar graph
import numpy as np
com=['goo','amz','fl']
rev=[1,2,3]
plt.xticks(ypos,com)#label             
ypos=np.arange(len(com))
plt.bar(com,rev)
plt.show() 
