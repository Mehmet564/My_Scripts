# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 12:54:14 2020

@author: mehmet
"""

import numpy 
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from os import listdir
from os.path import isfile, join

dpi = 30
data=np.genfromtxt(r'C:\Users\mehmet\Desktop\dataset\dataset_04_1.txt', delimiter='  ', dtype=float)
unique = np.unique(data[:,0], axis = 0 ) 

mypath=r"C:\Users\mehmet\Desktop\images"

onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]

images = numpy.empty(len(onlyfiles), dtype=object)

b = np.arange(start=20, stop=21, step=1, dtype=int)
# data = data.reshape(1,-1)
for k in range(1,len(b)+1):
        
        for n in range(1,len(onlyfiles)+1):
             
                if b[k-1,] ==  unique[n-1,]:
            
                    dpi = 30
                        
                    images[n-1] = plt.imread( join(mypath,onlyfiles[n-1]))
                        
                    num_rows, num_cols,RGB = images[n-1].shape
                        
                    # What size does the figure need to be in inches to fit the image?
                    figsize = num_rows / float(dpi), num_cols / float(dpi)
                        
                    # Create a figure of the right size with one axes that takes up the full figure
                    fig = plt.figure(figsize=figsize)
                    ax = fig.add_axes([0, 0, 1, 1])
                        
                    ax.axis('off')
                        
                    ax.set(xlim=[-0.5, num_rows - 0.5], ylim=[num_cols - 0.5, -0.5], aspect=1)
                    for i in range(1,len(data)+1):
                            
                        if data[i-1,0]== b[k-1,]:
    
    
                            X =  data[i-1,  [1,3,5]]
                            y =  data[i-1,  [2,4,6]]
                            X1 = data[i-1,  [1,7,9]]
                            y1 = data[i-1, [2,8,10]]
                            X2 = data[i-1,[1,11,13]]
                            y2 = data[i-1,[2,12,14]]
                            X3 = data[i-1,[1,15,17]]
                            y3 = data[i-1,[2,16,18]]
                                
                            X = X.reshape(-1,1)
                            y =y.reshape(-1,1)
                            X1 =X1.reshape(-1,1)
                            y1 =y1.reshape(-1,1)
                            X2 = X2.reshape(-1,1)
                            y2 =y2.reshape(-1,1)
                            X3 = X3.reshape(-1,1)
                            y3 =y3.reshape(-1,1)
                                
                            
                                
                            # for i in range(len(X)):
                            def Polynomial_Regression():
                                # plt.scatter(X, y, color='Red')
                                poly_reg = PolynomialFeatures(degree=7)
                                X_poly = poly_reg.fit_transform(X)
                                pol_reg = LinearRegression()
                                pol_reg.fit(X_poly, y)
                                # oly_Predict = pol_reg.predict(X_poly)
                                plt.plot(X,pol_reg.predict(X_poly) , Linewidth = 3,color = 'Red')
                            Polynomial_Regression()         
                            def Polynomial_Regression1():
                                # plt.scatter(X1, y1, color='Blue')
                                poly_reg = PolynomialFeatures(degree=7)
                                X_poly = poly_reg.fit_transform(X1)
                                pol_reg = LinearRegression()
                                pol_reg.fit(X_poly, y1)
                                # Poly_Predict = pol_reg.predict(X_poly)
                                plt.plot(X1,pol_reg.predict(X_poly) , Linewidth = 3,color = 'Blue')
                            Polynomial_Regression1()        
                            def Polynomial_Regression2():
                                # plt.scatter(X2, y2, color='Green')
                                poly_reg = PolynomialFeatures(degree=7)
                                X_poly = poly_reg.fit_transform(X2)
                                pol_reg = LinearRegression()
                                pol_reg.fit(X_poly, y2)
                                # Poly_Predict = pol_reg.predict(X_poly)
                                plt.plot(X2,pol_reg.predict(X_poly) , Linewidth = 3,color = 'Green')
                            Polynomial_Regression2()    
                            def Polynomial_Regression3():
                                # plt.scatter(X3, y3, color='Black')
                                poly_reg = PolynomialFeatures(degree=7)
                                X_poly = poly_reg.fit_transform(X3)
                                pol_reg = LinearRegression()
                                pol_reg.fit(X_poly, y3)
                                # Poly_Predict = pol_reg.predict(X_poly)
                                plt.plot(X3,pol_reg.predict(X_poly) , Linewidth = 3,color = 'Black')
                            
                            
                            Polynomial_Regression3()
                            plt.imshow(images[n-1],Polynomial_Regression(),Polynomial_Regression1(),Polynomial_Regression2(), Polynomial_Regression3())
                                    
                            if int(data[i-1,0])<10:
                                fig.savefig("im00000{}".format(b[k-1])+'.jpg', dpi=dpi, transparent=True) 
                                    
                            elif 10<=int(data[i-1,0])<100:
                                fig.savefig("im0000{}".format(b[k-1])+'.jpg', dpi=dpi, transparent=True)  
                                    
                            elif 100<=int(data[i-1,0])<1000:
                                fig.savefig("im000{}".format(b[k-1])+'.jpg', dpi=dpi, transparent=True)    
                                    
                            elif 1000<=int(data[i-1,0])<10000:
                                fig.savefig("im00{}".format(b[k-1])+'.jpg', dpi=dpi, transparent=True)
                                
                            else:
                                fig.savefig("im0{}".format(b[k-1])+'.jpg', dpi=dpi, transparent=True)
                
