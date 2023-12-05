# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 10:21:12 2016

@author: Abinet Habtemariam 
"""


import numpy as np
import math 
from scipy.integrate import quad
import itertools
from sympy import *
import matplotlib.pyplot as plt
import pandas as pd
#_____________________________________________________________

class GBT_func_curve_orginal:
    def __init__ (self, k,r, vertheta,R):
        self.k=k
        if k.split("|")[0]=='t' or  k.split("|")[0]=='a' or  k.split("|")[0]=='1':
            self.m= 0
        elif  int(k.split("|")[0])  % 2 == 0 :
            self.m=int(k.split("|")[0])/2
        elif  int(k.split("|")[0]) % 2 != 0 :
            self.m=(int(k.split("|")[0])-1)/2
        self.r=r
        self.R=R
        self.vertheta=vertheta
#        self.vertheta=Symbol('vertheta')

    # section deformation function or modes    
    def warping_fun_u(self): #GBT u(theta) function 
        if self.k.split("|")[0]=='t' or  self.k.split("|")[0]=='a' or  self.k.split("|")[1]== 'v' :        
           return (0*cos(0))
        elif self.k.split("|")[0]=='1':
           return (1.0*cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'u') :
           return (self.r*sin(self.m*self.vertheta)/self.R)
        elif int(self.k.split("|")[0]) % 2 != 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'u')  :
           return (-self.r*cos(self.m*self.vertheta)/self.R) 
           
    def warping_fun_u_1xdiff(self): #GBT u(theta) function first derivative
        if self.k.split("|")[0]=='t' or  self.k.split("|")[0]=='a' or  self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' :        
           return (0*cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'u') :
           return (self.r*self.m*cos(self.m*self.vertheta)/self.R)
        elif int(self.k.split("|")[0]) % 2 != 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'u')  :
           return (self.r*self.m*sin(self.m*self.vertheta)/self.R) 

    def warping_fun_v(self): #GBT v(theta) function 
        if self.k.split("|")[0]=='t':        
           return (self.r*cos(0))
        elif self.k.split("|")[0]=='a' or  self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'u' : 
           return (0*cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'v') :
           return (-self.m*cos(self.m*self.vertheta))
        elif int(self.k.split("|")[0]) % 2 != 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'v') :
           return (-self.m*sin(self.m*self.vertheta))         
    def warping_fun_v_1xdiff(self): #GBT v(theta) function first derivative
        if self.k.split("|")[0]=='t' or  self.k.split("|")[0]=='a' or  self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'u' :        
           return (0*cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'v') :
           return (self.m**2*sin(self.m*self.vertheta))
        elif int(self.k.split("|")[0]) % 2 != 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'v')  :
           return (-self.m**2*cos(self.m*self.vertheta) )#/self.r 

    def warping_fun_w(self): #GBT w(theta) function 
        if self.k.split("|")[0]=='t' or  self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' or  self.k.split("|")[1]== 'u'  :        
           return (0*cos(0))
        elif self.k.split("|")[0]=='a': 
           return (1.0*cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and self.k.split("|")[1]== 'c':
           return (-self.m**2*sin(self.m*self.vertheta))
        elif int(self.k.split("|")[0]) % 2 != 0 and self.k.split("|")[1]== 'c' :
           return (self.m**2*cos(self.m*self.vertheta))
    def warping_fun_w_1xdiff(self): #GBT w(theta) function  first derivative
        if self.k.split("|")[0]=='t' or self.k.split("|")[0]=='a' or self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' or  self.k.split("|")[1]== 'u'  :        
           return (0*cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and self.k.split("|")[1]== 'c':
           return (-self.m**3*cos(self.m*self.vertheta))
        elif int(self.k.split("|")[0]) % 2 != 0 and self.k.split("|")[1]== 'c' :
           return (-self.m**3*sin(self.m*self.vertheta))    #/self.r   
    def warping_fun_w_2xdiff(self): #GBT w(theta) function  first derivative
        if self.k.split("|")[0]=='t' or self.k.split("|")[0]=='a' or self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' or  self.k.split("|")[1]== 'u'  :        
           return (0*cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and self.k.split("|")[1]== 'c':
           return (self.m**4*sin(self.m*self.vertheta))
        elif int(self.k.split("|")[0]) % 2 != 0 and self.k.split("|")[1]== 'c' :
           return (-self.m**4*cos(self.m*self.vertheta)) #/self.r**2
    def warping_fun_w_3xdiff(self): #GBT w(theta) function  first derivative
        if self.k.split("|")[0]=='t' or self.k.split("|")[0]=='a' or self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' or  self.k.split("|")[1]== 'u'  :        
           return (0*cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and self.k.split("|")[1]== 'c':
           return (self.m**5*cos(self.m*self.vertheta))
        elif int(self.k.split("|")[0]) % 2 != 0 and self.k.split("|")[1]== 'c' :
           return (self.m**5*sin(self.m*self.vertheta)) #/self.r**3


#_____________________________________________________________
class GBT_func_curve_without:
    def __init__ (self, k,r, vertheta,R):
        self.k=k
        if k.split("|")[0]=='t' or  k.split("|")[0]=='a' or  k.split("|")[0]=='1':
            self.m= 0
        elif  int(k.split("|")[0])  % 2 == 0 :
            self.m=int(k.split("|")[0])/2
        elif  int(k.split("|")[0]) % 2 != 0 :
            self.m=(int(k.split("|")[0])-1)/2
        self.r=r
        self.R=R
        self.vertheta=vertheta
#        self.vertheta=Symbol('vertheta')

    # section deformation function or modes    
    def warping_fun_u(self): #GBT u(theta) function 
        if self.k.split("|")[0]=='t' or  self.k.split("|")[0]=='a' or  self.k.split("|")[1]== 'v' :        
           return (0*cos(0))
        elif self.k.split("|")[0]=='1':
           return (1.0*cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'u') :
           return (self.r*sin(self.m*self.vertheta)/self.R)
        elif int(self.k.split("|")[0]) % 2 != 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'u')  :
           return (-self.r*cos(self.m*self.vertheta)/self.R) 
           
    def warping_fun_u_1xdiff(self): #GBT u(theta) function first derivative
        if self.k.split("|")[0]=='t' or  self.k.split("|")[0]=='a' or  self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' :        
           return (0*cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'u') :
           return (self.r*self.m*cos(self.m*self.vertheta)/self.R)
        elif int(self.k.split("|")[0]) % 2 != 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'u')  :
           return (self.r*self.m*sin(self.m*self.vertheta)/self.R) 

    def warping_fun_v(self): #GBT v(theta) function 
        if self.k.split("|")[0]=='t':        
           return (self.r*cos(0))
        elif self.k.split("|")[0]=='a' or  self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'u' : 
           return (0*cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'v') :
           return (-self.m*cos(self.m*self.vertheta)- 0.1e1 / self.R * self.r * sin(self.m * self.vertheta) * sin(self.vertheta))
        elif int(self.k.split("|")[0]) % 2 != 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'v') :
           return (0.1e1 / self.R * self.r * cos(self.m * self.vertheta) * sin(self.vertheta) - self.m * sin(self.m * self.vertheta))         
    def warping_fun_v_1xdiff(self): #GBT v(theta) function first derivative
        if self.k.split("|")[0]=='t' or  self.k.split("|")[0]=='a' or  self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'u' :        
           return (0*cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'v') :
           return (cos(self.vertheta) * self.m ** 2 * sin(self.m * self.vertheta) * self.r / self.R + self.m ** 2 * sin(self.m * self.vertheta) - 0.1e1 / self.R * self.r * sin(self.m * self.vertheta) * cos(self.vertheta))
        elif int(self.k.split("|")[0]) % 2 != 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'v')  :
           return (-0.1e1 / self.R * self.r * self.m * sin(self.m * self.vertheta) * sin(self.vertheta) + 0.1e1 / self.R * self.r * cos(self.m * self.vertheta) * cos(self.vertheta) - self.m ** 2 * cos(self.m * self.vertheta)) #/self.r 

    def warping_fun_w(self): #GBT w(theta) function 
        if self.k.split("|")[0]=='t' or  self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' or  self.k.split("|")[1]== 'u'  :        
           return (0*cos(0))
        elif self.k.split("|")[0]=='a': 
           return (1.0*cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and self.k.split("|")[1]== 'c':
           return (-self.m ** 2 * sin(self.m * self.vertheta) + 0.1e1 / self.R * self.r * self.m * cos(self.m * self.vertheta) * sin(self.vertheta) + 0.1e1 / self.R * self.r * sin(self.m * self.vertheta) * cos(self.vertheta))
        elif int(self.k.split("|")[0]) % 2 != 0 and self.k.split("|")[1]== 'c' :
           return (0.1e1 / self.R * self.r * self.m * sin(self.m * self.vertheta) * sin(self.vertheta) - 0.1e1 / self.R * self.r * cos(self.m * self.vertheta) * cos(self.vertheta) + self.m ** 2 * cos(self.m * self.vertheta))
    def warping_fun_w_1xdiff(self): #GBT w(theta) function  first derivative
        if self.k.split("|")[0]=='t' or self.k.split("|")[0]=='a' or self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' or  self.k.split("|")[1]== 'u'  :        
           return (0*cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and self.k.split("|")[1]== 'c':
           return (sin(self.vertheta) * self.m ** 2 * sin(self.m * self.vertheta) * self.r / self.R - cos(self.vertheta) * self.m ** 3 * cos(self.m * self.vertheta) * self.r / self.R - self.m ** 3 * cos(self.m * self.vertheta) + cos(self.vertheta) * cos(self.m * self.vertheta) * self.m * self.r / self.R - 0.1e1 / self.R * self.r * sin(self.m * self.vertheta) * sin(self.vertheta))
        elif int(self.k.split("|")[0]) % 2 != 0 and self.k.split("|")[1]== 'c' :
           return (0.1e1 / self.R * self.r * self.m ** 2 * cos(self.m * self.vertheta) * sin(self.vertheta) + 2 / self.R * self.r * self.m * sin(self.m * self.vertheta) * cos(self.vertheta) + 0.1e1 / self.R * self.r * cos(self.m * self.vertheta) * sin(self.vertheta) - self.m ** 3 * sin(self.m * self.vertheta))    #/self.r   
    def warping_fun_w_2xdiff(self): #GBT w(theta) function  first derivative
        if self.k.split("|")[0]=='t' or self.k.split("|")[0]=='a' or self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' or  self.k.split("|")[1]== 'u'  :        
           return (0*cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and self.k.split("|")[1]== 'c':
           return (2 * sin(self.vertheta) * self.m ** 3 * cos(self.m * self.vertheta) * self.r / self.R + cos(self.vertheta) * self.m ** 4 * sin(self.m * self.vertheta) * self.r / self.R + self.m ** 4 * sin(self.m * self.vertheta) - 2 / self.R * self.r * self.m * cos(self.m * self.vertheta) * sin(self.vertheta) - 0.1e1 / self.R * self.r * sin(self.m * self.vertheta) * cos(self.vertheta))
        elif int(self.k.split("|")[0]) % 2 != 0 and self.k.split("|")[1]== 'c' :
           return (-0.1e1 / self.R * self.r * self.m ** 3 * sin(self.m * self.vertheta) * sin(self.vertheta) + 3 / self.R * self.r * self.m ** 2 * cos(self.m * self.vertheta) * cos(self.vertheta) - 3 / self.R * self.r * self.m * sin(self.m * self.vertheta) * sin(self.vertheta) + 0.1e1 / self.R * self.r * cos(self.m * self.vertheta) * cos(self.vertheta) - self.m ** 4 * cos(self.m * self.vertheta)) #/self.r**2
    def warping_fun_w_3xdiff(self): #GBT w(theta) function  first derivative
        if self.k.split("|")[0]=='t' or self.k.split("|")[0]=='a' or self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' or  self.k.split("|")[1]== 'u'  :        
           return (0*cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and self.k.split("|")[1]== 'c':
           return ( 2 * cos(self.vertheta) * self.m ** 3 * cos(self.m * self.vertheta) * self.r / self.R - 3 * sin(self.vertheta) * self.m ** 4 * sin(self.m * self.vertheta) * self.r / self.R + cos(self.vertheta) * self.m ** 5 * cos(self.m * self.vertheta) * self.r / self.R + self.m ** 5 * cos(self.m * self.vertheta) - 3 * cos(self.vertheta) * cos(self.m * self.vertheta) * self.m * self.r / self.R + 2 * sin(self.vertheta) * self.m ** 2 * sin(self.m * self.vertheta) * self.r / self.R + 0.1e1 / self.R * self.r * sin(self.m * self.vertheta) * sin(self.vertheta))
        elif int(self.k.split("|")[0]) % 2 != 0 and self.k.split("|")[1]== 'c' :
           return (-0.1e1 / self.R * self.r * self.m ** 4 * cos(self.m * self.vertheta) * sin(self.vertheta) - 4 / self.R * self.r * self.m ** 3 * sin(self.m * self.vertheta) * cos(self.vertheta) - 6 / self.R * self.r * self.m ** 2 * cos(self.m * self.vertheta) * sin(self.vertheta) - 4 / self.R * self.r * self.m * sin(self.m * self.vertheta) * cos(self.vertheta) - 0.1e1 / self.R * self.r * cos(self.m * self.vertheta) * sin(self.vertheta) + self.m ** 5 * sin(self.m * self.vertheta)) #/self.r**3
#______________________________________
class GBT_func_curve:#_with: #with xi
    def __init__ (self, k,r, vertheta,R):
        self.k=k
        if k.split("|")[0]=='t' or  k.split("|")[0]=='a' or  k.split("|")[0]=='1':
            self.m= 0
        elif  int(k.split("|")[0])  % 2 == 0 :
            self.m=int(k.split("|")[0])/2
        elif  int(k.split("|")[0]) % 2 != 0 :
            self.m=(int(k.split("|")[0])-1)/2
        self.r=r
        self.R=R

        self.vertheta=vertheta
    #    self.vertheta=Symbol('vertheta')
  #      self.xi=(1+(1*(r/R)*cos(self.vertheta)))
    # section deformation function or modes    
    def warping_fun_u(self): #GBT u(theta) function 
        if self.k.split("|")[0]=='t' or  self.k.split("|")[0]=='a' or  self.k.split("|")[1]== 'v' :        
           return (0*cos(0))
        elif self.k.split("|")[0]=='1':
           return (1.0*cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'u') :
           return (self.r * sin(self.m * self.vertheta) / self.R)
        elif int(self.k.split("|")[0]) % 2 != 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'u')  :
           return (-self.r / self.R * cos(self.m * self.vertheta)) 
           
    def warping_fun_u_1xdiff(self): #GBT u(theta) function first derivative
        if self.k.split("|")[0]=='t' or  self.k.split("|")[0]=='a' or  self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' :        
            return (0*cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'u') :
            return (self.r * self.m * cos(self.m * self.vertheta) / self.R)
        elif int(self.k.split("|")[0]) % 2 != 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'u')  :
            return (self.r / self.R * self.m * sin(self.m * self.vertheta)) 

    def warping_fun_v(self): #GBT v(theta) function 
        if self.k.split("|")[0]=='t':        
            return (self.r*cos(0))
        elif self.k.split("|")[0]=='a' or  self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'u' : 
            return (0*cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'v') :
            return (-0.1e1 / self.R * cos(self.vertheta) * cos(self.m * self.vertheta) * self.m * self.r - self.m * cos(self.m * self.vertheta) - 0.1e1 / self.R * self.r * sin(self.m * self.vertheta) * sin(self.vertheta))
        elif int(self.k.split("|")[0]) % 2 != 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'v') :
            return (-0.1e1 / self.R * cos(self.vertheta) * sin(self.m * self.vertheta) * self.m * self.r + 0.1e1 / self.R * self.r * cos(self.m * self.vertheta) * sin(self.vertheta) - self.m * sin(self.m * self.vertheta))         

    def warping_fun_v_1xdiff(self): #GBT v(theta) function first derivative
        
        if self.k.split("|")[0]=='t' or  self.k.split("|")[0]=='a' or  self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'u' :        
            return (0*cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'v') :
            return (0.1e1 / self.R * cos(self.vertheta) * self.m ** 2 * sin(self.m * self.vertheta) * self.r + self.m ** 2 * sin(self.m * self.vertheta) - 0.1e1 / self.R * self.r * sin(self.m * self.vertheta) * cos(self.vertheta))
        elif int(self.k.split("|")[0]) % 2 != 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'v')  :
            return (-0.1e1 / self.R * cos(self.vertheta) * self.m ** 2 * cos(self.m * self.vertheta) * self.r + 0.1e1 / self.R * self.r * cos(self.m * self.vertheta) * cos(self.vertheta) - self.m ** 2 * cos(self.m * self.vertheta))

    def warping_fun_w(self): #GBT w(theta) function 
      
        
        if self.k.split("|")[0]=='t' or  self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' or  self.k.split("|")[1]== 'u'  :        
            return (0*cos(0))
        elif self.k.split("|")[0]=='a': 
            return (1.0*cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and self.k.split("|")[1]== 'c':
            return (-0.1e1 / self.R * cos(self.vertheta) * self.m ** 2 * sin(self.m * self.vertheta) * self.r - self.m ** 2 * sin(self.m * self.vertheta) + 0.1e1 / self.R * self.r * sin(self.m * self.vertheta) * cos(self.vertheta))
        elif int(self.k.split("|")[0]) % 2 != 0 and self.k.split("|")[1]== 'c' :
            return (0.1e1 / self.R * cos(self.vertheta) * self.m ** 2 * cos(self.m * self.vertheta) * self.r - 0.1e1 / self.R * self.r * cos(self.m * self.vertheta) * cos(self.vertheta) + self.m ** 2 * cos(self.m * self.vertheta))

    def warping_fun_w_1xdiff(self): #GBT w(theta) function  first derivative

        if self.k.split("|")[0]=='t' or self.k.split("|")[0]=='a' or self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' or  self.k.split("|")[1]== 'u'  :        
            return (0*cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and self.k.split("|")[1]== 'c':
            return (0.1e1 / self.R * sin(self.vertheta) * self.m ** 2 * sin(self.m * self.vertheta) * self.r - 0.1e1 / self.R * cos(self.vertheta) * self.m ** 3 * cos(self.m * self.vertheta) * self.r - self.m ** 3 * cos(self.m * self.vertheta) + 0.1e1 / self.R * cos(self.vertheta) * cos(self.m * self.vertheta) * self.m * self.r - 0.1e1 / self.R * self.r * sin(self.m * self.vertheta) * sin(self.vertheta))
        elif int(self.k.split("|")[0]) % 2 != 0 and self.k.split("|")[1]== 'c' :
            return (-0.1e1 / self.R * sin(self.vertheta) * self.m ** 2 * cos(self.m * self.vertheta) * self.r - 0.1e1 / self.R * cos(self.vertheta) * self.m ** 3 * sin(self.m * self.vertheta) * self.r + 0.1e1 / self.R * cos(self.vertheta) * sin(self.m * self.vertheta) * self.m * self.r + 0.1e1 / self.R * self.r * cos(self.m * self.vertheta) * sin(self.vertheta) - self.m ** 3 * sin(self.m * self.vertheta))
    def warping_fun_w_2xdiff(self): #GBT w(theta) function  first derivative

        if self.k.split("|")[0]=='t' or self.k.split("|")[0]=='a' or self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' or  self.k.split("|")[1]== 'u'  :        
            return (0*cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and self.k.split("|")[1]== 'c':
            return (2 / self.R * sin(self.vertheta) * self.m ** 3 * cos(self.m * self.vertheta) * self.r + 0.1e1 / self.R * cos(self.vertheta) * self.m ** 4 * sin(self.m * self.vertheta) * self.r + self.m ** 4 * sin(self.m * self.vertheta) - 2 / self.R * sin(self.vertheta) * cos(self.m * self.vertheta) * self.m * self.r - 0.1e1 / self.R * self.r * sin(self.m * self.vertheta) * cos(self.vertheta))
        elif int(self.k.split("|")[0]) % 2 != 0 and self.k.split("|")[1]== 'c' :
            return (2 / self.R * sin(self.vertheta) * self.m ** 3 * sin(self.m * self.vertheta) * self.r - 0.1e1 / self.R * cos(self.vertheta) * self.m ** 4 * cos(self.m * self.vertheta) * self.r - 2 / self.R * sin(self.vertheta) * sin(self.m * self.vertheta) * self.m * self.r + 0.1e1 / self.R * self.r * cos(self.m * self.vertheta) * cos(self.vertheta) - self.m ** 4 * cos(self.m * self.vertheta))

    def warping_fun_w_3xdiff(self): #GBT w(theta) function  first derivative

        if self.k.split("|")[0]=='t' or self.k.split("|")[0]=='a' or self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' or  self.k.split("|")[1]== 'u'  :        
            return (0*cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and self.k.split("|")[1]== 'c':
            return ( 2 / self.R * cos(self.vertheta) * self.m ** 3 * cos(self.m * self.vertheta) * self.r - 3 / self.R * sin(self.vertheta) * self.m ** 4 * sin(self.m * self.vertheta) * self.r + 0.1e1 / self.R * cos(self.vertheta) * self.m ** 5 * cos(self.m * self.vertheta) * self.r + self.m ** 5 * cos(self.m * self.vertheta) - 3 / self.R * cos(self.vertheta) * cos(self.m * self.vertheta) * self.m * self.r + 2 / self.R * sin(self.vertheta) * self.m ** 2 * sin(self.m * self.vertheta) * self.r + 0.1e1 / self.R * self.r * sin(self.m * self.vertheta) * sin(self.vertheta))
        elif int(self.k.split("|")[0]) % 2 != 0 and self.k.split("|")[1]== 'c' :
            return (2 / self.R * cos(self.vertheta) * self.m ** 3 * sin(self.m * self.vertheta) * self.r + 3 / self.R * sin(self.vertheta) * self.m ** 4 * cos(self.m * self.vertheta) * self.r + 0.1e1 / self.R * cos(self.vertheta) * self.m ** 5 * sin(self.m * self.vertheta) * self.r - 3 / self.R * cos(self.vertheta) * sin(self.m * self.vertheta) * self.m * self.r - 2 / self.R * sin(self.vertheta) * self.m ** 2 * cos(self.m * self.vertheta) * self.r - 0.1e1 / self.R * self.r * cos(self.m * self.vertheta) * sin(self.vertheta) + self.m ** 5 * sin(self.m * self.vertheta))
#____________________________________________________
class GBT_func_curve_full:#_with: #symbolic diff in python
    def __init__ (self, k,r, vertheta,R):
        self.k=k
        if k.split("|")[0]=='t' or  k.split("|")[0]=='a' or  k.split("|")[0]=='1':
            self.m= 0
        elif  int(k.split("|")[0])  % 2 == 0 :
            self.m=int(k.split("|")[0])/2
        elif  int(k.split("|")[0]) % 2 != 0 :
            self.m=(int(k.split("|")[0])-1)/2
        self.r=r
        self.R=R

 #       self.vertheta=vertheta
        self.vertheta=Symbol('vertheta')
        self.xi=(1+(1*(r/R)*cos(self.vertheta)))
        self.coff=1
    # section deformation function or modes    
    def warping_fun_u(self): #GBT u(theta) function 
        if self.k.split("|")[0]=='t' or  self.k.split("|")[0]=='a' or  self.k.split("|")[1]== 'v' :        
           return (0*cos(0))
        elif self.k.split("|")[0]=='1':
           return (1.0*cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'u') :
           return (self.r * sin(self.m * self.vertheta) / self.R)
        elif int(self.k.split("|")[0]) % 2 != 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'u')  :
           return (-self.r / self.R * cos(self.m * self.vertheta)) 
    def warping_fun_u_for_vw(self): #GBT u(theta) function 
        if int(self.k.split("|")[0])  % 2 == 0  :
           return (self.r * sin(self.m * self.vertheta) / self.R)
        elif int(self.k.split("|")[0]) % 2 != 0 :
           return (-self.r / self.R * cos(self.m * self.vertheta))            
    def warping_fun_u_1xdiff(self): #GBT u(theta) function first derivative
       
        if self.k.split("|")[0]=='t' or  self.k.split("|")[0]=='a' or  self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' :        
            return (0*cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'u') :
            return (diff(self.warping_fun_u(), self.vertheta,1))
        elif int(self.k.split("|")[0]) % 2 != 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'u')  :
            return (diff(self.warping_fun_u(), self.vertheta,1)) 

    def warping_fun_v(self): #GBT v(theta) function 
    
        if self.k.split("|")[0]=='t':        
            return (self.r*cos(0))
        elif self.k.split("|")[0]=='a' or  self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'u' : 
            return (0*cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'v') :
            return (-self.R*self.xi*(diff(self.warping_fun_u_for_vw(), self.vertheta,1)/self.r+ self.coff*self.warping_fun_u_for_vw()*sin(self.vertheta)/(self.R*self.xi)))
        elif int(self.k.split("|")[0]) % 2 != 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'v') :
            return (-self.R*self.xi*(diff(self.warping_fun_u_for_vw(), self.vertheta,1)/self.r+ self.coff*self.warping_fun_u_for_vw()*sin(self.vertheta)/(self.R*self.xi)))

    def warping_fun_v_1xdiff(self): #GBT v(theta) function first derivative
        return (diff(self.warping_fun_v(), self.vertheta,1))
 
    def warping_fun_w(self): #GBT w(theta) function 

 
        if self.k.split("|")[0]=='t' or  self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' or  self.k.split("|")[1]== 'u'  :        
            return (0*cos(0))
        elif self.k.split("|")[0]=='a': 
            return (1.0*cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and self.k.split("|")[1]== 'c':
            return (-diff(-self.R*self.xi*(diff(self.warping_fun_u_for_vw(), self.vertheta,1)/self.r+ self.coff*self.warping_fun_u_for_vw()*sin(self.vertheta)/(self.R*self.xi)), self.vertheta,1))
        elif int(self.k.split("|")[0]) % 2 != 0 and self.k.split("|")[1]== 'c' :
            return (-diff(-self.R*self.xi*(diff(self.warping_fun_u_for_vw(), self.vertheta,1)/self.r+ self.coff*self.warping_fun_u_for_vw()*sin(self.vertheta)/(self.R*self.xi)), self.vertheta,1))

    def warping_fun_w_1xdiff(self): #GBT w(theta) function  first derivative

        if self.k.split("|")[0]=='t' or self.k.split("|")[0]=='a' or self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' or  self.k.split("|")[1]== 'u'  :        
            return (0*cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and self.k.split("|")[1]== 'c':
            return (diff(self.warping_fun_w(), self.vertheta,1))
        elif int(self.k.split("|")[0]) % 2 != 0 and self.k.split("|")[1]== 'c' :
            return (diff(self.warping_fun_w(), self.vertheta,1))
        
    def warping_fun_w_2xdiff(self): #GBT w(theta) function  first derivative
      #  
        if self.k.split("|")[0]=='t' or self.k.split("|")[0]=='a' or self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' or  self.k.split("|")[1]== 'u'  :        
            return (0*cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and self.k.split("|")[1]== 'c':
            return (diff(self.warping_fun_w(), self.vertheta,2))
        elif int(self.k.split("|")[0]) % 2 != 0 and self.k.split("|")[1]== 'c' :
            return (diff(self.warping_fun_w(), self.vertheta,2))

    def warping_fun_w_3xdiff(self): #GBT w(theta) function  first derivative
        
        if self.k.split("|")[0]=='t' or self.k.split("|")[0]=='a' or self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' or  self.k.split("|")[1]== 'u'  :        
            return (0*cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and self.k.split("|")[1]== 'c':
            return (diff(self.warping_fun_w(), self.vertheta,3))
        elif int(self.k.split("|")[0]) % 2 != 0 and self.k.split("|")[1]== 'c' :
            return (diff(self.warping_fun_w(), self.vertheta,3))

#______________________________________  
           
class GBT_func_numpy_curve_orginal:
    def __init__ (self, k,r, vertheta,R):
        self.k=k
        if k.split("|")[0]=='t' or  k.split("|")[0]=='a' or  k.split("|")[0]=='1':
            self.m= 0
        elif  int(k.split("|")[0])  % 2 == 0 :
            self.m=int(k.split("|")[0])/2
        elif  int(k.split("|")[0]) % 2 != 0 :
            self.m=(int(k.split("|")[0])-1)/2
        self.r=r
        self.R=R
        self.vertheta=vertheta
#        self.vertheta=Symbol('vertheta')

    # section deformation function or modes    
    def warping_fun_u(self): #GBT u(theta) function 
        if self.k.split("|")[0]=='t' or  self.k.split("|")[0]=='a' or  self.k.split("|")[1]== 'v' :        
           return (0*np.cos(0))
        elif self.k.split("|")[0]=='1':
           return (1.0*np.cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'u') :
           return (self.r*np.sin(self.m*self.vertheta)/self.R)
        elif int(self.k.split("|")[0]) % 2 != 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'u')  :
           return (-self.r*np.cos(self.m*self.vertheta)/self.R) 
           
    def warping_fun_u_1xdiff(self): #GBT u(theta) function first derivative
        if self.k.split("|")[0]=='t' or  self.k.split("|")[0]=='a' or  self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' :        
           return (0*np.cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'u') :
           return (self.r*self.m*np.cos(self.m*self.vertheta)/self.R)
        elif int(self.k.split("|")[0]) % 2 != 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'u')  :
           return (self.r*self.m*np.sin(self.m*self.vertheta)/self.R) 

    def warping_fun_v(self): #GBT v(theta) function 
        if self.k.split("|")[0]=='t':        
           return (self.r*np.cos(0))
        elif self.k.split("|")[0]=='a' or  self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'u' : 
           return (0*np.cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'v') :
           return (-self.m*np.cos(self.m*self.vertheta))
        elif int(self.k.split("|")[0]) % 2 != 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'v') :
           return (-self.m*np.sin(self.m*self.vertheta))         
    def warping_fun_v_1xdiff(self): #GBT v(theta) function first derivative
        if self.k.split("|")[0]=='t' or  self.k.split("|")[0]=='a' or  self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'u' :        
           return (0*np.cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'v') :
           return (self.m**2*np.sin(self.m*self.vertheta))
        elif int(self.k.split("|")[0]) % 2 != 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'v')  :
           return (-self.m**2*np.cos(self.m*self.vertheta) )#/self.r 

    def warping_fun_w(self): #GBT w(theta) function 
        if self.k.split("|")[0]=='t' or  self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' or  self.k.split("|")[1]== 'u'  :        
           return (0*np.cos(0))
        elif self.k.split("|")[0]=='a': 
           return (1.0*np.cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and self.k.split("|")[1]== 'c':
           return (-self.m**2*np.sin(self.m*self.vertheta))
        elif int(self.k.split("|")[0]) % 2 != 0 and self.k.split("|")[1]== 'c' :
           return (self.m**2*np.cos(self.m*self.vertheta))
    def warping_fun_w_1xdiff(self): #GBT w(theta) function  first derivative
        if self.k.split("|")[0]=='t' or self.k.split("|")[0]=='a' or self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' or  self.k.split("|")[1]== 'u'  :        
           return (0*np.cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and self.k.split("|")[1]== 'c':
           return (-self.m**3*np.cos(self.m*self.vertheta))
        elif int(self.k.split("|")[0]) % 2 != 0 and self.k.split("|")[1]== 'c' :
           return (-self.m**3*np.sin(self.m*self.vertheta))    #/self.r   
    def warping_fun_w_2xdiff(self): #GBT w(theta) function  first derivative
        if self.k.split("|")[0]=='t' or self.k.split("|")[0]=='a' or self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' or  self.k.split("|")[1]== 'u'  :        
           return (0*np.cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and self.k.split("|")[1]== 'c':
           return (self.m**4*np.sin(self.m*self.vertheta))
        elif int(self.k.split("|")[0]) % 2 != 0 and self.k.split("|")[1]== 'c' :
           return (-self.m**4*np.cos(self.m*self.vertheta)) #/self.r**2
    def warping_fun_w_3xdiff(self): #GBT w(theta) function  first derivative
        if self.k.split("|")[0]=='t' or self.k.split("|")[0]=='a' or self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' or  self.k.split("|")[1]== 'u'  :        
           return (0*np.cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and self.k.split("|")[1]== 'c':
           return (self.m**5*np.cos(self.m*self.vertheta))
        elif int(self.k.split("|")[0]) % 2 != 0 and self.k.split("|")[1]== 'c' :
           return (self.m**5*np.sin(self.m*self.vertheta)) #/self.r**3


#_____________________________________________________________
class GBT_func_numpy_curve_without:
    def __init__ (self, k,r, vertheta,R):
        self.k=k
        if k.split("|")[0]=='t' or  k.split("|")[0]=='a' or  k.split("|")[0]=='1':
            self.m= 0
        elif  int(k.split("|")[0])  % 2 == 0 :
            self.m=int(k.split("|")[0])/2
        elif  int(k.split("|")[0]) % 2 != 0 :
            self.m=(int(k.split("|")[0])-1)/2
        self.r=r
        self.R=R
        self.vertheta=vertheta
#        self.vertheta=Symbol('vertheta')

    # section deformation function or modes    
    def warping_fun_u(self): #GBT u(theta) function 
        if self.k.split("|")[0]=='t' or  self.k.split("|")[0]=='a' or  self.k.split("|")[1]== 'v' :        
           return (0*np.cos(0))
        elif self.k.split("|")[0]=='1':
           return (1.0*np.cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'u') :
           return (self.r*np.sin(self.m*self.vertheta)/self.R)
        elif int(self.k.split("|")[0]) % 2 != 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'u')  :
           return (-self.r*np.cos(self.m*self.vertheta)/self.R) 
           
    def warping_fun_u_1xdiff(self): #GBT u(theta) function first derivative
        if self.k.split("|")[0]=='t' or  self.k.split("|")[0]=='a' or  self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' :        
           return (0*np.cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'u') :
           return (self.r*self.m*np.cos(self.m*self.vertheta)/self.R)
        elif int(self.k.split("|")[0]) % 2 != 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'u')  :
           return (self.r*self.m*np.sin(self.m*self.vertheta)/self.R) 

    def warping_fun_v(self): #GBT v(theta) function 
        if self.k.split("|")[0]=='t':        
           return (self.r*np.cos(0))
        elif self.k.split("|")[0]=='a' or  self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'u' : 
           return (0*np.cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'v') :
           return (-self.m*np.cos(self.m*self.vertheta)- 0.1e1 / self.R * self.r * np.sin(self.m * self.vertheta) * np.sin(self.vertheta))
        elif int(self.k.split("|")[0]) % 2 != 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'v') :
           return (0.1e1 / self.R * self.r * np.cos(self.m * self.vertheta) * np.sin(self.vertheta) - self.m * np.sin(self.m * self.vertheta))         
    def warping_fun_v_1xdiff(self): #GBT v(theta) function first derivative
        if self.k.split("|")[0]=='t' or  self.k.split("|")[0]=='a' or  self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'u' :        
           return (0*np.cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'v') :
           return (np.cos(self.vertheta) * self.m ** 2 * np.sin(self.m * self.vertheta) * self.r / self.R + self.m ** 2 * np.sin(self.m * self.vertheta) - 0.1e1 / self.R * self.r * np.sin(self.m * self.vertheta) * np.cos(self.vertheta))
        elif int(self.k.split("|")[0]) % 2 != 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'v')  :
           return (-0.1e1 / self.R * self.r * self.m * np.sin(self.m * self.vertheta) * np.sin(self.vertheta) + 0.1e1 / self.R * self.r * np.cos(self.m * self.vertheta) * np.cos(self.vertheta) - self.m ** 2 * np.cos(self.m * self.vertheta)) #/self.r 

    def warping_fun_w(self): #GBT w(theta) function 
        if self.k.split("|")[0]=='t' or  self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' or  self.k.split("|")[1]== 'u'  :        
           return (0*np.cos(0))
        elif self.k.split("|")[0]=='a': 
           return (1.0*np.cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and self.k.split("|")[1]== 'c':
           return (-self.m ** 2 * np.sin(self.m * self.vertheta) + 0.1e1 / self.R * self.r * self.m * np.cos(self.m * self.vertheta) * np.sin(self.vertheta) + 0.1e1 / self.R * self.r * np.sin(self.m * self.vertheta) * np.cos(self.vertheta))
        elif int(self.k.split("|")[0]) % 2 != 0 and self.k.split("|")[1]== 'c' :
           return (0.1e1 / self.R * self.r * self.m * np.sin(self.m * self.vertheta) * np.sin(self.vertheta) - 0.1e1 / self.R * self.r * np.cos(self.m * self.vertheta) * np.cos(self.vertheta) + self.m ** 2 * np.cos(self.m * self.vertheta))
    def warping_fun_w_1xdiff(self): #GBT w(theta) function  first derivative
        if self.k.split("|")[0]=='t' or self.k.split("|")[0]=='a' or self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' or  self.k.split("|")[1]== 'u'  :        
           return (0*np.cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and self.k.split("|")[1]== 'c':
           return (np.sin(self.vertheta) * self.m ** 2 * np.sin(self.m * self.vertheta) * self.r / self.R - np.cos(self.vertheta) * self.m ** 3 * np.cos(self.m * self.vertheta) * self.r / self.R - self.m ** 3 * np.cos(self.m * self.vertheta) + np.cos(self.vertheta) * np.cos(self.m * self.vertheta) * self.m * self.r / self.R - 0.1e1 / self.R * self.r * np.sin(self.m * self.vertheta) * np.sin(self.vertheta))
        elif int(self.k.split("|")[0]) % 2 != 0 and self.k.split("|")[1]== 'c' :
           return (0.1e1 / self.R * self.r * self.m ** 2 * np.cos(self.m * self.vertheta) * np.sin(self.vertheta) + 2 / self.R * self.r * self.m * np.sin(self.m * self.vertheta) * np.cos(self.vertheta) + 0.1e1 / self.R * self.r * np.cos(self.m * self.vertheta) * np.sin(self.vertheta) - self.m ** 3 * np.sin(self.m * self.vertheta))    #/self.r   
    def warping_fun_w_2xdiff(self): #GBT w(theta) function  first derivative
        if self.k.split("|")[0]=='t' or self.k.split("|")[0]=='a' or self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' or  self.k.split("|")[1]== 'u'  :        
           return (0*np.cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and self.k.split("|")[1]== 'c':
           return (2 * np.sin(self.vertheta) * self.m ** 3 * np.cos(self.m * self.vertheta) * self.r / self.R + np.cos(self.vertheta) * self.m ** 4 * np.sin(self.m * self.vertheta) * self.r / self.R + self.m ** 4 * np.sin(self.m * self.vertheta) - 2 / self.R * self.r * self.m * np.cos(self.m * self.vertheta) * np.sin(self.vertheta) - 0.1e1 / self.R * self.r * np.sin(self.m * self.vertheta) * np.cos(self.vertheta))
        elif int(self.k.split("|")[0]) % 2 != 0 and self.k.split("|")[1]== 'c' :
           return (-0.1e1 / self.R * self.r * self.m ** 3 * np.sin(self.m * self.vertheta) * np.sin(self.vertheta) + 3 / self.R * self.r * self.m ** 2 * np.cos(self.m * self.vertheta) * np.cos(self.vertheta) - 3 / self.R * self.r * self.m * np.sin(self.m * self.vertheta) * np.sin(self.vertheta) + 0.1e1 / self.R * self.r * np.cos(self.m * self.vertheta) * np.cos(self.vertheta) - self.m ** 4 * np.cos(self.m * self.vertheta)) #/self.r**2
    def warping_fun_w_3xdiff(self): #GBT w(theta) function  first derivative
        if self.k.split("|")[0]=='t' or self.k.split("|")[0]=='a' or self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' or  self.k.split("|")[1]== 'u'  :        
           return (0*np.cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and self.k.split("|")[1]== 'c':
           return ( 2 * np.cos(self.vertheta) * self.m ** 3 * np.cos(self.m * self.vertheta) * self.r / self.R - 3 * np.sin(self.vertheta) * self.m ** 4 * np.sin(self.m * self.vertheta) * self.r / self.R + np.cos(self.vertheta) * self.m ** 5 * np.cos(self.m * self.vertheta) * self.r / self.R + self.m ** 5 * np.cos(self.m * self.vertheta) - 3 * np.cos(self.vertheta) * np.cos(self.m * self.vertheta) * self.m * self.r / self.R + 2 * np.sin(self.vertheta) * self.m ** 2 * np.sin(self.m * self.vertheta) * self.r / self.R + 0.1e1 / self.R * self.r * np.sin(self.m * self.vertheta) * np.sin(self.vertheta))
        elif int(self.k.split("|")[0]) % 2 != 0 and self.k.split("|")[1]== 'c' :
           return (-0.1e1 / self.R * self.r * self.m ** 4 * np.cos(self.m * self.vertheta) * np.sin(self.vertheta) - 4 / self.R * self.r * self.m ** 3 * np.sin(self.m * self.vertheta) * np.cos(self.vertheta) - 6 / self.R * self.r * self.m ** 2 * np.cos(self.m * self.vertheta) * np.sin(self.vertheta) - 4 / self.R * self.r * self.m * np.sin(self.m * self.vertheta) * np.cos(self.vertheta) - 0.1e1 / self.R * self.r * np.cos(self.m * self.vertheta) * np.sin(self.vertheta) + self.m ** 5 * np.sin(self.m * self.vertheta)) #/self.r**3
#______________________________________
class GBT_func_numpy_curve:#_with: #with xi
    def __init__ (self, k,r, vertheta,R):
        self.k=k
        if k.split("|")[0]=='t' or  k.split("|")[0]=='a' or  k.split("|")[0]=='1':
            self.m= 0
        elif  int(k.split("|")[0])  % 2 == 0 :
            self.m=int(k.split("|")[0])/2
        elif  int(k.split("|")[0]) % 2 != 0 :
            self.m=(int(k.split("|")[0])-1)/2
        self.r=r
        self.R=R
        self.vertheta=vertheta
#        self.vertheta=Symbol('vertheta')

    # section deformation function or modes    
    def warping_fun_u(self): #GBT u(theta) function 
        if self.k.split("|")[0]=='t' or  self.k.split("|")[0]=='a' or  self.k.split("|")[1]== 'v' :        
           return (0*np.cos(0))
        elif self.k.split("|")[0]=='1':
           return (1.0*np.cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'u') :
           return (self.r * np.sin(self.m * self.vertheta) / self.R)
        elif int(self.k.split("|")[0]) % 2 != 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'u')  :
           return (-self.r * np.cos(self.m * self.vertheta)/ self.R) 
           
    def warping_fun_u_1xdiff(self): #GBT u(theta) function first derivative
        if self.k.split("|")[0]=='t' or  self.k.split("|")[0]=='a' or  self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' :        
           return (0*np.cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'u') :
           return (self.r * self.m * np.cos(self.m * self.vertheta) / self.R)
        elif int(self.k.split("|")[0]) % 2 != 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'u')  :
           return (self.r / self.R * self.m * np.sin(self.m * self.vertheta)) 

    def warping_fun_v(self): #GBT v(theta) function 
        if self.k.split("|")[0]=='t':        
           return (self.r*np.cos(0))
        elif self.k.split("|")[0]=='a' or  self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'u' : 
           return (0*np.cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'v') :
           return (-0.1e1 / self.R * np.cos(self.vertheta) * np.cos(self.m * self.vertheta) * self.m * self.r - self.m * np.cos(self.m * self.vertheta) - 0.1e1 / self.R * self.r * np.sin(self.m * self.vertheta) * np.sin(self.vertheta))
        elif int(self.k.split("|")[0]) % 2 != 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'v') :
           return (-0.1e1 / self.R * np.cos(self.vertheta) * np.sin(self.m * self.vertheta) * self.m * self.r + 0.1e1 / self.R * self.r * np.cos(self.m * self.vertheta) * np.sin(self.vertheta) - self.m * np.sin(self.m * self.vertheta))         

    def warping_fun_v_1xdiff(self): #GBT v(theta) function first derivative
        if self.k.split("|")[0]=='t' or  self.k.split("|")[0]=='a' or  self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'u' :        
           return (0*np.cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'v') :
           return (0.1e1 / self.R * np.cos(self.vertheta) * self.m ** 2 * np.sin(self.m * self.vertheta) * self.r + self.m ** 2 * np.sin(self.m * self.vertheta) - 0.1e1 / self.R * self.r * np.sin(self.m * self.vertheta) * np.cos(self.vertheta))
        elif int(self.k.split("|")[0]) % 2 != 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'v')  :
           return (-0.1e1 / self.R * np.cos(self.vertheta) * self.m ** 2 * np.cos(self.m * self.vertheta) * self.r + 0.1e1 / self.R * self.r * np.cos(self.m * self.vertheta) * np.cos(self.vertheta) - self.m ** 2 * np.cos(self.m * self.vertheta))

    def warping_fun_w(self): #GBT w(theta) function 
        if self.k.split("|")[0]=='t' or  self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' or  self.k.split("|")[1]== 'u'  :        
           return (0*np.cos(0))
        elif self.k.split("|")[0]=='a': 
           return (1.0*np.cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and self.k.split("|")[1]== 'c':
           return (-0.1e1 / self.R * np.cos(self.vertheta) * self.m ** 2 * np.sin(self.m * self.vertheta) * self.r - self.m ** 2 * np.sin(self.m * self.vertheta) + 0.1e1 / self.R * self.r * np.sin(self.m * self.vertheta) * np.cos(self.vertheta))
        elif int(self.k.split("|")[0]) % 2 != 0 and self.k.split("|")[1]== 'c' :
           return (0.1e1 / self.R * np.cos(self.vertheta) * self.m ** 2 * np.cos(self.m * self.vertheta) * self.r - 0.1e1 / self.R * self.r * np.cos(self.m * self.vertheta) * np.cos(self.vertheta) + self.m ** 2 * np.cos(self.m * self.vertheta))
    def warping_fun_w_1xdiff(self): #GBT w(theta) function  first derivative
        if self.k.split("|")[0]=='t' or self.k.split("|")[0]=='a' or self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' or  self.k.split("|")[1]== 'u'  :        
           return (0*np.cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and self.k.split("|")[1]== 'c':
           return (0.1e1 / self.R * np.sin(self.vertheta) * self.m ** 2 * np.sin(self.m * self.vertheta) * self.r - 0.1e1 / self.R * np.cos(self.vertheta) * self.m ** 3 * np.cos(self.m * self.vertheta) * self.r - self.m ** 3 * np.cos(self.m * self.vertheta) + 0.1e1 / self.R * np.cos(self.vertheta) * np.cos(self.m * self.vertheta) * self.m * self.r - 0.1e1 / self.R * self.r * np.sin(self.m * self.vertheta) * np.sin(self.vertheta))
        elif int(self.k.split("|")[0]) % 2 != 0 and self.k.split("|")[1]== 'c' :
           return (-0.1e1 / self.R * np.sin(self.vertheta) * self.m ** 2 * np.cos(self.m * self.vertheta) * self.r - 0.1e1 / self.R * np.cos(self.vertheta) * self.m ** 3 * np.sin(self.m * self.vertheta) * self.r + 0.1e1 / self.R * np.cos(self.vertheta) * np.sin(self.m * self.vertheta) * self.m * self.r + 0.1e1 / self.R * self.r * np.cos(self.m * self.vertheta) * np.sin(self.vertheta) - self.m ** 3 * np.sin(self.m * self.vertheta))
    def warping_fun_w_2xdiff(self): #GBT w(theta) function  first derivative
        if self.k.split("|")[0]=='t' or self.k.split("|")[0]=='a' or self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' or  self.k.split("|")[1]== 'u'  :        
           return (0*np.cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and self.k.split("|")[1]== 'c':
           return (2 / self.R * np.sin(self.vertheta) * self.m ** 3 * np.cos(self.m * self.vertheta) * self.r + 0.1e1 / self.R * np.cos(self.vertheta) * self.m ** 4 * np.sin(self.m * self.vertheta) * self.r + self.m ** 4 * np.sin(self.m * self.vertheta) - 2 / self.R * np.sin(self.vertheta) * np.cos(self.m * self.vertheta) * self.m * self.r - 0.1e1 / self.R * self.r * np.sin(self.m * self.vertheta) * np.cos(self.vertheta))
        elif int(self.k.split("|")[0]) % 2 != 0 and self.k.split("|")[1]== 'c' :
           return (2 / self.R * np.sin(self.vertheta) * self.m ** 3 * np.sin(self.m * self.vertheta) * self.r - 0.1e1 / self.R * np.cos(self.vertheta) * self.m ** 4 * np.cos(self.m * self.vertheta) * self.r - 2 / self.R * np.sin(self.vertheta) * np.sin(self.m * self.vertheta) * self.m * self.r + 0.1e1 / self.R * self.r * np.cos(self.m * self.vertheta) * np.cos(self.vertheta) - self.m ** 4 * np.cos(self.m * self.vertheta))
    def warping_fun_w_3xdiff(self): #GBT w(theta) function  first derivative
        if self.k.split("|")[0]=='t' or self.k.split("|")[0]=='a' or self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' or  self.k.split("|")[1]== 'u'  :        
           return (0*np.cos(0))
        elif int(self.k.split("|")[0])  % 2 == 0 and self.k.split("|")[1]== 'c':
           return ( 2 / self.R * np.cos(self.vertheta) * self.m ** 3 * np.cos(self.m * self.vertheta) * self.r - 3 / self.R * np.sin(self.vertheta) * self.m ** 4 * np.sin(self.m * self.vertheta) * self.r + 0.1e1 / self.R * np.cos(self.vertheta) * self.m ** 5 * np.cos(self.m * self.vertheta) * self.r + self.m ** 5 * np.cos(self.m * self.vertheta) - 3 / self.R * np.cos(self.vertheta) * np.cos(self.m * self.vertheta) * self.m * self.r + 2 / self.R * np.sin(self.vertheta) * self.m ** 2 * np.sin(self.m * self.vertheta) * self.r + 0.1e1 / self.R * self.r * np.sin(self.m * self.vertheta) * np.sin(self.vertheta))
        elif int(self.k.split("|")[0]) % 2 != 0 and self.k.split("|")[1]== 'c' :
           return (2 / self.R * np.cos(self.vertheta) * self.m ** 3 * np.sin(self.m * self.vertheta) * self.r + 3 / self.R * np.sin(self.vertheta) * self.m ** 4 * np.cos(self.m * self.vertheta) * self.r + 0.1e1 / self.R * np.cos(self.vertheta) * self.m ** 5 * np.sin(self.m * self.vertheta) * self.r - 3 / self.R * np.cos(self.vertheta) * np.sin(self.m * self.vertheta) * self.m * self.r - 2 / self.R * np.sin(self.vertheta) * self.m ** 2 * np.cos(self.m * self.vertheta) * self.r - 0.1e1 / self.R * self.r * np.cos(self.m * self.vertheta) * np.sin(self.vertheta) + self.m ** 5 * np.sin(self.m * self.vertheta))
       
        
        
        
# class GBT_func_numpy:
#     def __init__ (self, k,r, vertheta):
#         self.k=k
#         if k.split("|")[0]=='t' or  k.split("|")[0]=='a' or  k.split("|")[0]=='1':
#             self.m= 0
#         elif  int(k.split("|")[0])  % 2 == 0 :
#             self.m=int(k.split("|")[0])/2
#         elif  int(k.split("|")[0]) % 2 != 0 :
#             self.m=(int(k.split("|")[0])-1)/2
#         self.r=r
#         self.vertheta=vertheta

#     # section deformation function or modes    
#     def warping_fun_u(self): #GBT u(theta) function 
#         if self.k.split("|")[0]=='t' or  self.k.split("|")[0]=='a' or  self.k.split("|")[1]== 'v' :        
#            return (0*np.cos(0))
#         elif self.k.split("|")[0]=='1':
#            return (1.0*np.cos(0))
#         elif int(self.k.split("|")[0])  % 2 == 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'u') :
#            return (self.r*np.sin(self.m*self.vertheta))
#         elif int(self.k.split("|")[0]) % 2 != 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'u')  :
#            return (-self.r*np.cos(self.m*self.vertheta)) 
           
#     def warping_fun_u_1xdiff(self): #GBT u(theta) function first derivative
#         if self.k.split("|")[0]=='t' or  self.k.split("|")[0]=='a' or  self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' :        
#            return (0*np.cos(0))
#         elif int(self.k.split("|")[0])  % 2 == 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'u') :
#            return (self.r*self.m*np.cos(self.m*self.vertheta))
#         elif int(self.k.split("|")[0]) % 2 != 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'u')  :
#            return (self.r*self.m*np.sin(self.m*self.vertheta)) 

#     def warping_fun_v(self): #GBT v(theta) function 
#         if self.k.split("|")[0]=='t':        
#            return (self.r*np.cos(0))
#         elif self.k.split("|")[0]=='a' or  self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'u' : 
#            return (0*np.cos(0))
#         elif int(self.k.split("|")[0])  % 2 == 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'v') :
#            return (-self.m*np.cos(self.m*self.vertheta))
#         elif int(self.k.split("|")[0]) % 2 != 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'v') :
#            return (-self.m*np.sin(self.m*self.vertheta))         
#     def warping_fun_v_1xdiff(self): #GBT v(theta) function first derivative
#         if self.k.split("|")[0]=='t' or  self.k.split("|")[0]=='a' or  self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'u' :        
#            return (0*np.cos(0))
#         elif int(self.k.split("|")[0])  % 2 == 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'v') :
#            return (self.m**2*np.sin(self.m*self.vertheta))
#         elif int(self.k.split("|")[0]) % 2 != 0 and (self.k.split("|")[1]== 'c' or self.k.split("|")[1]== 'v')  :
#            return (-self.m**2*np.cos(self.m*self.vertheta)) #/self.r 

#     def warping_fun_w(self): #GBT w(theta) function 
#         if self.k.split("|")[0]=='t' or  self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' or  self.k.split("|")[1]== 'u'  :        
#            return (0*np.cos(0))
#         elif self.k.split("|")[0]=='a': 
#            return (1.0*np.cos(0))
#         elif int(self.k.split("|")[0])  % 2 == 0 and self.k.split("|")[1]== 'c':
#            return (-self.m**2*np.sin(self.m*self.vertheta))
#         elif int(self.k.split("|")[0]) % 2 != 0 and self.k.split("|")[1]== 'c' :
#            return (self.m**2*np.cos(self.m*self.vertheta))
#     def warping_fun_w_1xdiff(self): #GBT w(theta) function  first derivative
#         if self.k.split("|")[0]=='t' or self.k.split("|")[0]=='a' or self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' or  self.k.split("|")[1]== 'u'  :        
#            return (0*np.cos(0))
#         elif int(self.k.split("|")[0])  % 2 == 0 and self.k.split("|")[1]== 'c':
#            return (-self.m**3*np.cos(self.m*self.vertheta))
#         elif int(self.k.split("|")[0]) % 2 != 0 and self.k.split("|")[1]== 'c' :
#            return (-self.m**3*np.sin(self.m*self.vertheta))    #/self.r   
#     def warping_fun_w_2xdiff(self): #GBT w(theta) function  first derivative
#         if self.k.split("|")[0]=='t' or self.k.split("|")[0]=='a' or self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' or  self.k.split("|")[1]== 'u'  :        
#            return (0*np.cos(0))
#         elif int(self.k.split("|")[0])  % 2 == 0 and self.k.split("|")[1]== 'c':
#            return (self.m**4*np.sin(self.m*self.vertheta))
#         elif int(self.k.split("|")[0]) % 2 != 0 and self.k.split("|")[1]== 'c' :
#            return (-self.m**4*np.cos(self.m*self.vertheta)) #/self.r**2
#     def warping_fun_w_3xdiff(self): #GBT w(theta) function  first derivative
#         if self.k.split("|")[0]=='t' or self.k.split("|")[0]=='a' or self.k.split("|")[0]=='1' or  self.k.split("|")[1]== 'v' or  self.k.split("|")[1]== 'u'  :        
#            return (0*np.cos(0))
#         elif int(self.k.split("|")[0])  % 2 == 0 and self.k.split("|")[1]== 'c':
#            return (self.m**5*np.cos(self.m*self.vertheta))
#         elif int(self.k.split("|")[0]) % 2 != 0 and self.k.split("|")[1]== 'c' :
#            return (self.m**5*np.sin(self.m*self.vertheta)) #/self.r**3
#_____________________________________________________________
            
class Hermite: # independent 
    # shape function 
    def __init__ (self,l):
        self.l=l
#        self.atx=None

    def shapefunc_H(self): # Coefficient of hermite shape functions
        return np.matrix([[2/self.l**3, 1/self.l**2,-2/self.l**3,1/self.l**2],[0,-1/(2*self.l),0,1/(2*self.l)],[-3/(2*self.l),-0.25,3/(2*self.l),-0.25],[0.5,self.l/8,0.5,-self.l/8]])

    def shapefunc_L(self): # Coefficient of lagrange shape functions
        L=self.l
        return np.mat([[-0.9e1 / 0.2e1 / L ** 3,0.27e2 / 0.2e1 / L ** 3,-0.27e2 / 0.2e1 / L ** 3,0.9e1 / 0.2e1 / L ** 3],[0.9e1 / 0.4e1 / L ** 2,-0.9e1 / 0.4e1 / L ** 2,-0.9e1 / 0.4e1 / L ** 2,0.9e1 / 0.4e1 / L ** 2],[0.1e1 / L / 8,-0.27e2 / 0.8e1 / L,0.27e2 / 0.8e1 / L,-0.1e1 / L / 8],[-0.1e1 / 0.16e2,0.9e1 / 0.16e2,0.9e1 / 0.16e2,-0.1e1 / 0.16e2]])


    # Elastic stiffness roots
    def ECrootel(self): 
        return np.matrix([[3*self.l**3, 0,0,0],[0,4*self.l,0,0],[0,0,0,0],[0,0,0,0]])
    def GDrootel(self): 
        return np.matrix([[9*self.l**5/80, 0,self.l**3/4,0],[0,self.l**3/3,0,0],[self.l**3/4,0,self.l,0],[0,0,0,0]])
    def Brootel(self): 
        return np.matrix([[self.l**7/448, 0,self.l**5/80,0],[0,self.l**5/80,0,self.l**3/12],[self.l**5/80,0,self.l**3/12,0],[0,self.l**3/12,0,self.l]])

    def GDstress(self):  
        return np.mat([[0.6e1 / 0.5e1 / self.l,0.1e1 / 0.10e2,-0.6e1 / 0.5e1 / self.l,0.1e1 / 0.10e2],[0.1e1 / 0.10e2,0.2e1 / 0.15e2 * self.l,-0.1e1 / 0.10e2,-self.l / 30],[-0.6e1 / 0.5e1 / self.l,-0.1e1 / 0.10e2,0.6e1 / 0.5e1 / self.l,-0.1e1 / 0.10e2],[0.1e1 / 0.10e2,-self.l / 30,-0.1e1 / 0.10e2,0.2e1 / 0.15e2 * self.l]])


    # linear and nonlinear stiffness components

    # iV,x*jV,x*kV,x if i is mode 1 (axial)

    def A_Tx_shHTxTxshH(self): #Sh_H_T_3x2_TxTx_Sh_H
        L=self.l
#        if self.atx is None:
#            self.atx =np.mat([[0.9e1 / 0.70e2 * L,-0.3e1 / 0.280e3 * L ** 2,-0.9e1 / 0.70e2 * L,-0.3e1 / 0.280e3 * L ** 2],[-0.3e1 / 0.280e3 * L ** 2,0.2e1 / 0.35e2 * L ** 3,0.3e1 / 0.280e3 * L ** 2,-L ** 3 / 56],[-0.9e1 / 0.70e2 * L,0.3e1 / 0.280e3 * L ** 2,0.9e1 / 0.70e2 * L,0.3e1 / 0.280e3 * L ** 2],[-0.3e1 / 0.280e3 * L ** 2,-L ** 3 / 56,0.3e1 / 0.280e3 * L ** 2,0.2e1 / 0.35e2 * L ** 3]])
#        return self.atx        
        return np.mat([[0.9e1 / 0.70e2 * L,-0.3e1 / 0.280e3 * L ** 2,-0.9e1 / 0.70e2 * L,-0.3e1 / 0.280e3 * L ** 2],[-0.3e1 / 0.280e3 * L ** 2,0.2e1 / 0.35e2 * L ** 3,0.3e1 / 0.280e3 * L ** 2,-L ** 3 / 56],[-0.9e1 / 0.70e2 * L,0.3e1 / 0.280e3 * L ** 2,0.9e1 / 0.70e2 * L,0.3e1 / 0.280e3 * L ** 2],[-0.3e1 / 0.280e3 * L ** 2,-L ** 3 / 56,0.3e1 / 0.280e3 * L ** 2,0.2e1 / 0.35e2 * L ** 3]])
    def B_Tx_shHTxTxshH(self): #Sh_H_T_2x_TxTx_Sh_H
        L=self.l
        return np.mat([[0,L / 10,0,-L / 10],[L / 10,-L ** 2 / 15,-L / 10,0],[0,-L / 10,0,L / 10],[-L / 10,0,L / 10,L ** 2 / 15]])
    def C_Tx_shHTxTxshH(self): #Sh_H_T_2x_TxTx_Sh_H
        L=self.l
        return np.mat([[0.6e1 / 0.5e1 / L,0.1e1 / 0.10e2,-0.6e1 / 0.5e1 / L,0.1e1 / 0.10e2],[0.1e1 / 0.10e2,0.2e1 / 0.15e2 * L,-0.1e1 / 0.10e2,-L / 30],[-0.6e1 / 0.5e1 / L,-0.1e1 / 0.10e2,0.6e1 / 0.5e1 / L,-0.1e1 / 0.10e2],[0.1e1 / 0.10e2,-L / 30,-0.1e1 / 0.10e2,0.2e1 / 0.15e2 * L]])

    # kV,x*iV,x*jV,x if i is mode 1 (axial)
    def A_Tx_shLTxTxshH(self): #Sh_L_T_3x2_TxTx_Sh_H
        L=self.l
#        if self.atx is None:
#            self.atx =np.mat([[0.9e1 / 0.70e2 * L,-0.3e1 / 0.280e3 * L ** 2,-0.9e1 / 0.70e2 * L,-0.3e1 / 0.280e3 * L ** 2],[-0.3e1 / 0.280e3 * L ** 2,0.2e1 / 0.35e2 * L ** 3,0.3e1 / 0.280e3 * L ** 2,-L ** 3 / 56],[-0.9e1 / 0.70e2 * L,0.3e1 / 0.280e3 * L ** 2,0.9e1 / 0.70e2 * L,0.3e1 / 0.280e3 * L ** 2],[-0.3e1 / 0.280e3 * L ** 2,-L ** 3 / 56,0.3e1 / 0.280e3 * L ** 2,0.2e1 / 0.35e2 * L ** 3]])
#        return self.atx        
        return np.mat([[0.111e3 / 0.560e3 * L,-0.43e2 / 0.140e3 * L ** 2,-0.111e3 / 0.560e3 * L,0.17e2 / 0.560e3 * L ** 2],[-0.81e2 / 0.560e3 * L,0.243e3 / 0.560e3 * L ** 2,0.81e2 / 0.560e3 * L,0.27e2 / 0.280e3 * L ** 2],[0.81e2 / 0.560e3 * L,-0.27e2 / 0.280e3 * L ** 2,-0.81e2 / 0.560e3 * L,-0.243e3 / 0.560e3 * L ** 2],[-0.111e3 / 0.560e3 * L,-0.17e2 / 0.560e3 * L ** 2,0.111e3 / 0.560e3 * L,0.43e2 / 0.140e3 * L ** 2]])

    def B_Tx_shLTxTxshH(self): #Sh_L_T_2x_TxTx_Sh_H
        L=self.l
        return np.mat([[-0.9e1 / 0.20e2,0.7e1 / 0.15e2 * L,0.9e1 / 0.20e2,-L / 6],[0.9e1 / 0.20e2,-0.3e1 / 0.5e1 * L,-0.9e1 / 0.20e2,0.3e1 / 0.10e2 * L],[0.9e1 / 0.20e2,0.3e1 / 0.10e2 * L,-0.9e1 / 0.20e2,-0.3e1 / 0.5e1 * L],[-0.9e1 / 0.20e2,-L / 6,0.9e1 / 0.20e2,0.7e1 / 0.15e2 * L]])

    def C_Tx_shLTxTxshH(self): #Sh_L_T_2x_TxTx_Sh_H
        L=self.l
        return np.mat([[0.11e2 / 0.20e2 / L,-0.3e1 / 0.5e1,-0.11e2 / 0.20e2 / L,0.3e1 / 0.20e2],[0.27e2 / 0.20e2 / L,0.21e2 / 0.20e2,-0.27e2 / 0.20e2 / L,0.3e1 / 0.10e2],[-0.27e2 / 0.20e2 / L,-0.3e1 / 0.10e2,0.27e2 / 0.20e2 / L,-0.21e2 / 0.20e2],[-0.11e2 / 0.20e2 / L,-0.3e1 / 0.20e2,0.11e2 / 0.20e2 / L,0.3e1 / 0.5e1]])



    # iV,xx*jV,x*kV,x if i is mode other than 1 (axial)
    def A_Txx_shHTxTxshH(self): #Sh_H_T_6x_TxTx_Sh_H
        L=self.l
        return np.mat([[0,0.3e1 / 0.10e2 * L,0,-0.3e1 / 0.10e2 * L],[0.3e1 / 0.10e2 * L,-L ** 2 / 5,-0.3e1 / 0.10e2 * L,0],[0,-0.3e1 / 0.10e2 * L,0,0.3e1 / 0.10e2 * L],[-0.3e1 / 0.10e2 * L,0,0.3e1 / 0.10e2 * L,L ** 2 / 5]])
    def B_Txx_shHTxTxshH(self): #Sh_H_T_2_TxTx_Sh_H
        L=self.l
        return np.mat([[0.12e2 / 0.5e1 / L,0.1e1 / 0.5e1,-0.12e2 / 0.5e1 / L,0.1e1 / 0.5e1],[0.1e1 / 0.5e1,0.4e1 / 0.15e2 * L,-0.1e1 / 0.5e1,-L / 15],[-0.12e2 / 0.5e1 / L,-0.1e1 / 0.5e1,0.12e2 / 0.5e1 / L,-0.1e1 / 0.5e1],[0.1e1 / 0.5e1,-L / 15,-0.1e1 / 0.5e1,0.4e1 / 0.15e2 * L]])

    # iV,x*jV,x*kV,xx if i is mode other than 1 (axial)
    def A_Tx_shHTxxTxshH(self): #Sh_H_T_3x2_TxTxx_Sh_H
        L=self.l
        return np.mat([[0,-0.9e1 / 0.20e2 * L,0,0.9e1 / 0.20e2 * L],[0.3e1 / 0.20e2 * L,-0.11e2 / 0.40e2 * L ** 2,-0.3e1 / 0.20e2 * L,0.7e1 / 0.40e2 * L ** 2],[0,0.9e1 / 0.20e2 * L,0,-0.9e1 / 0.20e2 * L],[-0.3e1 / 0.20e2 * L,-0.7e1 / 0.40e2 * L ** 2,0.3e1 / 0.20e2 * L,0.11e2 / 0.40e2 * L ** 2]])
    def B_Tx_shHTxxTxshH(self): #Sh_H_T_2x_TxTxx_Sh_H
        L=self.l
        return np.mat([[-0.6e1 / 0.5e1 / L,0.2e1 / 0.5e1,0.6e1 / 0.5e1 / L,0.2e1 / 0.5e1],[-0.3e1 / 0.5e1,0.11e2 / 0.30e2 * L,0.3e1 / 0.5e1,L / 30],[0.6e1 / 0.5e1 / L,-0.2e1 / 0.5e1,-0.6e1 / 0.5e1 / L,-0.2e1 / 0.5e1],[-0.3e1 / 0.5e1,L / 30,0.3e1 / 0.5e1,0.11e2 / 0.30e2 * L]])
    def C_Tx_shHTxxTxshH(self): #Sh_H_T_1_TxTxx_Sh_H
        L=self.l
        return np.mat([[0,-0.1e1 / L,0,0.1e1 / L],[0.1e1 / L,-0.1e1 / 0.2e1,-0.1e1 / L,0.1e1 / 0.2e1],[0,0.1e1 / L,0,-0.1e1 / L],[-0.1e1 / L,-0.1e1 / 0.2e1,0.1e1 / L,0.1e1 / 0.2e1]]) 

   # iV,*jV,*kV, if i all mode 
    def A_T_shHTTshH(self): #Sh_H_T_x3_TT_Sh_H
        L=self.l
        return np.mat([[-L ** 4 / 70,-L ** 5 / 1008,0,L ** 5 / 10080],[-L ** 5 / 1008,-L ** 6 / 10080,L ** 5 / 10080,0],[0,L ** 5 / 10080,L ** 4 / 70,-L ** 5 / 1008],[L ** 5 / 10080,0,-L ** 5 / 1008,L ** 6 / 10080]])
    def B_T_shHTTshH(self): #Sh_H_T_x2_TT_Sh_H
        L=self.l
        return np.mat([[0.47e2 / 0.1260e4 * L ** 3,L ** 4 / 315,0.11e2 / 0.2520e4 * L ** 3,-L ** 4 / 1008],[L ** 4 / 315,L ** 5 / 2520,L ** 4 / 1008,-L ** 5 / 5040],[0.11e2 / 0.2520e4 * L ** 3,L ** 4 / 1008,0.47e2 / 0.1260e4 * L ** 3,-L ** 4 / 315],[-L ** 4 / 1008,-L ** 5 / 5040,-L ** 4 / 315,L ** 5 / 2520]])
    def C_T_shHTTshH(self): #Sh_H_T_x_TT_Sh_H
        L=self.l
        return np.mat([[-L ** 2 / 10,-L ** 3 / 105,0,L ** 3 / 840],[-L ** 3 / 105,-L ** 4 / 840,L ** 3 / 840,0],[0,L ** 3 / 840,L ** 2 / 10,-L ** 3 / 105],[L ** 3 / 840,0,-L ** 3 / 105,L ** 4 / 840]])
    def D_T_shHTTshH(self): #Sh_H_T_1_TT_Sh_H
        L=self.l
        return np.mat([[0.13e2 / 0.35e2 * L,0.11e2 / 0.210e3 * L ** 2,0.9e1 / 0.70e2 * L,-0.13e2 / 0.420e3 * L ** 2],[0.11e2 / 0.210e3 * L ** 2,L ** 3 / 105,0.13e2 / 0.420e3 * L ** 2,-L ** 3 / 140],[0.9e1 / 0.70e2 * L,0.13e2 / 0.420e3 * L ** 2,0.13e2 / 0.35e2 * L,-0.11e2 / 0.210e3 * L ** 2],[-0.13e2 / 0.420e3 * L ** 2,-L ** 3 / 140,-0.11e2 / 0.210e3 * L ** 2,L ** 3 / 105]])


    # iV,x*jV,x*kV,x*1V,x all except _mode 1
    def AA_TxTx_shHTxTxshH(self): #Sh_H_T_9x4_TxTx_Sh_H
        L=self.l
        return np.mat([[0.9e1 / 0.280e3 * L ** 3,-0.9e1 / 0.1120e4 * L ** 4,-0.9e1 / 0.280e3 * L ** 3,-0.9e1 / 0.1120e4 * L ** 4],[-0.9e1 / 0.1120e4 * L ** 4,0.9e1 / 0.280e3 * L ** 5,0.9e1 / 0.1120e4 * L ** 4,-0.9e1 / 0.1120e4 * L ** 5],[-0.9e1 / 0.280e3 * L ** 3,0.9e1 / 0.1120e4 * L ** 4,0.9e1 / 0.280e3 * L ** 3,0.9e1 / 0.1120e4 * L ** 4],[-0.9e1 / 0.1120e4 * L ** 4,-0.9e1 / 0.1120e4 * L ** 5,0.9e1 / 0.1120e4 * L ** 4,0.9e1 / 0.280e3 * L ** 5]])
    def AB_TxTx_shHTxTxshH(self): #Sh_H_T_6x3_TxTx_Sh_H  ....*2
        L=self.l
        return np.mat([[0,0.9e1 / 0.280e3 * L ** 3,0,-0.9e1 / 0.280e3 * L ** 3],[0.9e1 / 0.280e3 * L ** 3,-0.3e1 / 0.70e2 * L ** 4,-0.9e1 / 0.280e3 * L ** 3,0],[0,-0.9e1 / 0.280e3 * L ** 3,0,0.9e1 / 0.280e3 * L ** 3],[-0.9e1 / 0.280e3 * L ** 3,0,0.9e1 / 0.280e3 * L ** 3,0.3e1 / 0.70e2 * L ** 4]])
    def AC_TxTx_shHTxTxshH(self): #Sh_H_T_3x2_TxTx_Sh_H  ....*2
        L=self.l
        return np.mat([[0.9e1 / 0.70e2 * L,-0.3e1 / 0.280e3 * L ** 2,-0.9e1 / 0.70e2 * L,-0.3e1 / 0.280e3 * L ** 2],[-0.3e1 / 0.280e3 * L ** 2,0.2e1 / 0.35e2 * L ** 3,0.3e1 / 0.280e3 * L ** 2,-L ** 3 / 56],[-0.9e1 / 0.70e2 * L,0.3e1 / 0.280e3 * L ** 2,0.9e1 / 0.70e2 * L,0.3e1 / 0.280e3 * L ** 2],[-0.3e1 / 0.280e3 * L ** 2,-L ** 3 / 56,0.3e1 / 0.280e3 * L ** 2,0.2e1 / 0.35e2 * L ** 3]])
    def BB_TxTx_shHTxTxshH(self): #Sh_H_T_4x2_TxTx_Sh_H
        L=self.l
        return np.mat([[0.6e1 / 0.35e2 * L,-L ** 2 / 70,-0.6e1 / 0.35e2 * L,-L ** 2 / 70],[-L ** 2 / 70,0.8e1 / 0.105e3 * L ** 3,L ** 2 / 70,-L ** 3 / 42],[-0.6e1 / 0.35e2 * L,L ** 2 / 70,0.6e1 / 0.35e2 * L,L ** 2 / 70],[-L ** 2 / 70,-L ** 3 / 42,L ** 2 / 70,0.8e1 / 0.105e3 * L ** 3]])
    def BC_TxTx_shHTxTxshH(self): #Sh_H_T_2x_TxTx_Sh_H   ....*2
        L=self.l
        return np.mat([[0,L / 10,0,-L / 10],[L / 10,-L ** 2 / 15,-L / 10,0],[0,-L / 10,0,L / 10],[-L / 10,0,L / 10,L ** 2 / 15]])
    def CC_TxTx_shHTxTxshH(self): #Sh_H_T_1_TxTx_Sh_H   ....*2
        L=self.l
        return np.mat([[0.6e1 / 0.5e1 / L,0.1e1 / 0.10e2,-0.6e1 / 0.5e1 / L,0.1e1 / 0.10e2],[0.1e1 / 0.10e2,0.2e1 / 0.15e2 * L,-0.1e1 / 0.10e2,-L / 30],[-0.6e1 / 0.5e1 / L,-0.1e1 / 0.10e2,0.6e1 / 0.5e1 / L,-0.1e1 / 0.10e2],[0.1e1 / 0.10e2,-L / 30,-0.1e1 / 0.10e2,0.2e1 / 0.15e2 * L]])

    # iV,*jV,*kV,*1V,
    def AA_TT_shHTTshH(self): #Sh_H_T_x6_TT_Sh_H
        L=self.l
        return np.mat([[0.69e2 / 0.64064e5 * L ** 7,0.31e2 / 0.576576e6 * L ** 8,0.5e1 / 0.128128e6 * L ** 7,-0.19e2 / 0.2306304e7 * L ** 8],[0.31e2 / 0.576576e6 * L ** 8,0.5e1 / 0.1153152e7 * L ** 9,0.19e2 / 0.2306304e7 * L ** 8,-L ** 9 / 768768],[0.5e1 / 0.128128e6 * L ** 7,0.19e2 / 0.2306304e7 * L ** 8,0.69e2 / 0.64064e5 * L ** 7,-0.31e2 / 0.576576e6 * L ** 8],[-0.19e2 / 0.2306304e7 * L ** 8,-L ** 9 / 768768,-0.31e2 / 0.576576e6 * L ** 8,0.5e1 / 0.1153152e7 * L ** 9]])
    def AB_TT_shHTTshH(self): #Sh_H_T_x5_TxTx_Sh_H  ....*2
        L=self.l
        return np.mat([[-0.5e1 / 0.2016e4 * L ** 6,-L ** 7 / 7392,0,L ** 7 / 88704],[-L ** 7 / 7392,-L ** 8 / 88704,L ** 7 / 88704,0],[0,L ** 7 / 88704,0.5e1 / 0.2016e4 * L ** 6,-L ** 7 / 7392],[L ** 7 / 88704,0,-L ** 7 / 7392,L ** 8 / 88704]])
    def AC_TT_shHTTshH(self): #Sh_H_T_x4_TT_Sh_H  ....*2
        L=self.l
        return np.mat([[0.109e3 / 0.18480e5 * L ** 5,0.41e2 / 0.110880e6 * L ** 6,0.13e2 / 0.36960e5 * L ** 5,-0.17e2 / 0.221760e6 * L ** 6],[0.41e2 / 0.110880e6 * L ** 6,L ** 7 / 27720,0.17e2 / 0.221760e6 * L ** 6,-L ** 7 / 73920],[0.13e2 / 0.36960e5 * L ** 5,0.17e2 / 0.221760e6 * L ** 6,0.109e3 / 0.18480e5 * L ** 5,-0.41e2 / 0.110880e6 * L ** 6],[-0.17e2 / 0.221760e6 * L ** 6,-L ** 7 / 73920,-0.41e2 / 0.110880e6 * L ** 6,L ** 7 / 27720]])
    def AD_TT_shHTTshH(self): #Sh_H_T_x3_TT_Sh_H  ....*2
        L=self.l
        return np.mat([[-L ** 4 / 70,-L ** 5 / 1008,0,L ** 5 / 10080],[-L ** 5 / 1008,-L ** 6 / 10080,L ** 5 / 10080,0],[0,L ** 5 / 10080,L ** 4 / 70,-L ** 5 / 1008],[L ** 5 / 10080,0,-L ** 5 / 1008,L ** 6 / 10080]])
    def BB_TT_shHTTshH(self): #Sh_H_T_x4_TT_Sh_H
        L=self.l
        return np.mat([[0.109e3 / 0.18480e5 * L ** 5,0.41e2 / 0.110880e6 * L ** 6,0.13e2 / 0.36960e5 * L ** 5,-0.17e2 / 0.221760e6 * L ** 6],[0.41e2 / 0.110880e6 * L ** 6,L ** 7 / 27720,0.17e2 / 0.221760e6 * L ** 6,-L ** 7 / 73920],[0.13e2 / 0.36960e5 * L ** 5,0.17e2 / 0.221760e6 * L ** 6,0.109e3 / 0.18480e5 * L ** 5,-0.41e2 / 0.110880e6 * L ** 6],[-0.17e2 / 0.221760e6 * L ** 6,-L ** 7 / 73920,-0.41e2 / 0.110880e6 * L ** 6,L ** 7 / 27720]])
    def BC_TT_shHTTshH(self): #Sh_H_T_x3_TT_Sh_H   ....*2
        L=self.l
        return np.mat([[-L ** 4 / 70,-L ** 5 / 1008,0,L ** 5 / 10080],[-L ** 5 / 1008,-L ** 6 / 10080,L ** 5 / 10080,0],[0,L ** 5 / 10080,L ** 4 / 70,-L ** 5 / 1008],[L ** 5 / 10080,0,-L ** 5 / 1008,L ** 6 / 10080]])
    def BD_TT_shHTTshH(self): #Sh_H_T_x2_TT_Sh_H   ....*2
        L=self.l
        return np.mat([[0.47e2 / 0.1260e4 * L ** 3,L ** 4 / 315,0.11e2 / 0.2520e4 * L ** 3,-L ** 4 / 1008],[L ** 4 / 315,L ** 5 / 2520,L ** 4 / 1008,-L ** 5 / 5040],[0.11e2 / 0.2520e4 * L ** 3,L ** 4 / 1008,0.47e2 / 0.1260e4 * L ** 3,-L ** 4 / 315],[-L ** 4 / 1008,-L ** 5 / 5040,-L ** 4 / 315,L ** 5 / 2520]])
    def CC_TT_shHTTshH(self): #Sh_H_T_x2_TT_Sh_H  
        L=self.l
        return np.mat([[0.47e2 / 0.1260e4 * L ** 3,L ** 4 / 315,0.11e2 / 0.2520e4 * L ** 3,-L ** 4 / 1008],[L ** 4 / 315,L ** 5 / 2520,L ** 4 / 1008,-L ** 5 / 5040],[0.11e2 / 0.2520e4 * L ** 3,L ** 4 / 1008,0.47e2 / 0.1260e4 * L ** 3,-L ** 4 / 315],[-L ** 4 / 1008,-L ** 5 / 5040,-L ** 4 / 315,L ** 5 / 2520]])
    def CD_TT_shHTTshH(self): #Sh_H_T_x_TT_Sh_H  ....*2
        L=self.l
        return np.mat([[-L ** 2 / 10,-L ** 3 / 105,0,L ** 3 / 840],[-L ** 3 / 105,-L ** 4 / 840,L ** 3 / 840,0],[0,L ** 3 / 840,L ** 2 / 10,-L ** 3 / 105],[L ** 3 / 840,0,-L ** 3 / 105,L ** 4 / 840]])
    def DD_TT_shHTTshH(self): #Sh_H_T_1_TT_Sh_H  
        L=self.l
        return np.mat([[0.13e2 / 0.35e2 * L,0.11e2 / 0.210e3 * L ** 2,0.9e1 / 0.70e2 * L,-0.13e2 / 0.420e3 * L ** 2],[0.11e2 / 0.210e3 * L ** 2,L ** 3 / 105,0.13e2 / 0.420e3 * L ** 2,-L ** 3 / 140],[0.9e1 / 0.70e2 * L,0.13e2 / 0.420e3 * L ** 2,0.13e2 / 0.35e2 * L,-0.11e2 / 0.210e3 * L ** 2],[-0.13e2 / 0.420e3 * L ** 2,-L ** 3 / 140,-0.11e2 / 0.210e3 * L ** 2,L ** 3 / 105]])


    ## first mode
    ## stress t(shapefunc)*(GDself.rooself.t*shapefunc)
    
    def Axstress(self):
        return (1/(30*self.l))*np.matrix([[36, 3*self.l,-36,3*self.l],[3*self.l,4*self.l**2,-3*self.l,-self.l**2],[-36,-3*self.l,36,-3*self.l],[3*self.l,-self.l**2,-3*self.l,4*self.l**2]])
    def ASroot(self):
        return np.matrix([[0, (9*self.l**5)/20,0,0],[(9*self.l**5)/20,0,self.l**3,0],[0,self.l**3,0,0],[0,0,0,0]])
    def BSroot(self):
        return np.matrix([[(9*self.l**5)/40,0,self.l**3/2,0],[0,2*self.l**3/3,0,0],[self.l**3/2,0,2*self.l,0],[0,0,0,0]])
    # stress t(shapefunc)*(ASself.rooself.t*shapefunc)
    def AStress(self):
        return np.matrix([[0, 3*self.l/10,0,-3*self.l/10],[3*self.l/10,-self.l**2/5,-3*self.l/10,0],[0,-3*self.l/10,0,3*self.l/10],[-3*self.l/10,0,3*self.l/10,self.l**2/5]])
    def BStress(self):
        return np.matrix([[12/(self.l*5), 0.2,-12/(self.l*5),0.2],[0.2,4*self.l/15,-0.2,-self.l/15],[-12/(self.l*5), -0.2, 12/(self.l*5),-0.2],[0.2,-self.l/15,-0.2,4*self.l/15]])
    #internal force
    def AStressf(self):
        return np.matrix([[0,0.3e1 / 0.20e2 * self.l,0,-0.3e1 / 0.20e2 * self.l],[-0.9e1 / 0.20e2 * self.l,-0.11e2 / 0.40e2 * self.l ** 2,0.9e1 / 0.20e2 * self.l,-0.7e1 / 0.40e2 * self.l ** 2],[0,-0.3e1 / 0.20e2 * self.l,0,0.3e1 / 0.20e2 * self.l],[0.9e1 / 0.20e2 * self.l,0.7e1 / 0.40e2 * self.l ** 2,-0.9e1 / 0.20e2 * self.l,0.11e2 / 0.40e2 * self.l ** 2]])
    def BStressf(self):
        return np.matrix([[-0.6e1 / 0.5e1 / self.l,-0.3e1 / 0.5e1,0.6e1 / 0.5e1 / self.l,-0.3e1 / 0.5e1],[0.2e1 / 0.5e1,0.11e2 / 0.30e2 * self.l,-0.2e1 / 0.5e1,self.l / 30],[0.6e1 / 0.5e1 / self.l,0.3e1 / 0.5e1,-0.6e1 / 0.5e1 / self.l,0.3e1 / 0.5e1],[0.2e1 / 0.5e1,self.l / 30,-0.2e1 / 0.5e1,0.11e2 / 0.30e2 * self.l]])
    def CStressf(self):
        return np.matrix([[0.6e1 / 0.5e1 / self.l,0.1e1 / 0.10e2,-0.6e1 / 0.5e1 / self.l,0.1e1 / 0.10e2],[0.1e1 / 0.10e2,0.2e1 / 0.15e2 * self.l,-0.1e1 / 0.10e2,-self.l / 30],[-0.6e1 / 0.5e1 / self.l,-0.1e1 / 0.10e2,0.6e1 / 0.5e1 / self.l,-0.1e1 / 0.10e2],[0.1e1 / 0.10e2,-self.l / 30,-0.1e1 / 0.10e2,0.2e1 / 0.15e2 * self.l]])

   
   
    # initial deformation tangent stiffness matrix Ax3+Bx2+Cx+D
    
    def ADroot(self):
        return np.matrix([[0, 9*self.l**5/20,0,0],[9*self.l**5/20,0,self.l**3/2,0],[0,0,0,0],[0,0,0,0]])
    def BDroot(self):
        return np.matrix([[9*self.l**5/20,0,self.l**3,0],[0,0,2*self.l**3/3,0],[0,0,0,0],[0,0,0,0]])
    def CDroot(self):
        return np.matrix([[0,self.l**3,0,0],[self.l**3/2,0,2*self.l,0],[0,0,0,0],[0,0,0,0]])
    # disp t(shapefunc)*(ADroot*shapefunc)    
    def ADisp(self):
        return np.matrix([[0, -9*self.l/20,0,9*self.l/20],[3*self.l/20,-11*self.l**2/40,-3*self.l/20,7*self.l**2/40],[0, 9*self.l/20,0,-9*self.l/20],[-3*self.l/20,-7*self.l**2/40,3*self.l/20,11*self.l**2/40]])
#cg = numpy.mat([[0,-0.9e1 / 0.20e2 * L,0,0.9e1 / 0.20e2 * L],[0.3e1 / 0.20e2 * L,-0.11e2 / 0.40e2 * L ** 2,-0.3e1 / 0.20e2 * L,0.7e1 / 0.40e2 * L ** 2],[0,0.9e1 / 0.20e2 * L,0,-0.9e1 / 0.20e2 * L],[-0.3e1 / 0.20e2 * L,-0.7e1 / 0.40e2 * L ** 2,0.3e1 / 0.20e2 * L,0.11e2 / 0.40e2 * L ** 2]])


    def BDisp(self):
        return np.matrix([[-6/(5*self.l),0.4,6/(5*self.l),0.4],[-0.6,11*self.l/30,0.6,self.l/30],[6/(5*self.l),-0.4,-6/(5*self.l),-0.4],[-0.6,self.l/30,0.6,11*self.l/30]])
#cg = numpy.mat([[-0.6e1 / 0.5e1 / L,0.2e1 / 0.5e1,0.6e1 / 0.5e1 / L,0.2e1 / 0.5e1],[-0.3e1 / 0.5e1,0.11e2 / 0.30e2 * L,0.3e1 / 0.5e1,L / 30],[0.6e1 / 0.5e1 / L,-0.2e1 / 0.5e1,-0.6e1 / 0.5e1 / L,-0.2e1 / 0.5e1],[-0.3e1 / 0.5e1,L / 30,0.3e1 / 0.5e1,0.11e2 / 0.30e2 * L]])


    def CDisp(self):
        return np.matrix([[0,-1/self.l,0,1/self.l],[1/self.l,-0.5,-1/self.l,0.5],[0,1/self.l,0,-1/self.l],[-1/self.l,-0.5,1/self.l,0.5]])
    # shape funcself.tion Axiaself.l fiself.rsself.t self.mode onself.ly in dispself.laceself.menself.t 
#g0 = numpy.mat([[0,-0.1e1 / L,0,0.1e1 / L],[0.1e1 / L,-0.1e1 / 0.2e1,-0.1e1 / L,0.1e1 / 0.2e1],[0,0.1e1 / L,0,-0.1e1 / L],[-0.1e1 / L,-0.1e1 / 0.2e1,0.1e1 / L,0.1e1 / 0.2e1]])    



    def ADisp_ux(self):
        L=self.l
        return np.mat([[0,-6 / L,0,6 / L],[-6 / L,-6,6 / L,0],[0,6 / L,0,-6 / L],[6 / L,0,-6 / L,6]])
    def BDisp_ux(self):
        L=self.l
        return np.mat([[24 / L ** 3,12 / L ** 2,-24 / L ** 3,12 / L ** 2],[12 / L ** 2,8 / L,-12 / L ** 2,4 / L],[-24 / L ** 3,-12 / L ** 2,24 / L ** 3,-12 / L ** 2],[12 / L ** 2,4 / L,-12 / L ** 2,8 / L]])





    def shapefuncAxi(self):
        return np.matrix([[-1/self.l,1/self.l],[1/2,1/2]])

    def AXDroot(self):
        return np.matrix([[9*self.l**5/80,0,self.l**3/4,0],[0,0,0,0]])
    def BXDroot(self):
        return np.matrix([[0,self.l**3/3,0,0],[0,0,0,0]])
    def CXDroot(self):
        return np.matrix([[self.l**3/4,0,self.l,0],[0,0,0,0]])

    def AXDdisp(self):
        return np.matrix([[3*self.l/20,-self.l**2/20,-3*self.l/20,-self.l**2/20],[-3*self.l/20,self.l**2/20,3*self.l/20,self.l**2/20]])
    def BXDdisp(self):
        return np.matrix([[0,self.l/6,0,-self.l/6],[0,-self.l/6,0,self.l/6]])
    def CXDdisp(self):
        return np.matrix([[1/self.l,0,-1/self.l,0],[-1/self.l,0,1/self.l,0]])

    def AXDdisp_ux(self):
        L=self.l
        return np.mat([[-6 / L,-3,6 / L,-3],[6 / L,3,-6 / L,3]])
    def BXDdisp_ux(self):
        L=self.l
        return np.mat([[0,2 / L,0,-2 / L],[0,-2 / L,0,2 / L]])



    def Aaxisymmetric(self):
        return np.matrix([[0,0],[self.l / 2,-self.l / 2],[0,0],[-self.l / 2,self.l / 2]])
    def Baxisymmetric(self):
        return np.matrix([[2 / self.l,-2 / self.l],[0,0],[-2 / self.l,2 / self.l],[0,0]])
 
    def Aaxisymmetric_s(self):
        L=float(self.l)
        return np.mat([[-L ** 2 / 10,-L ** 3 / 105,0,L ** 3 / 840],[-L ** 3 / 105,-L ** 4 / 840,L ** 3 / 840,0],[0,L ** 3 / 840,L ** 2 / 10,-L ** 3 / 105],[L ** 3 / 840,0,-L ** 3 / 105,L ** 4 / 840]])
    def Baxisymmetric_s(self):
        L=self.l
        return np.mat([[0.13e2 / 0.35e2 * L,0.11e2 / 0.210e3 * L ** 2,0.9e1 / 0.70e2 * L,-0.13e2 / 0.420e3 * L ** 2],[0.11e2 / 0.210e3 * L ** 2,L ** 3 / 105,0.13e2 / 0.420e3 * L ** 2,-L ** 3 / 140],[0.9e1 / 0.70e2 * L,0.13e2 / 0.420e3 * L ** 2,0.13e2 / 0.35e2 * L,-0.11e2 / 0.210e3 * L ** 2],[-0.13e2 / 0.420e3 * L ** 2,-L ** 3 / 140,-0.11e2 / 0.210e3 * L ** 2,L ** 3 / 105]])

        
    def Astress_comp(self):
        L=self.l
        return np.mat([[-L ** 4 / 70,-L ** 5 / 1008,0,L ** 5 / 10080],[-L ** 5 / 1008,-L ** 6 / 10080,L ** 5 / 10080,0],[0,L ** 5 / 10080,L ** 4 / 70,-L ** 5 / 1008],[L ** 5 / 10080,0,-L ** 5 / 1008,L ** 6 / 10080]])
    def Bstress_comp(self):
        L=self.l
        return np.mat([[0.47e2 / 0.1260e4 * L ** 3,L ** 4 / 315,0.11e2 / 0.2520e4 * L ** 3,-L ** 4 / 1008],[L ** 4 / 315,L ** 5 / 2520,L ** 4 / 1008,-L ** 5 / 5040],[0.11e2 / 0.2520e4 * L ** 3,L ** 4 / 1008,0.47e2 / 0.1260e4 * L ** 3,-L ** 4 / 315],[-L ** 4 / 1008,-L ** 5 / 5040,-L ** 4 / 315,L ** 5 / 2520]])
    def Cstress_comp(self):
        L=self.l
        return np.mat([[-L ** 2 / 10,-L ** 3 / 105,0,L ** 3 / 840],[-L ** 3 / 105,-L ** 4 / 840,L ** 3 / 840,0],[0,L ** 3 / 840,L ** 2 / 10,-L ** 3 / 105],[L ** 3 / 840,0,-L ** 3 / 105,L ** 4 / 840]])
    def Dstress_comp(self):
        L=self.l
        return np.mat([[0.13e2 / 0.35e2 * L,0.11e2 / 0.210e3 * L ** 2,0.9e1 / 0.70e2 * L,-0.13e2 / 0.420e3 * L ** 2],[0.11e2 / 0.210e3 * L ** 2,L ** 3 / 105,0.13e2 / 0.420e3 * L ** 2,-L ** 3 / 140],[0.9e1 / 0.70e2 * L,0.13e2 / 0.420e3 * L ** 2,0.13e2 / 0.35e2 * L,-0.11e2 / 0.210e3 * L ** 2],[-0.13e2 / 0.420e3 * L ** 2,-L ** 3 / 140,-0.11e2 / 0.210e3 * L ** 2,L ** 3 / 105]])




    def Aaxisymmetric_us(self):
        L=self.l
        return np.mat([[0,L / 20,0,-L / 20],[L / 20,-L ** 2 / 30,-L / 20,0],[0,-L / 20,0,L / 20],[-L / 20,0,L / 20,L ** 2 / 30]])
    def Baxisymmetric_us(self):
        L=self.l
        return np.mat([[0.6e1 / 0.5e1 / L,0.1e1 / 0.10e2,-0.6e1 / 0.5e1 / L,0.1e1 / 0.10e2],[0.1e1 / 0.10e2,0.2e1 / 0.15e2 * L,-0.1e1 / 0.10e2,-L / 30],[-0.6e1 / 0.5e1 / L,-0.1e1 / 0.10e2,0.6e1 / 0.5e1 / L,-0.1e1 / 0.10e2],[0.1e1 / 0.10e2,-L / 30,-0.1e1 / 0.10e2,0.2e1 / 0.15e2 * L]])

    def Astress_comp_us(self):
        L=self.l
        return np.mat([[0,0.3e1 / 0.560e3 * L ** 3,0,-0.3e1 / 0.560e3 * L ** 3],[0.3e1 / 0.560e3 * L ** 3,-L ** 4 / 140,-0.3e1 / 0.560e3 * L ** 3,0],[0,-0.3e1 / 0.560e3 * L ** 3,0,0.3e1 / 0.560e3 * L ** 3],[-0.3e1 / 0.560e3 * L ** 3,0,0.3e1 / 0.560e3 * L ** 3,L ** 4 / 140]])
    def Bstress_comp_us(self):
        L=self.l
        return np.mat([[0.3e1 / 0.70e2 * L,-L ** 2 / 280,-0.3e1 / 0.70e2 * L,-L ** 2 / 280],[-L ** 2 / 280,0.2e1 / 0.105e3 * L ** 3,L ** 2 / 280,-L ** 3 / 168],[-0.3e1 / 0.70e2 * L,L ** 2 / 280,0.3e1 / 0.70e2 * L,L ** 2 / 280],[-L ** 2 / 280,-L ** 3 / 168,L ** 2 / 280,0.2e1 / 0.105e3 * L ** 3]])
    def Cstress_comp_us(self):
        L=self.l
        return np.mat([[0,L / 20,0,-L / 20],[L / 20,-L ** 2 / 30,-L / 20,0],[0,-L / 20,0,L / 20],[-L / 20,0,L / 20,L ** 2 / 30]])
    def Dstress_comp_us(self):
        L=self.l
        return np.mat([[0.6e1 / 0.5e1 / L,0.1e1 / 0.10e2,-0.6e1 / 0.5e1 / L,0.1e1 / 0.10e2],[0.1e1 / 0.10e2,0.2e1 / 0.15e2 * L,-0.1e1 / 0.10e2,-L / 30],[-0.6e1 / 0.5e1 / L,-0.1e1 / 0.10e2,0.6e1 / 0.5e1 / L,-0.1e1 / 0.10e2],[0.1e1 / 0.10e2,-L / 30,-0.1e1 / 0.10e2,0.2e1 / 0.15e2 * L]])




    def Aaxisymmetric_sa(self):
        L=self.l
        return np.mat([[-L ** 2 / 12,0],[0,L ** 2 / 12]])
        #return np.mat([[0.1e1 / L,-0.1e1 / L],[-0.1e1 / L,0.1e1 / L]])
    def Baxisymmetric_sa(self):
        L=self.l
        return np.mat([[L / 3,L / 6],[L / 6,L / 3]])





   
    def Axisymmetric(self):
        return np.matrix([[0.1e1 / self.l,-0.1e1 / self.l],[-0.1e1 / self.l,0.1e1 / self.l]])


    def Axisymmetric_ux(self):

        return np.matrix([[0.1e1 / self.l,-0.1e1 / self.l],[-0.1e1 / self.l,0.1e1 / self.l]])

    def AAxisymmetric_ux(self):
        L=self.l
        return np.mat([[12 / L ** 3,6 / L ** 2,-12 / L ** 3,6 / L ** 2],[6 / L ** 2,4 / L,-6 / L ** 2,2 / L],[-12 / L ** 3,-6 / L ** 2,12 / L ** 3,-6 / L ** 2],[6 / L ** 2,2 / L,-6 / L ** 2,4 / L]])
    def Astress_ux(self):
        L=self.l
        return np.mat([[0,-6 / L,0,6 / L],[-6 / L,-6,6 / L,0],[0,6 / L,0,-6 / L],[6 / L,0,-6 / L,6]])
    def Bstress_ux(self):
        L=self.l
        return np.mat([[24 / L ** 3,12 / L ** 2,-24 / L ** 3,12 / L ** 2],[12 / L ** 2,8 / L,-12 / L ** 2,4 / L],[-24 / L ** 3,-12 / L ** 2,24 / L ** 3,-12 / L ** 2],[12 / L ** 2,4 / L,-12 / L ** 2,8 / L]])



   
    def DAxisymmetric(self):
        L=self.l
        return np.matrix([[0,0.1e1 / L,0,-0.1e1 / L],[-0.1e1 / L,-0.1e1 / 0.2e1,0.1e1 / L,-0.1e1 / 0.2e1],[0,-0.1e1 / L,0,0.1e1 / L],[0.1e1 / L,0.1e1 / 0.2e1,-0.1e1 / L,0.1e1 / 0.2e1]])         
       
    def DAaxisymmetric(self):
        L=self.l
        return np.matrix([[0,L / 4.0,0,-L / 4.0],[0,-L / 4.0,0,L / 4.0]])
    def DBaxisymmetric(self):
        L=self.l
        return np.matrix([[-2.0 / L,-1.0,2.0 / L,-1.0],[2.0 / L,1.0,-2.0 / L,1.0]])
    def DCaxisymmetric(self):
        L=self.l
        return np.matrix([[0,0.1e1 / L,0,-0.1e1 / L],[0,-0.1e1 / L,0,0.1e1 / L]])


    def DAaxisymmetric_s(self):
        L=self.l
        return np.mat([[-0.3e1 / 0.224e3 * L ** 4,-L ** 5 / 1120,L ** 4 / 1120,0],[-L ** 4 / 1120,0,0.3e1 / 0.224e3 * L ** 4,-L ** 5 / 1120]])
        #return np.mat([[L ** 3 / 70,L ** 4 / 1120,-L ** 3 / 70,L ** 4 / 1120],[-L ** 3 / 70,-L ** 4 / 1120,L ** 3 / 70,-L ** 4 / 1120]])
    def DBaxisymmetric_s(self):
        L=self.l
        return np.mat([[0.59e2 / 0.1680e4 * L ** 3,L ** 4 / 336,0.11e2 / 0.1680e4 * L ** 3,-L ** 4 / 840],[0.11e2 / 0.1680e4 * L ** 3,L ** 4 / 840,0.59e2 / 0.1680e4 * L ** 3,-L ** 4 / 336]])
        #return np.mat([[-L ** 2 / 24,-L ** 3 / 240,-L ** 2 / 24,L ** 3 / 240],[L ** 2 / 24,L ** 3 / 240,L ** 2 / 24,-L ** 3 / 240]])
    def DCaxisymmetric_s(self):
        L=self.l
        return np.mat([[-0.11e2 / 0.120e3 * L ** 2,-L ** 3 / 120,L ** 2 / 120,0],[-L ** 2 / 120,0,0.11e2 / 0.120e3 * L ** 2,-L ** 3 / 120]])
        #return np.mat([[L / 10,L ** 2 / 120,-L / 10,L ** 2 / 120],[-L / 10,-L ** 2 / 120,L / 10,-L ** 2 / 120]])
        
    def DDaxisymmetric_s(self):
        L=self.l
        return np.mat([[0.7e1 / 0.20e2 * L,L ** 2 / 20,0.3e1 / 0.20e2 * L,-L ** 2 / 30],[0.3e1 / 0.20e2 * L,L ** 2 / 30,0.7e1 / 0.20e2 * L,-L ** 2 / 20]])
        #return np.mat([[-0.1e1 / 0.2e1,-L / 12,-0.1e1 / 0.2e1,L / 12],[0.1e1 / 0.2e1,L / 12,0.1e1 / 0.2e1,-L / 12]])

    def Aaxisymmetric_comp(self):
        L=self.l
        return np.mat([[-0.3e1 / 0.224e3 * L ** 4,-L ** 4 / 1120],[-L ** 5 / 1120,0],[L ** 4 / 1120,0.3e1 / 0.224e3 * L ** 4],[0,-L ** 5 / 1120]])
        #return np.transpose(np.mat([[L ** 3 / 70,L ** 4 / 1120,-L ** 3 / 70,L ** 4 / 1120],[-L ** 3 / 70,-L ** 4 / 1120,L ** 3 / 70,-L ** 4 / 1120]]))

    def Baxisymmetric_comp(self):
        L=self.l
        return np.mat([[0.59e2 / 0.1680e4 * L ** 3,0.11e2 / 0.1680e4 * L ** 3],[L ** 4 / 336,L ** 4 / 840],[0.11e2 / 0.1680e4 * L ** 3,0.59e2 / 0.1680e4 * L ** 3],[-L ** 4 / 840,-L ** 4 / 336]])
        #return np.transpose(np.mat([[-L ** 2 / 24,-L ** 3 / 240,-L ** 2 / 24,L ** 3 / 240],[L ** 2 / 24,L ** 3 / 240,L ** 2 / 24,-L ** 3 / 240]]))

    def Caxisymmetric_comp(self):
        L=self.l
        return np.mat([[-0.11e2 / 0.120e3 * L ** 2,-L ** 2 / 120],[-L ** 3 / 120,0],[L ** 2 / 120,0.11e2 / 0.120e3 * L ** 2],[0,-L ** 3 / 120]])
        #return np.transpose(np.mat([[L / 10,L ** 2 / 120,-L / 10,L ** 2 / 120],[-L / 10,-L ** 2 / 120,L / 10,-L ** 2 / 120]]))

    def Daxisymmetric_comp(self):
        L=self.l
        return np.mat([[0.7e1 / 0.20e2 * L,0.3e1 / 0.20e2 * L],[L ** 2 / 20,L ** 2 / 30],[0.3e1 / 0.20e2 * L,0.7e1 / 0.20e2 * L],[-L ** 2 / 30,-L ** 2 / 20]])
        #return np.transpose(np.mat([[-0.1e1 / 0.2e1,-L / 12,-0.1e1 / 0.2e1,L / 12],[0.1e1 / 0.2e1,L / 12,0.1e1 / 0.2e1,-L / 12]]))






    def DAaxisymmetric_xs(self):
        L=self.l
        return np.mat([[-L ** 2 / 8,L ** 2 / 8],[-L ** 3 / 80,L ** 3 / 80],[-L ** 2 / 8,L ** 2 / 8],[L ** 3 / 80,-L ** 3 / 80]])
    def DBaxisymmetric_xs(self):
        L=self.l
        return np.mat([[L / 5,-L / 5],[L ** 2 / 60,-L ** 2 / 60],[-L / 5,L / 5],[L ** 2 / 60,-L ** 2 / 60]])
    def DCaxisymmetric_xs(self):
        L=self.l
        return np.mat([[-0.1e1 / 0.2e1,0.1e1 / 0.2e1],[-L / 12,L / 12],[-0.1e1 / 0.2e1,0.1e1 / 0.2e1],[L / 12,-L / 12]])

    def DAaxisymmetric_xsd(self):
        L=self.l
        return np.mat([[0,0],[L ** 3 / 80,-L ** 3 / 80],[0,0],[-L ** 3 / 80,L ** 3 / 80]])
    def DBaxisymmetric_xsd(self):
        L=self.l
        return np.mat([[L / 20,-L / 20],[-L ** 2 / 60,L ** 2 / 60],[-L / 20,L / 20],[-L ** 2 / 60,L ** 2 / 60]])
    def DCaxisymmetric_xsd(self):
        L=self.l
        return np.mat([[0,0],[L / 12,-L / 12],[0,0],[-L / 12,L / 12]])
    def DDaxisymmetric_xsd(self):
        L=self.l
        return np.mat([[0.1e1 / L,-0.1e1 / L],[0,0],[-0.1e1 / L,0.1e1 / L],[0,0]])


    def DAaxisymmetric_us(self):
        L=self.l
        return np.mat([[-0.3e1 / 0.40e2 * L ** 2,L ** 3 / 16,0.3e1 / 0.40e2 * L ** 2,-L ** 3 / 80],[-0.3e1 / 0.40e2 * L ** 2,-L ** 3 / 80,0.3e1 / 0.40e2 * L ** 2,L ** 3 / 16]])
    def DBaxisymmetric_us(self):
        L=self.l
        return np.mat([[L / 10,-0.7e1 / 0.60e2 * L ** 2,-L / 10,L ** 2 / 20],[-L / 10,-L ** 2 / 20,L / 10,0.7e1 / 0.60e2 * L ** 2]])
    def DCaxisymmetric_us(self):
        L=self.l
        return np.mat([[-0.1e1 / 0.2e1,L / 12,0.1e1 / 0.2e1,-L / 12],[-0.1e1 / 0.2e1,-L / 12,0.1e1 / 0.2e1,L / 12]])  
 
    def DAdisp_comp_us(self):
        L=self.l
        return np.mat([[-0.3e1 / 0.40e2 * L ** 2,0.19e2 / 0.280e3 * L ** 3,0.3e1 / 0.40e2 * L ** 2,-L ** 3 / 56],[-0.3e1 / 0.280e3 * L ** 3,L ** 4 / 280,0.3e1 / 0.280e3 * L ** 3,-L ** 4 / 560],[-0.3e1 / 0.40e2 * L ** 2,-L ** 3 / 56,0.3e1 / 0.40e2 * L ** 2,0.19e2 / 0.280e3 * L ** 3],[0.3e1 / 0.280e3 * L ** 3,L ** 4 / 560,-0.3e1 / 0.280e3 * L ** 3,-L ** 4 / 280]])
    def DBdisp_comp_us(self):
        L=self.l
        return np.mat([[0.9e1 / 0.70e2 * L,-0.5e1 / 0.42e2 * L ** 2,-0.9e1 / 0.70e2 * L,L ** 2 / 21],[L ** 2 / 70,-L ** 3 / 105,-L ** 2 / 70,L ** 3 / 140],[-0.9e1 / 0.70e2 * L,-L ** 2 / 21,0.9e1 / 0.70e2 * L,0.5e1 / 0.42e2 * L ** 2],[L ** 2 / 70,L ** 3 / 140,-L ** 2 / 70,-L ** 3 / 105]])
    def DCdisp_comp_us(self):
        L=self.l
        return np.mat([[-0.1e1 / 0.2e1,L / 10,0.1e1 / 0.2e1,-L / 10],[-L / 10,0,L / 10,-L ** 2 / 60],[-0.1e1 / 0.2e1,-L / 10,0.1e1 / 0.2e1,L / 10],[L / 10,L ** 2 / 60,-L / 10,0]])


    def Aaxisymmetric_cs(self):
        L=self.l
        return np.mat([[-L ** 4 / 70,-L ** 5 / 1008,0,L ** 5 / 10080],[-L ** 5 / 1008,-L ** 6 / 10080,L ** 5 / 10080,0],[0,L ** 5 / 10080,L ** 4 / 70,-L ** 5 / 1008],[L ** 5 / 10080,0,-L ** 5 / 1008,L ** 6 / 10080]])
    def Baxisymmetric_cs(self):
        L=self.l
        return np.mat([[0.47e2 / 0.1260e4 * L ** 3,L ** 4 / 315,0.11e2 / 0.2520e4 * L ** 3,-L ** 4 / 1008],[L ** 4 / 315,L ** 5 / 2520,L ** 4 / 1008,-L ** 5 / 5040],[0.11e2 / 0.2520e4 * L ** 3,L ** 4 / 1008,0.47e2 / 0.1260e4 * L ** 3,-L ** 4 / 315],[-L ** 4 / 1008,-L ** 5 / 5040,-L ** 4 / 315,L ** 5 / 2520]])
    def Caxisymmetric_cs(self):
        L=self.l
        return np.mat([[-L ** 2 / 10,-L ** 3 / 105,0,L ** 3 / 840],[-L ** 3 / 105,-L ** 4 / 840,L ** 3 / 840,0],[0,L ** 3 / 840,L ** 2 / 10,-L ** 3 / 105],[L ** 3 / 840,0,-L ** 3 / 105,L ** 4 / 840]])
    def Daxisymmetric_cs(self):
        L=self.l
        return np.mat([[0.13e2 / 0.35e2 * L,0.11e2 / 0.210e3 * L ** 2,0.9e1 / 0.70e2 * L,-0.13e2 / 0.420e3 * L ** 2],[0.11e2 / 0.210e3 * L ** 2,L ** 3 / 105,0.13e2 / 0.420e3 * L ** 2,-L ** 3 / 140],[0.9e1 / 0.70e2 * L,0.13e2 / 0.420e3 * L ** 2,0.13e2 / 0.35e2 * L,-0.11e2 / 0.210e3 * L ** 2],[-0.13e2 / 0.420e3 * L ** 2,-L ** 3 / 140,-0.11e2 / 0.210e3 * L ** 2,L ** 3 / 105]])


       
#shear        
    def AAxisymmetric_sx(self):
        L=self.l
        return np.mat([[-0.1e1 / 0.2e1,L / 10,0.1e1 / 0.2e1,-L / 10],[-L / 10,0,L / 10,-L ** 2 / 60],[-0.1e1 / 0.2e1,-L / 10,0.1e1 / 0.2e1,L / 10],[L / 10,L ** 2 / 60,-L / 10,0]])
        
        
#_____________________________________________________________

class linear_curved_GBT_ik_coupling_matrix:    
    def __init__ (self,k1,k2,R,t,r,xi,mu):
        self.r=r
        self.t=t
        self.R=R
        self.mu=mu        
        self.vertheta=Symbol('vertheta')
        self.i=GBT_func_curve(k1,r,self.vertheta,R)     
        self.k=GBT_func_curve(k2,r,self.vertheta,R)
        self.xi=(1+(xi*(r/R)*cos(self.vertheta)))
    # 
    def C1_ik(self):

        func = lambdify((self.vertheta),self.xi*( (self.t/((self.R*self.xi)**2*(1-self.mu**2)))*self.i.warping_fun_u() *self.k.warping_fun_u()+\
                                         self.t**3/((self.R*self.xi)**4*12*(1-self.mu**2))*(self.k.warping_fun_u()*cos(self.vertheta)-self.k.warping_fun_w())*\
                                          (self.i.warping_fun_u()*cos(self.vertheta)-self.i.warping_fun_w()) ), 'numpy')       
        C_1_ik=(quad(lambda vertheta:func(vertheta),0,math.pi*2)[0])*self.r
        return C_1_ik
    def C2_ik(self):

        func = lambdify((self.vertheta),self.xi*( (self.t/((self.R*self.xi)**2*(1-self.mu**2)))*(self.i.warping_fun_w()*cos(self.vertheta)-self.i.warping_fun_v()*sin(self.vertheta)) *self.k.warping_fun_u()+\
                                         sin(self.vertheta)*self.t**3/((self.R*self.xi)**3*self.r*12*(1-self.mu**2))*(self.k.warping_fun_u()*cos(self.vertheta)-self.k.warping_fun_w())*\
                                          (self.i.warping_fun_w_1xdiff()-self.i.warping_fun_v())), 'numpy')       
        C_2_ik=(quad(lambda vertheta:func(vertheta),0,math.pi*2)[0])*self.r
        return C_2_ik
    def C3_ik(self):

        func = lambdify((self.vertheta),self.xi*( (self.t/((self.R*self.xi)**2*(1-self.mu**2)))*(self.k.warping_fun_w()*cos(self.vertheta)-self.k.warping_fun_v()*sin(self.vertheta)) *self.i.warping_fun_u()+\
                                         sin(self.vertheta)*self.t**3/((self.R*self.xi)**3*self.r*12*(1-self.mu**2))*(self.i.warping_fun_u()*cos(self.vertheta)-self.i.warping_fun_w())*\
                                          (self.k.warping_fun_w_1xdiff()-self.k.warping_fun_v())), 'numpy')       
        C_3_ik=(quad(lambda vertheta:func(vertheta),0,math.pi*2)[0])*self.r
        return C_3_ik
    def C4_ik(self):

        func = lambdify((self.vertheta),self.xi*( (self.t/((self.R*self.xi)**2*(1-self.mu**2)))*(self.k.warping_fun_w()*cos(self.vertheta)-self.k.warping_fun_v()*sin(self.vertheta)) *(self.i.warping_fun_w()*cos(self.vertheta)-self.i.warping_fun_v()*sin(self.vertheta))+\
                                         sin(self.vertheta)*sin(self.vertheta)*self.t**3/((self.R*self.xi*self.r)**2*12*(1-self.mu**2))*(self.i.warping_fun_w_1xdiff()-self.i.warping_fun_v())*\
                                          (self.k.warping_fun_w_1xdiff()-self.k.warping_fun_v())), 'numpy')       
        C_4_ik=(quad(lambda vertheta:func(vertheta),0,math.pi*2)[0])*self.r
        return C_4_ik
    def B_ik(self):

        func = lambdify((self.vertheta),self.xi*( (self.t/((self.r)**2*(1-self.mu**2)))*(self.i.warping_fun_v_1xdiff()+self.i.warping_fun_w()) *(self.k.warping_fun_v_1xdiff()+self.k.warping_fun_w())+\
                                         self.t**3/((self.r)**4*12*(1-self.mu**2))*(-self.i.warping_fun_w_2xdiff()+self.i.warping_fun_v_1xdiff())*\
                                          (-self.k.warping_fun_w_2xdiff()+self.k.warping_fun_v_1xdiff())), 'numpy')       
        B_ik=(quad(lambda vertheta:func(vertheta),0,math.pi*2)[0])*self.r
        return B_ik
    def D_ik(self):

        func = lambdify((self.vertheta),self.xi*( self.t*(self.k.warping_fun_u_1xdiff()/self.r + self.k.warping_fun_u()*sin(self.vertheta)/(self.R*self.xi) + self.k.warping_fun_v()/(self.R*self.xi)) *\
                                                (self.i.warping_fun_u_1xdiff()/self.r + self.i.warping_fun_u()*sin(self.vertheta)/(self.R*self.xi) + self.i.warping_fun_v()/(self.R*self.xi))+\
                                          (self.t**3/12)*2*(-self.i.warping_fun_w_1xdiff()/(self.R*self.r*self.xi)-self.i.warping_fun_w()*sin(self.vertheta)/(self.R*self.xi)**2+\
                                                          self.i.warping_fun_v()*(1+2*self.xi)/(4*self.R*self.r*self.xi**2)+self.i.warping_fun_u_1xdiff()*cos(self.vertheta)/(2*self.R*self.r*self.xi)-\
                                                              self.i.warping_fun_u_1xdiff()/(4*self.r*self.r*self.xi) + self.i.warping_fun_u()*(2*self.r**2*sin(2*self.vertheta)+(1-2*self.xi)*self.R*self.r*sin(self.vertheta))/(2*self.R*self.r*self.xi)**2)*\
                                              2*(-self.k.warping_fun_w_1xdiff()/(self.R*self.r*self.xi)-self.k.warping_fun_w()*sin(self.vertheta)/(self.R*self.xi)**2+\
                                                          self.k.warping_fun_v()*(1+2*self.xi)/(4*self.R*self.r*self.xi**2)+self.k.warping_fun_u_1xdiff()*cos(self.vertheta)/(2*self.R*self.r*self.xi)-\
                                                              self.k.warping_fun_u_1xdiff()/(4*self.r*self.r*self.xi) + self.k.warping_fun_u()*(2*self.r**2*sin(2*self.vertheta)+(1-2*self.xi)*self.R*self.r*sin(self.vertheta))/(2*self.R*self.r*self.xi)**2)), 'numpy')
        D_ik=(quad(lambda vertheta:func(vertheta),0,math.pi*2)[0])*self.r
        return D_ik
    # def D_ik(self):

    #     func = lambdify((self.vertheta),self.xi*( self.t*(self.k.warping_fun_u_1xdiff()/self.r + self.k.warping_fun_u()*sin(self.vertheta)/(self.R*self.xi) + self.k.warping_fun_v()/(self.R*self.xi)) *\
    #                                             (self.i.warping_fun_u_1xdiff()/self.r + self.i.warping_fun_u()*sin(self.vertheta)/(self.R*self.xi) + self.i.warping_fun_v()/(self.R*self.xi))+\
    #                                      (self.t**3/12)*(-self.i.warping_fun_w_1xdiff()/(self.R*self.r*self.xi)-self.i.warping_fun_w()*sin(self.vertheta)/(self.R*self.xi)**2+\
    #                                                      self.i.warping_fun_v()/(self.R*self.r*self.xi)+self.i.warping_fun_u_1xdiff()*cos(self.vertheta)/(self.R*self.r*self.xi)-\
    #                                                          self.i.warping_fun_u()*(cos(self.vertheta)*sin(self.vertheta))/(self.R*self.xi)**2)*\
    #                                          (-self.k.warping_fun_w_1xdiff()/(self.R*self.r*self.xi)-self.k.warping_fun_w()*sin(self.vertheta)/(self.R*self.xi)**2+\
    #                                                      self.k.warping_fun_v()/(self.R*self.r*self.xi)+self.k.warping_fun_u_1xdiff()*cos(self.vertheta)/(self.R*self.r*self.xi)-\
    #                                                          self.k.warping_fun_u()*(cos(self.vertheta)*sin(self.vertheta))/(self.R*self.xi)**2)), 'numpy')
    #     D_ik=(quad(lambda vertheta:func(vertheta),0,math.pi*2)[0])*self.r
    #     return D_ik
    def D_1mu_ik(self):
        if self.mu!=0:
            func = lambdify((self.vertheta),self.xi*( (self.t/((self.r*self.R*self.xi)*(1-self.mu**2)))*(self.i.warping_fun_v_1xdiff()+self.i.warping_fun_w()) *(self.k.warping_fun_u())+\
                                             self.t**3/((self.r*self.R*self.xi)**2*12*(1-self.mu**2))*(-self.i.warping_fun_w_2xdiff()+self.i.warping_fun_v_1xdiff())*\
                                              (self.k.warping_fun_u()*cos(self.vertheta)-self.k.warping_fun_w())), 'numpy')       
            D_1mu_ik=(quad(lambda vertheta:func(vertheta),0,math.pi*2)[0])*self.r*self.mu
        else:
            D_1mu_ik=0
        return D_1mu_ik

    def D_2mu_ik(self):
        if self.mu!=0:
            func = lambdify((self.vertheta),self.xi*( (self.t/((self.r*self.R*self.xi)*(1-self.mu**2)))*(self.k.warping_fun_v_1xdiff()+self.k.warping_fun_w()) *(self.i.warping_fun_u())+\
                                             self.t**3/((self.r*self.R*self.xi)**2*12*(1-self.mu**2))*(-self.k.warping_fun_w_2xdiff()+self.k.warping_fun_v_1xdiff())*\
                                              (self.i.warping_fun_u()*cos(self.vertheta)-self.i.warping_fun_w())), 'numpy')       
            D_2mu_ik=(quad(lambda vertheta:func(vertheta),0,math.pi*2)[0])*self.r*self.mu
        else:
            D_2mu_ik=0
        return D_2mu_ik

    def D_3mu_ik(self):
        if self.mu!=0:
            func = lambdify((self.vertheta),self.xi*( (self.t/((self.r*self.R*self.xi)*(1-self.mu**2)))*(self.i.warping_fun_v_1xdiff()+self.i.warping_fun_w()) *(self.k.warping_fun_w()*cos(self.vertheta)-self.k.warping_fun_v()*sin(self.vertheta))+\
                                             sin(self.vertheta)*self.t**3/((self.r**3*self.R*self.xi)*12*(1-self.mu**2))*(-self.i.warping_fun_w_2xdiff()+self.i.warping_fun_v_1xdiff())*\
                                              (self.k.warping_fun_w_1xdiff()-self.k.warping_fun_v())), 'numpy')       
            D_3mu_ik=(quad(lambda vertheta:func(vertheta),0,math.pi*2)[0])*self.r*self.mu
        else:
            D_3mu_ik=0
        return D_3mu_ik
    def D_4mu_ik(self):
        if self.mu!=0:
            func = lambdify((self.vertheta),self.xi*( (self.t/((self.r*self.R*self.xi)*(1-self.mu**2)))*(self.k.warping_fun_v_1xdiff()+self.k.warping_fun_w()) *(self.i.warping_fun_w()*cos(self.vertheta)-self.i.warping_fun_v()*sin(self.vertheta))+\
                                             sin(self.vertheta)*self.t**3/((self.r**3*self.R*self.xi)*12*(1-self.mu**2))*(-self.k.warping_fun_w_2xdiff()+self.k.warping_fun_v_1xdiff())*\
                                              (self.i.warping_fun_w_1xdiff()-self.i.warping_fun_v())), 'numpy')       
            D_4mu_ik=(quad(lambda vertheta:func(vertheta),0,math.pi*2)[0])*self.r*self.mu
        else:
            D_4mu_ik=0
        return D_4mu_ik    
#_____________________________________________________________

class linear_curved_GBT_ik_coupling_matrix_additional:  #if R is small in compartion with r  
    def __init__ (self,k1,k2,R,t,r,xi,mu):
        self.r=r
        self.t=t
        self.R=R
        self.mu=mu        
        self.vertheta=Symbol('vertheta')
        self.i=GBT_func_curve(k1,r,self.vertheta,R)     
        self.k=GBT_func_curve(k2,r,self.vertheta,R)
        self.xi=(1+(xi*(r/R)*cos(self.vertheta)))
    # 
    def C1_ik(self):

        func = lambdify((self.vertheta),cos(self.vertheta)*( (self.t/((self.R*self.xi)**2*(1-self.mu**2)))*self.i.warping_fun_u() *self.k.warping_fun_u()+\
                                         self.t**3/((self.R*self.xi)**4*12*(1-self.mu**2))*(self.k.warping_fun_u()*cos(self.vertheta)-self.k.warping_fun_w())*\
                                          (self.i.warping_fun_u()*cos(self.vertheta)-self.i.warping_fun_w())), 'numpy')       
        C_1_ik=(quad(lambda vertheta:func(vertheta),0,math.pi*2)[0])*self.r**2
        return C_1_ik
    def C2_ik(self):

        func = lambdify((self.vertheta),cos(self.vertheta)*( (self.t/((self.R*self.xi)**2*(1-self.mu**2)))*(self.i.warping_fun_w()*cos(self.vertheta)-self.i.warping_fun_v()*sin(self.vertheta)) *self.k.warping_fun_u()+\
                                         sin(self.vertheta)*self.t**3/((self.R*self.xi)**3*self.r*12*(1-self.mu**2))*(self.k.warping_fun_u()*cos(self.vertheta)-self.k.warping_fun_w())*\
                                          (self.i.warping_fun_w_1xdiff()-self.i.warping_fun_v())), 'numpy')       
        C_2_ik=(quad(lambda vertheta:func(vertheta),0,math.pi*2)[0])*self.r**2
        return C_2_ik
    def C3_ik(self):

        func = lambdify((self.vertheta), cos(self.vertheta)*((self.t/((self.R*self.xi)**2*(1-self.mu**2)))*(self.k.warping_fun_w()*cos(self.vertheta)-self.k.warping_fun_v()*sin(self.vertheta)) *self.i.warping_fun_u()+\
                                         sin(self.vertheta)*self.t**3/((self.R*self.xi)**3*self.r*12*(1-self.mu**2))*(self.i.warping_fun_u()*cos(self.vertheta)-self.i.warping_fun_w())*\
                                          (self.k.warping_fun_w_1xdiff()-self.k.warping_fun_v())), 'numpy')       
        C_3_ik=(quad(lambda vertheta:func(vertheta),0,math.pi*2)[0])*self.r**2
        return C_3_ik
    def C4_ik(self):

        func = lambdify((self.vertheta), cos(self.vertheta)*((self.t/((self.R*self.xi)**2*(1-self.mu**2)))*(self.k.warping_fun_w()*cos(self.vertheta)-self.k.warping_fun_v()*sin(self.vertheta)) *(self.i.warping_fun_w()*cos(self.vertheta)-self.i.warping_fun_v()*sin(self.vertheta))+\
                                         sin(self.vertheta)*sin(self.vertheta)*self.t**3/((self.R*self.xi*self.r)**2*12*(1-self.mu**2))*(self.i.warping_fun_w_1xdiff()-self.i.warping_fun_v())*\
                                          (self.k.warping_fun_w_1xdiff()-self.k.warping_fun_v())), 'numpy')       
        C_4_ik=(quad(lambda vertheta:func(vertheta),0,math.pi*2)[0])*self.r**2
        return C_4_ik
    def B_ik(self):

        func = lambdify((self.vertheta),cos(self.vertheta)*( (self.t/((self.r)**2*(1-self.mu**2)))*(self.i.warping_fun_v_1xdiff()+self.i.warping_fun_w()) *(self.k.warping_fun_v_1xdiff()+self.k.warping_fun_w())+\
                                         self.t**3/((self.r)**4*12*(1-self.mu**2))*(-self.i.warping_fun_w_2xdiff()+self.i.warping_fun_v_1xdiff())*\
                                          (-self.k.warping_fun_w_2xdiff()+self.k.warping_fun_v_1xdiff())), 'numpy')       
        B_ik=(quad(lambda vertheta:func(vertheta),0,math.pi*2)[0])*self.r**2
        return B_ik
    def D_ik(self):

        func = lambdify((self.vertheta), cos(self.vertheta)*(self.t*(self.k.warping_fun_u_1xdiff()/self.r + self.k.warping_fun_u()*sin(self.vertheta)/(self.R*self.xi) + self.k.warping_fun_v()/(self.R*self.xi)) *\
                                                (self.i.warping_fun_u_1xdiff()/self.r + self.i.warping_fun_u()*sin(self.vertheta)/(self.R*self.xi) + self.i.warping_fun_v()/(self.R*self.xi))+\
                                         (self.t**3/12)*2*(-self.i.warping_fun_w_1xdiff()/(self.R*self.r*self.xi)-self.i.warping_fun_w()*sin(self.vertheta)/(self.R*self.xi)**2+\
                                                         self.i.warping_fun_v()*(1+2*self.xi)/(4*self.R*self.r*self.xi**2)+self.i.warping_fun_u_1xdiff()*cos(self.vertheta)/(2*self.R*self.r*self.xi)-\
                                                             self.i.warping_fun_u_1xdiff()/(4*self.r*self.r*self.xi) + self.i.warping_fun_u()*(2*self.r**2*sin(2*self.vertheta)+(1-2*self.xi)*self.R*self.r*sin(self.vertheta))/(2*self.R*self.r*self.xi)**2)*\
                                             2*(-self.k.warping_fun_w_1xdiff()/(self.R*self.r*self.xi)-self.k.warping_fun_w()*sin(self.vertheta)/(self.R*self.xi)**2+\
                                                         self.k.warping_fun_v()*(1+2*self.xi)/(4*self.R*self.r*self.xi**2)+self.k.warping_fun_u_1xdiff()*cos(self.vertheta)/(2*self.R*self.r*self.xi)-\
                                                             self.k.warping_fun_u_1xdiff()/(4*self.r*self.r*self.xi) + self.k.warping_fun_u()*(2*self.r**2*sin(2*self.vertheta)+(1-2*self.xi)*self.R*self.r*sin(self.vertheta))/(2*self.R*self.r*self.xi)**2)), 'numpy')
        D_ik=(quad(lambda vertheta:func(vertheta),0,math.pi*2)[0])*self.r**2
        return D_ik

    def D_1mu_ik(self):
        if self.mu!=0:
            func = lambdify((self.vertheta), cos(self.vertheta)*((self.t/((self.r*self.R*self.xi)*(1-self.mu**2)))*(self.i.warping_fun_v_1xdiff()+self.i.warping_fun_w()) *(self.k.warping_fun_u())+\
                                             self.t**3/((self.r*self.R*self.xi)**2*12*(1-self.mu**2))*(-self.i.warping_fun_w_2xdiff()+self.i.warping_fun_v_1xdiff())*\
                                              (self.k.warping_fun_u()*cos(self.vertheta)-self.k.warping_fun_w())), 'numpy')       
            D_1mu_ik=(quad(lambda vertheta:func(vertheta),0,math.pi*2)[0])*self.r**2*self.mu
        else:
            D_1mu_ik=0
        return D_1mu_ik

    def D_2mu_ik(self):
        if self.mu!=0:
            func = lambdify((self.vertheta), cos(self.vertheta)*((self.t/((self.r*self.R*self.xi)*(1-self.mu**2)))*(self.k.warping_fun_v_1xdiff()+self.k.warping_fun_w()) *(self.i.warping_fun_u())+\
                                             self.t**3/((self.r*self.R*self.xi)**2*12*(1-self.mu**2))*(-self.k.warping_fun_w_2xdiff()+self.k.warping_fun_v_1xdiff())*\
                                              (self.i.warping_fun_u()*cos(self.vertheta)-self.i.warping_fun_w())), 'numpy')       
            D_2mu_ik=(quad(lambda vertheta:func(vertheta),0,math.pi*2)[0])*self.r**2*self.mu
        else:
            D_2mu_ik=0
        return D_2mu_ik

    def D_3mu_ik(self):
        if self.mu!=0:
            func = lambdify((self.vertheta),cos(self.vertheta)*( (self.t/((self.r*self.R*self.xi)*(1-self.mu**2)))*(self.i.warping_fun_v_1xdiff()+self.i.warping_fun_w()) *(self.k.warping_fun_w()*cos(self.vertheta)-self.k.warping_fun_v()*sin(self.vertheta))+\
                                             sin(self.vertheta)*self.t**3/((self.r**3*self.R*self.xi)*12*(1-self.mu**2))*(-self.i.warping_fun_w_2xdiff()+self.i.warping_fun_v_1xdiff())*\
                                              (self.k.warping_fun_w_1xdiff()-self.k.warping_fun_v())), 'numpy')       
            D_3mu_ik=(quad(lambda vertheta:func(vertheta),0,math.pi*2)[0])*self.r**2*self.mu
        else:
            D_3mu_ik=0
        return D_3mu_ik
    def D_4mu_ik(self):
        if self.mu!=0:
            func = lambdify((self.vertheta), cos(self.vertheta)*((self.t/((self.r*self.R*self.xi)*(1-self.mu**2)))*(self.k.warping_fun_v_1xdiff()+self.k.warping_fun_w()) *(self.i.warping_fun_w()*cos(self.vertheta)-self.i.warping_fun_v()*sin(self.vertheta))+\
                                             sin(self.vertheta)*self.t**3/((self.r**3*self.R*self.xi)*12*(1-self.mu**2))*(-self.k.warping_fun_w_2xdiff()+self.k.warping_fun_v_1xdiff())*\
                                              (self.i.warping_fun_w_1xdiff()-self.i.warping_fun_v())), 'numpy')       
            D_4mu_ik=(quad(lambda vertheta:func(vertheta),0,math.pi*2)[0])*self.r**2*self.mu
        else:
            D_4mu_ik=0
        return D_4mu_ik  
#_____________________________________________________________



class LM_connectivity_array:      
# Boolean array or LM_marix size element dof X number of element  #assembly      """
    def __init__ (self,mode_list,no_beam_elem):
        self.mode_list_size=len(mode_list)
        self.no_beam_elem=no_beam_elem
        self.mode_list=mode_list
        
    def LM_array(self):
        LM_matrix=np.zeros((self.mode_list_size*4,self.no_beam_elem),int)
        sum_b=0
        for i in range(0,self.no_beam_elem):
            sum_c=0
            sum_tax=0
            sum_high=0         
            for k in range(0,self.mode_list_size):
                sum_a=0
                for j in range(0,4):
                    if   self.mode_list[k].split("|")[0]=='1':
                       LM_matrix[j+sum_c,i]= i+sum_a+sum_tax+sum_b+sum_high*(2+self.no_beam_elem*2)
                       sum_a +=1
                       
                    else:
                       LM_matrix[j+sum_c,i]= sum_high*(2+self.no_beam_elem*2)+sum_a+sum_b+sum_tax
#                       print j, sum_c,i 
#                       print LM_matrix
                       sum_a +=1
                if  self.mode_list[k].split("|")[0]=='1':
                   sum_tax+=((self.no_beam_elem*4)-(self.no_beam_elem-1))  
                else:
                   sum_high+=1
                sum_c+=4  
            sum_b +=2

            
        return LM_matrix 
#____________________________________________________________
        
class Spherical_transformation_matrix:   # independent 

    def __init__ (self,vertheta,varphi):
        self.vertheta=vertheta
        self.varphi=varphi

    def local_to_global(self):
        theta=self.vertheta
        varphi=self.varphi
        return np.mat([[-math.sin(theta) * math.sin(varphi),math.cos(varphi),math.cos(theta) * math.sin(varphi)],[-math.sin(theta) * math.cos(varphi),-math.sin(varphi),math.cos(theta) * math.cos(varphi)],[math.cos(theta),0,math.sin(theta)]])
        # c_vertheta=math.cos(self.vertheta)
        # s_vertheta=math.sin(self.vertheta)
        # c_varphi=math.cos(self.varphi)
        # s_varphi=math.sin(self.varphi)        
        # return np.matrix([[s_varphi*c_vertheta,c_varphi*c_vertheta,-s_vertheta],[s_varphi*s_vertheta,c_varphi*s_vertheta,c_vertheta],[c_varphi,-s_varphi,0]])
    
    
    
    
#_____________________________________________________________            
class Dof:#all_Hermite_only
    def __init__ (self,mode_list,mat_csi,no_beam_elem):
        self.mode_list_size=len(mode_list)
        self.no_beam_elem=no_beam_elem
        self.mode_list=mode_list
        self.mat_csi=mat_csi
    def Element(self):
        return (self.mode_list_size*4)            
    def Global(self):  
        ax=0
        count=0
        for k in range(0,self.mode_list_size):
            if self.mode_list[k].split("|")[0]=='1':
                ax=((self.no_beam_elem*4)-(self.no_beam_elem-1)) 
                count+=1
                 
        return (ax+(self.mode_list_size-count)*(2+self.no_beam_elem*2))              
        
#________________________________________

class Stiffness_coff:    
    # gbt stifness coefficient
    def __init__ (self, k, m, r, t, E, mu):
        self.k=k
        self.m=m
        self.r=r
        self.t=t
        self.E=E
        self.mu=mu
        self.G=self.E/(2.0*(1.0+self.mu))
    def stiffness_c(self):
#        if self.k==1:
#            return (2*math.pi*self.t*self.r)
#        else:
#            return (math.pi*self.t*self.r**3*(1+(self.m**4/((1-self.mu**2)*12)*(self.t**2/self.r**2))))
        K=self.E*self.t**3/(12.0*(1.0-self.mu**2))
        if self.k.split("|")[0]=='t':   
           return (0)
        elif self.k.split("|")[0]=='a' : 
           return (2*math.pi*self.r*K)
        elif self.k.split("|")[0]=='1' :
           return (2*math.pi*self.t*self.r)
        elif self.k.split("|")[1]== 'c' :
           #return(math.pi/4)*((self.r+self.t/2)**4-(self.r-self.t/2)**4) 
           return (math.pi*self.t*self.r**3*(1+1*(self.m**4/((1-self.mu**2)*12)*(self.t**2/self.r**2))))

    def stiffness_d(self):
  #      K=self.E*self.t**3/(12.0*(1.0-self.mu**2))
        if self.k.split("|")[0]=='t':         
           return((self.t*self.r**2+(self.t**3)/12)*2*math.pi*self.r)# (self.t*self.r**3*2*math.pi)
        elif self.k.split("|")[0]=='a' or self.k.split("|")[0]=='1' : 
           return (0)
        elif self.k.split("|")[1]== 'c' :
           return (math.pi*self.t**3/(self.r*3)*self.m**2*(self.m**2-1)*((self.m**2)/(1-self.mu)-1))
        
    def stiffness_b(self):
        K=self.E*self.t**3/(12.0*(1.0-self.mu**2))
        if  self.k.split("|")[0]=='t' or  self.k.split("|")[0]=='1':         
           return (0)
        elif self.k.split("|")[0]=='a' : 
           return (2*math.pi*self.E*self.t/(self.r*(1.0-self.mu**2)))
        elif self.k.split("|")[1]== 'c' :       
           return math.pi*(K/self.r**3)*self.m**4*(self.m**2-1)**2

#_____________________________________________________________            
class Elastic_stiffness_matrix_GBT_curved:
    # combined elastic stiffness matrix
    #u1,u2,u3,u4 mode_1 v1,w1,v2,w2 mode_3,5,7....
    def __init__ (self,varphi,E,R,mode_list,mat_csi,ik_vector,mu,k_e):
        self.varphi=varphi       
        self.E=E
        self.R=R
        self.mode_list=mode_list
        self.mode_list_size=len(mode_list)
        self.ik_vector=ik_vector
        self.k_e=k_e       
        self.LM_matrix=LM_connectivity_array.LM_array(LM_connectivity_array(self.mode_list,1))
        self.mat_csi=mat_csi
        self.mu=mu
        self.G=self.E/(2.0*(1.0+self.mu))
    def element(self) :
        varphi=self.varphi  
        R=self.R

        # #mode_highmode_1
        # shHTxxTxshL=np.mat([[0.9e1 / 0.2e1 / varphi ** 2 * R,-0.9e1 / 0.2e1 / varphi ** 2 * R,-0.9e1 / 0.2e1 / varphi ** 2 * R,0.9e1 / 0.2e1 / varphi ** 2 * R],[0.13e2 / 0.4e1 / varphi * R ** 2,-0.9e1 / 0.4e1 / varphi * R ** 2,-0.9e1 / 0.4e1 / varphi * R ** 2,0.5e1 / 0.4e1 / varphi * R ** 2],[-0.9e1 / 0.2e1 / varphi ** 2 * R,0.9e1 / 0.2e1 / varphi ** 2 * R,0.9e1 / 0.2e1 / varphi ** 2 * R,-0.9e1 / 0.2e1 / varphi ** 2 * R],[0.5e1 / 0.4e1 / varphi * R ** 2,-0.9e1 / 0.4e1 / varphi * R ** 2,-0.9e1 / 0.4e1 / varphi * R ** 2,0.13e2 / 0.4e1 / varphi * R ** 2]])
        VxxVxx=self.R*np.mat([[12 / varphi ** 3 * 1.0,6 / varphi ** 2 * 1.0 ** 2,-12 / varphi ** 3 * 1.0,6 / varphi ** 2 * 1.0 ** 2],[6 / varphi ** 2 * 1.0 ** 2,4 / varphi * 1.0 ** 3,-6 / varphi ** 2 * 1.0 ** 2,2 / varphi * 1.0 ** 3],[-12 / varphi ** 3 * 1.0,-6 / varphi ** 2 * 1.0 ** 2,12 / varphi ** 3 * 1.0,-6 / varphi ** 2 * 1.0 ** 2],[6 / varphi ** 2 * 1.0 ** 2,2 / varphi * 1.0 ** 3,-6 / varphi ** 2 * 1.0 ** 2,4 / varphi * 1.0 ** 3]])
        VxxV=self.R*np.mat([[-0.6e1 / 0.5e1 / varphi * 1.0,-1.0 ** 2 / 10,0.6e1 / 0.5e1 / varphi * 1.0,-1.0 ** 2 / 10],[-0.11e2 / 0.10e2 * 1.0 ** 2,-0.2e1 / 0.15e2 * varphi * 1.0 ** 3,1.0 ** 2 / 10,varphi * 1.0 ** 3 / 30],[0.6e1 / 0.5e1 / varphi * 1.0,1.0 ** 2 / 10,-0.6e1 / 0.5e1 / varphi * 1.0,1.0 ** 2 / 10],[-1.0 ** 2 / 10,varphi * 1.0 ** 3 / 30,0.11e2 / 0.10e2 * 1.0 ** 2,-0.2e1 / 0.15e2 * varphi * 1.0 ** 3]])
        VVxx=self.R*np.mat([[-0.6e1 / 0.5e1 / varphi * 1.0,-0.11e2 / 0.10e2 * 1.0 ** 2,0.6e1 / 0.5e1 / varphi * 1.0,-1.0 ** 2 / 10],[-1.0 ** 2 / 10,-0.2e1 / 0.15e2 * varphi * 1.0 ** 3,1.0 ** 2 / 10,varphi * 1.0 ** 3 / 30],[0.6e1 / 0.5e1 / varphi * 1.0,1.0 ** 2 / 10,-0.6e1 / 0.5e1 / varphi * 1.0,0.11e2 / 0.10e2 * 1.0 ** 2],[-1.0 ** 2 / 10,varphi * 1.0 ** 3 / 30,1.0 ** 2 / 10,-0.2e1 / 0.15e2 * varphi * 1.0 ** 3]])
        VV=self.R*np.mat([[0.13e2 / 0.35e2 * varphi * 1.0,0.11e2 / 0.210e3 * 1.0 ** 2 * varphi ** 2,0.9e1 / 0.70e2 * varphi * 1.0,-0.13e2 / 0.420e3 * 1.0 ** 2 * varphi ** 2],[0.11e2 / 0.210e3 * 1.0 ** 2 * varphi ** 2,varphi ** 3 * 1.0 ** 3 / 105,0.13e2 / 0.420e3 * 1.0 ** 2 * varphi ** 2,-varphi ** 3 * 1.0 ** 3 / 140],[0.9e1 / 0.70e2 * varphi * 1.0,0.13e2 / 0.420e3 * 1.0 ** 2 * varphi ** 2,0.13e2 / 0.35e2 * varphi * 1.0,-0.11e2 / 0.210e3 * 1.0 ** 2 * varphi ** 2],[-0.13e2 / 0.420e3 * 1.0 ** 2 * varphi ** 2,-varphi ** 3 * 1.0 ** 3 / 140,-0.11e2 / 0.210e3 * 1.0 ** 2 * varphi ** 2,varphi ** 3 * 1.0 ** 3 / 105]])
        VxVx=self.R*np.mat([[0.6e1 / 0.5e1 / varphi * 1.0,1.0 ** 2 / 10,-0.6e1 / 0.5e1 / varphi * 1.0,1.0 ** 2 / 10],[1.0 ** 2 / 10,0.2e1 / 0.15e2 * varphi * 1.0 ** 3,-1.0 ** 2 / 10,-varphi * 1.0 ** 3 / 30],[-0.6e1 / 0.5e1 / varphi * 1.0,-1.0 ** 2 / 10,0.6e1 / 0.5e1 / varphi * 1.0,-1.0 ** 2 / 10],[1.0 ** 2 / 10,-varphi * 1.0 ** 3 / 30,-1.0 ** 2 / 10,0.2e1 / 0.15e2 * varphi * 1.0 ** 3]])
        #mode_1_1
        A_VxVx=self.R*np.mat([[0.37e2 / 0.10e2 / varphi * 1.0,-0.189e3 / 0.40e2 / varphi * 1.0,0.27e2 / 0.20e2 / varphi * 1.0,-0.13e2 / 0.40e2 / varphi * 1.0],[-0.189e3 / 0.40e2 / varphi * 1.0,0.54e2 / 0.5e1 / varphi * 1.0,-0.297e3 / 0.40e2 / varphi * 1.0,0.27e2 / 0.20e2 / varphi * 1.0],[0.27e2 / 0.20e2 / varphi * 1.0,-0.297e3 / 0.40e2 / varphi * 1.0,0.54e2 / 0.5e1 / varphi * 1.0,-0.189e3 / 0.40e2 / varphi * 1.0],[-0.13e2 / 0.40e2 / varphi * 1.0,0.27e2 / 0.20e2 / varphi * 1.0,-0.189e3 / 0.40e2 / varphi * 1.0,0.37e2 / 0.10e2 / varphi * 1.0]])
        A_VV=self.R*np.mat([[0.8e1 / 0.105e3 * varphi * 1.0,0.33e2 / 0.560e3 * varphi * 1.0,-0.3e1 / 0.140e3 * varphi * 1.0,0.19e2 / 0.1680e4 * varphi * 1.0],[0.33e2 / 0.560e3 * varphi * 1.0,0.27e2 / 0.70e2 * varphi * 1.0,-0.27e2 / 0.560e3 * varphi * 1.0,-0.3e1 / 0.140e3 * varphi * 1.0],[-0.3e1 / 0.140e3 * varphi * 1.0,-0.27e2 / 0.560e3 * varphi * 1.0,0.27e2 / 0.70e2 * varphi * 1.0,0.33e2 / 0.560e3 * varphi * 1.0],[0.19e2 / 0.1680e4 * varphi * 1.0,-0.3e1 / 0.140e3 * varphi * 1.0,0.33e2 / 0.560e3 * varphi * 1.0,0.8e1 / 0.105e3 * varphi * 1.0]])
 
        element_linear_mat_local=np.zeros((Dof.Element(Dof(self.mode_list,self.mat_csi,0)),Dof.Element(Dof(self.mode_list,self.mat_csi,0))),dtype=float)
        #mode_1_highmode
        C1_shHTxxTxshL=self.R*np.mat([[0.9e1 / 0.2e1 / varphi ** 2 * 1.0,-0.9e1 / 0.2e1 / varphi ** 2 * 1.0,-0.9e1 / 0.2e1 / varphi ** 2 * 1.0,0.9e1 / 0.2e1 / varphi ** 2 * 1.0],[0.13e2 / 0.4e1 / varphi * 1.0 ** 2,-0.9e1 / 0.4e1 / varphi * 1.0 ** 2,-0.9e1 / 0.4e1 / varphi * 1.0 ** 2,0.5e1 / 0.4e1 / varphi * 1.0 ** 2],[-0.9e1 / 0.2e1 / varphi ** 2 * 1.0,0.9e1 / 0.2e1 / varphi ** 2 * 1.0,0.9e1 / 0.2e1 / varphi ** 2 * 1.0,-0.9e1 / 0.2e1 / varphi ** 2 * 1.0],[0.5e1 / 0.4e1 / varphi * 1.0 ** 2,-0.9e1 / 0.4e1 / varphi * 1.0 ** 2,-0.9e1 / 0.4e1 / varphi * 1.0 ** 2,0.13e2 / 0.4e1 / varphi * 1.0 ** 2]])
        C2_shHTxxTshL=self.R*np.mat([[-0.11e2 / 0.20e2 / varphi * 1.0,-0.27e2 / 0.20e2 / varphi * 1.0,0.27e2 / 0.20e2 / varphi * 1.0,0.11e2 / 0.20e2 / varphi * 1.0],[-0.2e1 / 0.5e1 * 1.0 ** 2,-0.21e2 / 0.20e2 * 1.0 ** 2,0.3e1 / 0.10e2 * 1.0 ** 2,0.3e1 / 0.20e2 * 1.0 ** 2],[0.11e2 / 0.20e2 / varphi * 1.0,0.27e2 / 0.20e2 / varphi * 1.0,-0.27e2 / 0.20e2 / varphi * 1.0,-0.11e2 / 0.20e2 / varphi * 1.0],[-0.3e1 / 0.20e2 * 1.0 ** 2,-0.3e1 / 0.10e2 * 1.0 ** 2,0.21e2 / 0.20e2 * 1.0 ** 2,0.2e1 / 0.5e1 * 1.0 ** 2]])
        C3_shHTTxshL=self.R*np.mat([[-0.19e2 / 0.20e2 * 1.0,0.9e1 / 0.20e2 * 1.0,0.9e1 / 0.20e2 * 1.0,1.0 / 20],[-varphi * 1.0 ** 2 / 12,-0.3e1 / 0.40e2 * varphi * 1.0 ** 2,0.3e1 / 0.20e2 * varphi * 1.0 ** 2,varphi * 1.0 ** 2 / 120],[-1.0 / 20,-0.9e1 / 0.20e2 * 1.0,-0.9e1 / 0.20e2 * 1.0,0.19e2 / 0.20e2 * 1.0],[varphi * 1.0 ** 2 / 120,0.3e1 / 0.20e2 * varphi * 1.0 ** 2,-0.3e1 / 0.40e2 * varphi * 1.0 ** 2,-varphi * 1.0 ** 2 / 12]])
        C4_B_shHTTshL=self.R*np.mat([[0.4e1 / 0.35e2 * varphi * 1.0,0.93e2 / 0.280e3 * varphi * 1.0,0.3e1 / 0.70e2 * varphi * 1.0,0.3e1 / 0.280e3 * varphi * 1.0],[1.0 ** 2 * varphi ** 2 / 140,0.3e1 / 0.56e2 * 1.0 ** 2 * varphi ** 2,0.3e1 / 0.140e3 * 1.0 ** 2 * varphi ** 2,1.0 ** 2 * varphi ** 2 / 840],[0.3e1 / 0.280e3 * varphi * 1.0,0.3e1 / 0.70e2 * varphi * 1.0,0.93e2 / 0.280e3 * varphi * 1.0,0.4e1 / 0.35e2 * varphi * 1.0],[-1.0 ** 2 * varphi ** 2 / 840,-0.3e1 / 0.140e3 * 1.0 ** 2 * varphi ** 2,-0.3e1 / 0.56e2 * 1.0 ** 2 * varphi ** 2,-1.0 ** 2 * varphi ** 2 / 140]])
        D_shHTxTshL=self.R*np.mat([[-1.0 / 20,-0.9e1 / 0.20e2 * 1.0,-0.9e1 / 0.20e2 * 1.0,-1.0 / 20],[varphi * 1.0 ** 2 / 12,0.3e1 / 0.40e2 * varphi * 1.0 ** 2,-0.3e1 / 0.20e2 * varphi * 1.0 ** 2,-varphi * 1.0 ** 2 / 120],[1.0 / 20,0.9e1 / 0.20e2 * 1.0,0.9e1 / 0.20e2 * 1.0,1.0 / 20],[-varphi * 1.0 ** 2 / 120,-0.3e1 / 0.20e2 * varphi * 1.0 ** 2,0.3e1 / 0.40e2 * varphi * 1.0 ** 2,varphi * 1.0 ** 2 / 12]])

        #mode_highmode_1
        shHTxxTxshL=np.mat([[0.9e1 / 0.2e1 / varphi ** 2 * 1.0,-0.9e1 / 0.2e1 / varphi ** 2 * 1.0,-0.9e1 / 0.2e1 / varphi ** 2 * 1.0,0.9e1 / 0.2e1 / varphi ** 2 * 1.0],[0.13e2 / 0.4e1 / varphi * 1.0 ** 2,-0.9e1 / 0.4e1 / varphi * 1.0 ** 2,-0.9e1 / 0.4e1 / varphi * 1.0 ** 2,0.5e1 / 0.4e1 / varphi * 1.0 ** 2],[-0.9e1 / 0.2e1 / varphi ** 2 * 1.0,0.9e1 / 0.2e1 / varphi ** 2 * 1.0,0.9e1 / 0.2e1 / varphi ** 2 * 1.0,-0.9e1 / 0.2e1 / varphi ** 2 * 1.0],[0.5e1 / 0.4e1 / varphi * 1.0 ** 2,-0.9e1 / 0.4e1 / varphi * 1.0 ** 2,-0.9e1 / 0.4e1 / varphi * 1.0 ** 2,0.13e2 / 0.4e1 / varphi * 1.0 ** 2]])

        for i in range(len(self.ik_vector)):
            [k1,k2]=(self.ik_vector[i,0:2]).astype(str)
            dim_1=(self.ik_vector[i,12:16]).astype(int)
            dim_2=(self.ik_vector[i,16:20]).astype(int)


            [C1,C2,C3,C4,B_t,G_s,D_1mu,D_2mu,D_3mu,D_4mu]=(self.ik_vector[i,2:12]).astype(float)  
        #    [C1,C2,C3,C4,B_t,G_s]=[0,0,0,0,0,0] #test
         #   print (k1,k2,C1,C2,C3,C4,B_t,G_s,D_1mu,D_2mu,D_3mu,D_4mu)
            if k1.split("|")[0]=='1' and k2.split("|")[0]=='1' :
            #    [C1,C2,C3,C4,B_t,G_s,D_1mu,D_2mu,D_3mu,D_4mu]=[0,0,0,0,0,0,0,0,0,0]
                element_linear_mat_local[np.tile(dim_1, (4, 1)),np.tile(dim_2, (4, 1)).T]+=self.E* C1*A_VxVx + self.G*G_s*A_VV
            
            if k1.split("|")[0]=='1' and k2.split("|")[0]!='1' :
             #   [C1,C2,C3,C4,B_t,G_s,D_3mu,D_4mu]=[0,0,0,0,0,0,0,0]
                element_linear_mat_local[np.tile(dim_1, (4, 1)),np.tile(dim_2, (4, 1)).T]+=self.E* (C1*C1_shHTxxTxshL + C2*C2_shHTxxTshL + C3*C3_shHTTxshL + C4* C4_B_shHTTshL  + B_t* C4_B_shHTTshL + D_2mu*C3_shHTTxshL)+self.G*G_s*D_shHTxTshL

            if k1.split("|")[0]!='1' and k2.split("|")[0]=='1' :
             #   [C1,C2,C3,C4,B_t,G_s,D_3mu,D_4mu]=[0,0,0,0,0,0,0,0]
                element_linear_mat_local[np.tile(dim_1, (4, 1)),np.tile(dim_2, (4, 1)).T]+=self.E* (C1*np.transpose(C1_shHTxxTxshL) + C2*np.transpose(C3_shHTTxshL) + C3*np.transpose(C2_shHTxxTshL) + C4* np.transpose(C4_B_shHTTshL)  + B_t* np.transpose(C4_B_shHTTshL ) +  D_1mu* np.transpose(C3_shHTTxshL) )+self.G*G_s*np.transpose(D_shHTxTshL)
            
            
            if k1.split("|")[0]!='1' and k2.split("|")[0]!='1' :
               # [C1,C2,C3,C4,B_t,G_s,D_1mu,D_2mu,D_3mu,D_4mu]=[0,0,0,0,0,0,0,0,0,0]
                element_linear_mat_local[np.tile(dim_1, (4, 1)),np.tile(dim_2, (4, 1)).T]+=self.E* (C1*VxxVxx + C2*VxxV + C3*VVxx + C4*VV  + B_t*VV + D_1mu*VxxV + D_2mu*VVxx + VV*(D_3mu+D_4mu) )+self.G*G_s*VxVx
                
        return element_linear_mat_local*self.k_e

#_____________________________________________________________            
class Elastic_stiffness_matrix_GBT_curved_additional:
    # combined elastic stiffness matrix
    #u1,u2,u3,u4 mode_1 v1,w1,v2,w2 mode_3,5,7....
    def __init__ (self,varphi,E,R,mode_list,mat_csi,ik_vector,mu,k_e):
        self.varphi=varphi       
        self.E=E
        self.R=R
        self.mode_list=mode_list
        self.mode_list_size=len(mode_list)
        self.ik_vector=ik_vector
        self.k_e=k_e       
        self.LM_matrix=LM_connectivity_array.LM_array(LM_connectivity_array(self.mode_list,1))
        self.mat_csi=mat_csi
        self.mu=mu
        self.G=self.E/(2.0*(1.0+self.mu))
    def element(self) :
        varphi=self.varphi  
       # #mode_highmode_1
        # shHTxxTxshL=np.mat([[0.9e1 / 0.2e1 / varphi ** 2 * R,-0.9e1 / 0.2e1 / varphi ** 2 * R,-0.9e1 / 0.2e1 / varphi ** 2 * R,0.9e1 / 0.2e1 / varphi ** 2 * R],[0.13e2 / 0.4e1 / varphi * R ** 2,-0.9e1 / 0.4e1 / varphi * R ** 2,-0.9e1 / 0.4e1 / varphi * R ** 2,0.5e1 / 0.4e1 / varphi * R ** 2],[-0.9e1 / 0.2e1 / varphi ** 2 * R,0.9e1 / 0.2e1 / varphi ** 2 * R,0.9e1 / 0.2e1 / varphi ** 2 * R,-0.9e1 / 0.2e1 / varphi ** 2 * R],[0.5e1 / 0.4e1 / varphi * R ** 2,-0.9e1 / 0.4e1 / varphi * R ** 2,-0.9e1 / 0.4e1 / varphi * R ** 2,0.13e2 / 0.4e1 / varphi * R ** 2]])
        VxxVxx=np.mat([[12 / varphi ** 3 * 1.0,6 / varphi ** 2 * 1.0 ** 2,-12 / varphi ** 3 * 1.0,6 / varphi ** 2 * 1.0 ** 2],[6 / varphi ** 2 * 1.0 ** 2,4 / varphi * 1.0 ** 3,-6 / varphi ** 2 * 1.0 ** 2,2 / varphi * 1.0 ** 3],[-12 / varphi ** 3 * 1.0,-6 / varphi ** 2 * 1.0 ** 2,12 / varphi ** 3 * 1.0,-6 / varphi ** 2 * 1.0 ** 2],[6 / varphi ** 2 * 1.0 ** 2,2 / varphi * 1.0 ** 3,-6 / varphi ** 2 * 1.0 ** 2,4 / varphi * 1.0 ** 3]])
        VxxV=np.mat([[-0.6e1 / 0.5e1 / varphi * 1.0,-1.0 ** 2 / 10,0.6e1 / 0.5e1 / varphi * 1.0,-1.0 ** 2 / 10],[-0.11e2 / 0.10e2 * 1.0 ** 2,-0.2e1 / 0.15e2 * varphi * 1.0 ** 3,1.0 ** 2 / 10,varphi * 1.0 ** 3 / 30],[0.6e1 / 0.5e1 / varphi * 1.0,1.0 ** 2 / 10,-0.6e1 / 0.5e1 / varphi * 1.0,1.0 ** 2 / 10],[-1.0 ** 2 / 10,varphi * 1.0 ** 3 / 30,0.11e2 / 0.10e2 * 1.0 ** 2,-0.2e1 / 0.15e2 * varphi * 1.0 ** 3]])
        VVxx=np.mat([[-0.6e1 / 0.5e1 / varphi * 1.0,-0.11e2 / 0.10e2 * 1.0 ** 2,0.6e1 / 0.5e1 / varphi * 1.0,-1.0 ** 2 / 10],[-1.0 ** 2 / 10,-0.2e1 / 0.15e2 * varphi * 1.0 ** 3,1.0 ** 2 / 10,varphi * 1.0 ** 3 / 30],[0.6e1 / 0.5e1 / varphi * 1.0,1.0 ** 2 / 10,-0.6e1 / 0.5e1 / varphi * 1.0,0.11e2 / 0.10e2 * 1.0 ** 2],[-1.0 ** 2 / 10,varphi * 1.0 ** 3 / 30,1.0 ** 2 / 10,-0.2e1 / 0.15e2 * varphi * 1.0 ** 3]])
        VV=np.mat([[0.13e2 / 0.35e2 * varphi * 1.0,0.11e2 / 0.210e3 * 1.0 ** 2 * varphi ** 2,0.9e1 / 0.70e2 * varphi * 1.0,-0.13e2 / 0.420e3 * 1.0 ** 2 * varphi ** 2],[0.11e2 / 0.210e3 * 1.0 ** 2 * varphi ** 2,varphi ** 3 * 1.0 ** 3 / 105,0.13e2 / 0.420e3 * 1.0 ** 2 * varphi ** 2,-varphi ** 3 * 1.0 ** 3 / 140],[0.9e1 / 0.70e2 * varphi * 1.0,0.13e2 / 0.420e3 * 1.0 ** 2 * varphi ** 2,0.13e2 / 0.35e2 * varphi * 1.0,-0.11e2 / 0.210e3 * 1.0 ** 2 * varphi ** 2],[-0.13e2 / 0.420e3 * 1.0 ** 2 * varphi ** 2,-varphi ** 3 * 1.0 ** 3 / 140,-0.11e2 / 0.210e3 * 1.0 ** 2 * varphi ** 2,varphi ** 3 * 1.0 ** 3 / 105]])
        VxVx=np.mat([[0.6e1 / 0.5e1 / varphi * 1.0,1.0 ** 2 / 10,-0.6e1 / 0.5e1 / varphi * 1.0,1.0 ** 2 / 10],[1.0 ** 2 / 10,0.2e1 / 0.15e2 * varphi * 1.0 ** 3,-1.0 ** 2 / 10,-varphi * 1.0 ** 3 / 30],[-0.6e1 / 0.5e1 / varphi * 1.0,-1.0 ** 2 / 10,0.6e1 / 0.5e1 / varphi * 1.0,-1.0 ** 2 / 10],[1.0 ** 2 / 10,-varphi * 1.0 ** 3 / 30,-1.0 ** 2 / 10,0.2e1 / 0.15e2 * varphi * 1.0 ** 3]])
        #mode_1_1
        A_VxVx=np.mat([[0.37e2 / 0.10e2 / varphi * 1.0,-0.189e3 / 0.40e2 / varphi * 1.0,0.27e2 / 0.20e2 / varphi * 1.0,-0.13e2 / 0.40e2 / varphi * 1.0],[-0.189e3 / 0.40e2 / varphi * 1.0,0.54e2 / 0.5e1 / varphi * 1.0,-0.297e3 / 0.40e2 / varphi * 1.0,0.27e2 / 0.20e2 / varphi * 1.0],[0.27e2 / 0.20e2 / varphi * 1.0,-0.297e3 / 0.40e2 / varphi * 1.0,0.54e2 / 0.5e1 / varphi * 1.0,-0.189e3 / 0.40e2 / varphi * 1.0],[-0.13e2 / 0.40e2 / varphi * 1.0,0.27e2 / 0.20e2 / varphi * 1.0,-0.189e3 / 0.40e2 / varphi * 1.0,0.37e2 / 0.10e2 / varphi * 1.0]])
        A_VV=np.mat([[0.8e1 / 0.105e3 * varphi * 1.0,0.33e2 / 0.560e3 * varphi * 1.0,-0.3e1 / 0.140e3 * varphi * 1.0,0.19e2 / 0.1680e4 * varphi * 1.0],[0.33e2 / 0.560e3 * varphi * 1.0,0.27e2 / 0.70e2 * varphi * 1.0,-0.27e2 / 0.560e3 * varphi * 1.0,-0.3e1 / 0.140e3 * varphi * 1.0],[-0.3e1 / 0.140e3 * varphi * 1.0,-0.27e2 / 0.560e3 * varphi * 1.0,0.27e2 / 0.70e2 * varphi * 1.0,0.33e2 / 0.560e3 * varphi * 1.0],[0.19e2 / 0.1680e4 * varphi * 1.0,-0.3e1 / 0.140e3 * varphi * 1.0,0.33e2 / 0.560e3 * varphi * 1.0,0.8e1 / 0.105e3 * varphi * 1.0]])
 
        element_linear_mat_local=np.zeros((Dof.Element(Dof(self.mode_list,self.mat_csi,0)),Dof.Element(Dof(self.mode_list,self.mat_csi,0))),dtype=float)
        #mode_1_highmode
        C1_shHTxxTxshL=np.mat([[0.9e1 / 0.2e1 / varphi ** 2 * 1.0,-0.9e1 / 0.2e1 / varphi ** 2 * 1.0,-0.9e1 / 0.2e1 / varphi ** 2 * 1.0,0.9e1 / 0.2e1 / varphi ** 2 * 1.0],[0.13e2 / 0.4e1 / varphi * 1.0 ** 2,-0.9e1 / 0.4e1 / varphi * 1.0 ** 2,-0.9e1 / 0.4e1 / varphi * 1.0 ** 2,0.5e1 / 0.4e1 / varphi * 1.0 ** 2],[-0.9e1 / 0.2e1 / varphi ** 2 * 1.0,0.9e1 / 0.2e1 / varphi ** 2 * 1.0,0.9e1 / 0.2e1 / varphi ** 2 * 1.0,-0.9e1 / 0.2e1 / varphi ** 2 * 1.0],[0.5e1 / 0.4e1 / varphi * 1.0 ** 2,-0.9e1 / 0.4e1 / varphi * 1.0 ** 2,-0.9e1 / 0.4e1 / varphi * 1.0 ** 2,0.13e2 / 0.4e1 / varphi * 1.0 ** 2]])
        C2_shHTxxTshL=np.mat([[-0.11e2 / 0.20e2 / varphi * 1.0,-0.27e2 / 0.20e2 / varphi * 1.0,0.27e2 / 0.20e2 / varphi * 1.0,0.11e2 / 0.20e2 / varphi * 1.0],[-0.2e1 / 0.5e1 * 1.0 ** 2,-0.21e2 / 0.20e2 * 1.0 ** 2,0.3e1 / 0.10e2 * 1.0 ** 2,0.3e1 / 0.20e2 * 1.0 ** 2],[0.11e2 / 0.20e2 / varphi * 1.0,0.27e2 / 0.20e2 / varphi * 1.0,-0.27e2 / 0.20e2 / varphi * 1.0,-0.11e2 / 0.20e2 / varphi * 1.0],[-0.3e1 / 0.20e2 * 1.0 ** 2,-0.3e1 / 0.10e2 * 1.0 ** 2,0.21e2 / 0.20e2 * 1.0 ** 2,0.2e1 / 0.5e1 * 1.0 ** 2]])
        C3_shHTTxshL=np.mat([[-0.19e2 / 0.20e2 * 1.0,0.9e1 / 0.20e2 * 1.0,0.9e1 / 0.20e2 * 1.0,1.0 / 20],[-varphi * 1.0 ** 2 / 12,-0.3e1 / 0.40e2 * varphi * 1.0 ** 2,0.3e1 / 0.20e2 * varphi * 1.0 ** 2,varphi * 1.0 ** 2 / 120],[-1.0 / 20,-0.9e1 / 0.20e2 * 1.0,-0.9e1 / 0.20e2 * 1.0,0.19e2 / 0.20e2 * 1.0],[varphi * 1.0 ** 2 / 120,0.3e1 / 0.20e2 * varphi * 1.0 ** 2,-0.3e1 / 0.40e2 * varphi * 1.0 ** 2,-varphi * 1.0 ** 2 / 12]])
        C4_B_shHTTshL=np.mat([[0.4e1 / 0.35e2 * varphi * 1.0,0.93e2 / 0.280e3 * varphi * 1.0,0.3e1 / 0.70e2 * varphi * 1.0,0.3e1 / 0.280e3 * varphi * 1.0],[1.0 ** 2 * varphi ** 2 / 140,0.3e1 / 0.56e2 * 1.0 ** 2 * varphi ** 2,0.3e1 / 0.140e3 * 1.0 ** 2 * varphi ** 2,1.0 ** 2 * varphi ** 2 / 840],[0.3e1 / 0.280e3 * varphi * 1.0,0.3e1 / 0.70e2 * varphi * 1.0,0.93e2 / 0.280e3 * varphi * 1.0,0.4e1 / 0.35e2 * varphi * 1.0],[-1.0 ** 2 * varphi ** 2 / 840,-0.3e1 / 0.140e3 * 1.0 ** 2 * varphi ** 2,-0.3e1 / 0.56e2 * 1.0 ** 2 * varphi ** 2,-1.0 ** 2 * varphi ** 2 / 140]])
        D_shHTxTshL=np.mat([[-1.0 / 20,-0.9e1 / 0.20e2 * 1.0,-0.9e1 / 0.20e2 * 1.0,-1.0 / 20],[varphi * 1.0 ** 2 / 12,0.3e1 / 0.40e2 * varphi * 1.0 ** 2,-0.3e1 / 0.20e2 * varphi * 1.0 ** 2,-varphi * 1.0 ** 2 / 120],[1.0 / 20,0.9e1 / 0.20e2 * 1.0,0.9e1 / 0.20e2 * 1.0,1.0 / 20],[-varphi * 1.0 ** 2 / 120,-0.3e1 / 0.20e2 * varphi * 1.0 ** 2,0.3e1 / 0.40e2 * varphi * 1.0 ** 2,varphi * 1.0 ** 2 / 12]])

        #mode_highmode_1
        shHTxxTxshL=np.mat([[0.9e1 / 0.2e1 / varphi ** 2 * 1.0,-0.9e1 / 0.2e1 / varphi ** 2 * 1.0,-0.9e1 / 0.2e1 / varphi ** 2 * 1.0,0.9e1 / 0.2e1 / varphi ** 2 * 1.0],[0.13e2 / 0.4e1 / varphi * 1.0 ** 2,-0.9e1 / 0.4e1 / varphi * 1.0 ** 2,-0.9e1 / 0.4e1 / varphi * 1.0 ** 2,0.5e1 / 0.4e1 / varphi * 1.0 ** 2],[-0.9e1 / 0.2e1 / varphi ** 2 * 1.0,0.9e1 / 0.2e1 / varphi ** 2 * 1.0,0.9e1 / 0.2e1 / varphi ** 2 * 1.0,-0.9e1 / 0.2e1 / varphi ** 2 * 1.0],[0.5e1 / 0.4e1 / varphi * 1.0 ** 2,-0.9e1 / 0.4e1 / varphi * 1.0 ** 2,-0.9e1 / 0.4e1 / varphi * 1.0 ** 2,0.13e2 / 0.4e1 / varphi * 1.0 ** 2]])

        for i in range(len(self.ik_vector)):
            [k1,k2]=(self.ik_vector[i,0:2]).astype(str)
            dim_1=(self.ik_vector[i,12:16]).astype(int)
            dim_2=(self.ik_vector[i,16:20]).astype(int)


            [C1,C2,C3,C4,B_t,G_s,D_1mu,D_2mu,D_3mu,D_4mu]=(self.ik_vector[i,2:12]).astype(float)  
        #    [C1,C2,C3,C4,B_t,G_s]=[0,0,0,0,0,0] #test
         #   print (k1,k2,C1,C2,C3,C4,B_t,G_s,D_1mu,D_2mu,D_3mu,D_4mu)
            if k1.split("|")[0]=='1' and k2.split("|")[0]=='1' :
            #    [C1,C2,C3,C4,B_t,G_s,D_1mu,D_2mu,D_3mu,D_4mu]=[0,0,0,0,0,0,0,0,0,0]
                element_linear_mat_local[np.tile(dim_1, (4, 1)),np.tile(dim_2, (4, 1)).T]+=self.E* C1*A_VxVx + self.G*G_s*A_VV
            
            if k1.split("|")[0]=='1' and k2.split("|")[0]!='1' :
             #   [C1,C2,C3,C4,B_t,G_s,D_3mu,D_4mu]=[0,0,0,0,0,0,0,0]
                element_linear_mat_local[np.tile(dim_1, (4, 1)),np.tile(dim_2, (4, 1)).T]+=self.E* (C1*C1_shHTxxTxshL + C2*C2_shHTxxTshL + C3*C3_shHTTxshL + C4* C4_B_shHTTshL  + B_t* C4_B_shHTTshL + D_2mu*C3_shHTTxshL)+self.G*G_s*D_shHTxTshL

            if k1.split("|")[0]!='1' and k2.split("|")[0]=='1' :
             #   [C1,C2,C3,C4,B_t,G_s,D_3mu,D_4mu]=[0,0,0,0,0,0,0,0]
                element_linear_mat_local[np.tile(dim_1, (4, 1)),np.tile(dim_2, (4, 1)).T]+=self.E* (C1*np.transpose(C1_shHTxxTxshL) + C2*np.transpose(C3_shHTTxshL) + C3*np.transpose(C2_shHTxxTshL) + C4* np.transpose(C4_B_shHTTshL)  + B_t* np.transpose(C4_B_shHTTshL ) +  D_1mu* np.transpose(C3_shHTTxshL) )+self.G*G_s*np.transpose(D_shHTxTshL)
            
            
            if k1.split("|")[0]!='1' and k2.split("|")[0]!='1' :
               # [C1,C2,C3,C4,B_t,G_s,D_1mu,D_2mu,D_3mu,D_4mu]=[0,0,0,0,0,0,0,0,0,0]
                element_linear_mat_local[np.tile(dim_1, (4, 1)),np.tile(dim_2, (4, 1)).T]+=self.E* (C1*VxxVxx + C2*VxxV + C3*VVxx + C4*VV  + B_t*VV + D_1mu*VxxV + D_2mu*VVxx + VV*(D_3mu+D_4mu) )+self.G*G_s*VxVx
                
        return element_linear_mat_local*self.k_e
    
class Elastic_stiffness_matrix_GBT:
    # combined elastic stiffness matrix 
    def __init__ (self,L_cur,Lo,r,t,E,spring_stiffness,mu,mode_list,mat_csi,k_e):
        self.l=Lo
        self.mode_list=mode_list
        self.E=E
        self.spring_stiffness=spring_stiffness
        self.r=r
        self.t=t
        self.mu=mu
        self.L_cur=L_cur
        self.k_e=k_e
        self.mode_list_size=len(mode_list)
        self.mat_csi=mat_csi
        self.G=self.E/(2.0*(1.0+self.mu))
    def element(self):
        element_elastic_mat_local=np.zeros((Dof.Element(Dof(self.mode_list,self.mat_csi,0)),Dof.Element(Dof(self.mode_list,self.mat_csi,0))),float)
        L= self.l
        count=0
        for i in range(self.mode_list_size):
            self.k=self.mode_list[i]
            if self.k.split("|")[0]=='t' or  self.k.split("|")[0]=='a' or  self.k.split("|")[0]=='1' :
                m=0
            else:
                m=self.mat_csi[3+(int(self.k.split("|")[0])-2)*3,1]
            self.C_k=Stiffness_coff.stiffness_c(Stiffness_coff(self.k,m,self.r,self.t, self.E, self.mu))
            self.B_k=Stiffness_coff.stiffness_b(Stiffness_coff(self.k,m,self.r,self.t, self.E, self.mu))#+#self.spring_stiffness[k,1]
            self.D_k=Stiffness_coff.stiffness_d(Stiffness_coff(self.k,m,self.r,self.t, self.E, self.mu))
            if self.k.split("|")[0]=='t':
                   element_elastic_mat_local[count:count+4,count:count+4]=(self.G*self.D_k)*np.mat([[0.6e1 / 0.5e1 / L,0.1e1 / 0.10e2,-0.6e1 / 0.5e1 / L,0.1e1 / 0.10e2],[0.1e1 / 0.10e2,0.2e1 / 0.15e2 * L,-0.1e1 / 0.10e2,-L / 30],[-0.6e1 / 0.5e1 / L,-0.1e1 / 0.10e2,0.6e1 / 0.5e1 / L,-0.1e1 / 0.10e2],[0.1e1 / 0.10e2,-L / 30,-0.1e1 / 0.10e2,0.2e1 / 0.15e2 * L]])
                   count+=4
            elif self.k.split("|")[0]=='a':
                   element_elastic_mat_local[count:count+4,count:count+4]=(((2*self.C_k/self.l**3)*np.matrix([[6, 3*self.l,-6,3*self.l],[3*self.l,2*self.l**2,-3*self.l,self.l**2],[-6,-3*self.l,6,-3*self.l],[3*self.l,self.l**2,-3*self.l,2*self.l**2]]))+\
                   ((self.B_k*self.l/420)*np.matrix([[156, 22*self.l, 54,-13*self.l],[22*self.l, 4*self.l**2, 13*self.l, -3*self.l**2],[54, 13*self.l, 156, -22*self.l],[-13*self.l, -3*self.l**2, -22*self.l, 4*self.l**2]])))
                   count+=4
            elif self.k.split("|")[0]=='1':                                     
                    #V,x
                   element_elastic_mat_local[count:count+4,count:count+4]=(self.E*self.C_k)*np.mat([[0.37e2 / 0.10e2 / L,-0.189e3 / 0.40e2 / L,0.27e2 / 0.20e2 / L,-0.13e2 / 0.40e2 / L],[-0.189e3 / 0.40e2 / L,0.54e2 / 0.5e1 / L,-0.297e3 / 0.40e2 / L,0.27e2 / 0.20e2 / L],[0.27e2 / 0.20e2 / L,-0.297e3 / 0.40e2 / L,0.54e2 / 0.5e1 / L,-0.189e3 / 0.40e2 / L],[-0.13e2 / 0.40e2 / L,0.27e2 / 0.20e2 / L,-0.189e3 / 0.40e2 / L,0.37e2 / 0.10e2 / L]])
                   count+=4
 
            elif int(self.k.split("|")[0]) > 1 and self.k.split("|")[1]== 'c' :
                   element_elastic_mat_local[count:count+4,count:count+4]=(((2*self.E*self.C_k/self.l**3)*np.matrix([[6, 3*self.l,-6,3*self.l],[3*self.l,2*self.l**2,-3*self.l,self.l**2],[-6,-3*self.l,6,-3*self.l],[3*self.l,self.l**2,-3*self.l,2*self.l**2]]))+\
                                                                               ((self.G*self.D_k/(30*self.l))*np.matrix([[36, 3*self.l,-36,3*self.l],[3*self.l,4*self.l**2,-3*self.l,-self.l**2],[-36,-3*self.l,36,-3*self.l],[3*self.l,-self.l**2,-3*self.l,4*self.l**2]]))+\
                                                                               ((self.B_k*self.l/420)*np.matrix([[156, 22*self.l, 54,-13*self.l],[22*self.l, 4*self.l**2, 13*self.l, -3*self.l**2],[54, 13*self.l, 156, -22*self.l],[-13*self.l, -3*self.l**2, -22*self.l, 4*self.l**2]])))
                   count +=4                
            elif int(self.k.split("|")[0]) > 1 and self.k.split("|")[1]== 'v' :      
                   self.A_s=m**2*self.t*math.pi*(16*self.r**2+3*self.t**2)/(16*self.r)
                 
                   self.A_teta=m**4*self.t*math.pi*(12*self.r**2+self.t**2)/(12*self.r**3)

                   element_elastic_mat_local[count:count+4,count:count+4]=(((self.G*self.A_s/(30*self.l))*np.matrix([[36, 3*self.l,-36,3*self.l],[3*self.l,4*self.l**2,-3*self.l,-self.l**2],[-36,-3*self.l,36,-3*self.l],[3*self.l,-self.l**2,-3*self.l,4*self.l**2]]))+\
                                                                               ((self.E*self.A_teta*self.l/420)*np.matrix([[156, 22*self.l, 54,-13*self.l],[22*self.l, 4*self.l**2, 13*self.l, -3*self.l**2],[54, 13*self.l, 156, -22*self.l],[-13*self.l, -3*self.l**2, -22*self.l, 4*self.l**2]])))
                   #coupling classical and shear_v
                   self.A_s_coupling=-m**2*self.t**3*math.pi*(m**2-1)/(4*self.r) 
                                                                              
                   self.A_teta_coupling=-m**4*self.t**3*math.pi*(m**2-1)/(12*self.r**3)
                                                                               
                   element_elastic_mat_local[count-4:count,count:count+4]=(((self.G*self.A_s_coupling/(30*self.l))*np.matrix([[36, 3*self.l,-36,3*self.l],[3*self.l,4*self.l**2,-3*self.l,-self.l**2],[-36,-3*self.l,36,-3*self.l],[3*self.l,-self.l**2,-3*self.l,4*self.l**2]]))+\
                                                                               ((self.E*self.A_teta_coupling*self.l/420)*np.matrix([[156, 22*self.l, 54,-13*self.l],[22*self.l, 4*self.l**2, 13*self.l, -3*self.l**2],[54, 13*self.l, 156, -22*self.l],[-13*self.l, -3*self.l**2, -22*self.l, 4*self.l**2]])))

                   element_elastic_mat_local[count:count+4,count-4:count]=(((self.G*self.A_s_coupling/(30*self.l))*np.matrix([[36, 3*self.l,-36,3*self.l],[3*self.l,4*self.l**2,-3*self.l,-self.l**2],[-36,-3*self.l,36,-3*self.l],[3*self.l,-self.l**2,-3*self.l,4*self.l**2]]))+\
                                                                               ((self.E*self.A_teta_coupling*self.l/420)*np.matrix([[156, 22*self.l, 54,-13*self.l],[22*self.l, 4*self.l**2, 13*self.l, -3*self.l**2],[54, 13*self.l, 156, -22*self.l],[-13*self.l, -3*self.l**2, -22*self.l, 4*self.l**2]])))

                   count +=4                
                
            elif int(self.k.split("|")[0]) > 1 and self.k.split("|")[1]== 'u' :                  

                   self.A_x=self.t*math.pi*self.r**3
                   
                   self.A_s_u=self.t*m**2*math.pi*(48*self.r**2 + self.t**2)/(48*self.r)
                   
                   


                   element_elastic_mat_local[count:count+4,count:count+4]=(((2*self.E*self.A_x/self.l**3)*np.matrix([[6, 3*self.l,-6,3*self.l],[3*self.l,2*self.l**2,-3*self.l,self.l**2],[-6,-3*self.l,6,-3*self.l],[3*self.l,self.l**2,-3*self.l,2*self.l**2]]))+\
                                                                               ((self.G*self.A_s_u/(30*self.l))*np.matrix([[36, 3*self.l,-36,3*self.l],[3*self.l,4*self.l**2,-3*self.l,-self.l**2],[-36,-3*self.l,36,-3*self.l],[3*self.l,-self.l**2,-3*self.l,4*self.l**2]])))
                   #coupling classical and shear_u
                   self.A_s_coupling=-m**2*self.t**3*math.pi*(m**2-1)/(12*self.r)
                   
                   element_elastic_mat_local[count-8:count-4,count:count+4]=(((2*self.E*self.A_x/self.l**3)*np.matrix([[6, 3*self.l,-6,3*self.l],[3*self.l,2*self.l**2,-3*self.l,self.l**2],[-6,-3*self.l,6,-3*self.l],[3*self.l,self.l**2,-3*self.l,2*self.l**2]]))+\
                                                                               ((self.G*self.A_s_coupling/(30*self.l))*np.matrix([[36, 3*self.l,-36,3*self.l],[3*self.l,4*self.l**2,-3*self.l,-self.l**2],[-36,-3*self.l,36,-3*self.l],[3*self.l,-self.l**2,-3*self.l,4*self.l**2]])))

                   element_elastic_mat_local[count:count+4,count-8:count-4]=(((2*self.E*self.A_x/self.l**3)*np.matrix([[6, 3*self.l,-6,3*self.l],[3*self.l,2*self.l**2,-3*self.l,self.l**2],[-6,-3*self.l,6,-3*self.l],[3*self.l,self.l**2,-3*self.l,2*self.l**2]]))+\
                                                                               ((self.G*self.A_s_coupling/(30*self.l))*np.matrix([[36, 3*self.l,-36,3*self.l],[3*self.l,4*self.l**2,-3*self.l,-self.l**2],[-36,-3*self.l,36,-3*self.l],[3*self.l,-self.l**2,-3*self.l,4*self.l**2]])))
                   #coupling shear_v and shear_u
                   self.A_s_vu=-m**2*self.t*math.pi*(16*self.r**2-self.t**2)/(16*self.r)
                                                            
                   element_elastic_mat_local[count-4:count,count:count+4]=((self.G*self.A_s_vu/(30*self.l))*np.matrix([[36, 3*self.l,-36,3*self.l],[3*self.l,4*self.l**2,-3*self.l,-self.l**2],[-36,-3*self.l,36,-3*self.l],[3*self.l,-self.l**2,-3*self.l,4*self.l**2]]))

                   element_elastic_mat_local[count:count+4,count-4:count]=((self.G*self.A_s_vu/(30*self.l))*np.matrix([[36, 3*self.l,-36,3*self.l],[3*self.l,4*self.l**2,-3*self.l,-self.l**2],[-36,-3*self.l,36,-3*self.l],[3*self.l,-self.l**2,-3*self.l,4*self.l**2]]))
                                                                               
                   count +=4



            
        return element_elastic_mat_local*self.k_e 



#_____________________________________________________________            


          
#_____________________________________________________________
class Elastic_internal_force_GBT:
    # combined elastic stiffness matrix
    #u1,w1,teta1,u2,w2,teta2 
    def __init__ (self,L_cur,Lo,r,t,E,spring_stiffness,u_element,mu,mode_list,mat_csi,int_e):
        self.L_cur=L_cur
        self.Lo=Lo
        self.mode_list=mode_list
        self.E=E
        self.spring_stiffness=spring_stiffness
        self.r=r
        self.t=t
        self.u=u_element
        self.mu=mu
        self.mode_list_size=len(mode_list)
        self.mat_csi=mat_csi
        self.int_e=int_e
        self.k_elastic=Elastic_stiffness_matrix_GBT.element(Elastic_stiffness_matrix_GBT( L_cur,Lo,r,t,E,spring_stiffness,mu,mode_list,mat_csi,1))
    def element(self) :
        return (np.dot(self.k_elastic,self.u))

#_____________________________________________________________

class Rotation_matrix_TL:   # independent 

    def __init__ (self,rotation):
        self.rotation=rotation

    def transformation_matrix(self):
        c=math.cos(self.rotation)#+ math.pi/2)
        s=math.sin(self.rotation)#+ math.pi/2)
       # return np.matrix([[c,-s,0,0,0,0],[s,c,0,0,0,0],[0,0,1,0,0,0],[0,0,0,c,-s,0],[0,0,0,s,c,0],[0,0,0,0,0,1]])
        return np.matrix([[c,0,s,0,0,0],[0,c,0,0,s,0],[-s,0,c,0,0,0],[0,0,0,1,0,0],[0,-s,0,0,c,0],[0,0,0,0,0,1]])
#_____________________________________________________________            

class Update_element_TL:
    def __init__ (self, u, no_beam_elem, x, y, A, E, I, LM_matrix):
        self.u=u
        self.x=x
        self.y=y
        self.E=E
        self.I=I
        self.A=A
        self.no_beam_elem=no_beam_elem 
        self.LM_matrix=LM_matrix
        self.EA=E*A
        self.EI=E*I
    def update_all(self):
        X_delta=np.zeros((self.no_beam_elem),float)
        Y_delta=np.zeros((self.no_beam_elem),float)
        L_cur=np.zeros((self.no_beam_elem),float)
        Lo=np.zeros((self.no_beam_elem),float)
        beta=np.zeros((self.no_beam_elem),float)
        beta_o=np.zeros((self.no_beam_elem),float)
        incre=0
        for i in range(0,self.no_beam_elem):
            #print incre
            #print i
            beta_o[i] = math.atan2( ( self.y[i+1]- self.y[i] ),(self.x[i+1]-self.x[i]))

            Lo[i] = math.sqrt(( self.y[i+1]- self.y[i] )**2+(self.x[i+1]-self.x[i])**2 )
            X_delta[i]=self.x[i+1]+self.u[3+incre]-self.x[i]-self.u[incre]
          #  print X_delta
            Y_delta[i]=self.y[i+1]+self.u[4+incre]-self.y[i]-self.u[1+incre]
          #  print Y_delta
            L_cur[i]=math.sqrt(X_delta[i]**2+Y_delta[i]**2)
    
            beta[i]=math.atan2(Y_delta[i],X_delta[i])

            incre +=3
        
        return L_cur,Lo,beta 



#_____________________________________________________________  

class Load_shared_mode:
    def __init__ (self, l, k, m ,r, t, q, s_start,s_end, num_modes, type_of_load,gravity,density,xi,R):
        self.k=k
        self.vertheta=Symbol('vertheta')
        self.i=GBT_func_curve(k,r,self.vertheta,R) 
        self.xi=(1+(xi*(r/R)*cos(self.vertheta)))
        self.m=m
        self.r=r
        self.t=t
        self.q=q 
        self.l=l
        self.s_start=s_start
        self.s_end=s_end
        self.num_modes=num_modes
        self.type_of_load=type_of_load
        self.density=density
        self.gravity=gravity
        self.mat_csi=np.zeros((self.num_modes-1,5),object)
    """ load participation"""
    def proj_load_y(self):

         if self.k.split("|")[0]=='t':
            return self.r*self.q*((quad(lambda vertheta: self.r*np.sin(vertheta)*np.cos(vertheta),self.s_start,self.s_end))[0]) 
         elif self.k.split("|")[0]=='a':
            return 0 
         elif self.k.split("|")[0]=='1':
            return 0 

         elif int(self.k.split("|")[0])  % 2 == 0  and self.k.split("|")[1]== 'c' :
            return self.r*self.q*self.m*((quad(lambda vertheta: -np.cos(self.m*vertheta)*np.sin(vertheta)*np.cos(vertheta),self.s_start,self.s_end))[0])
         elif int(self.k.split("|")[0])  % 2 != 0  and self.k.split("|")[1]== 'c' :
            return self.r*self.q*self.m*((quad(lambda vertheta: -np.sin(self.m*vertheta)*np.sin(vertheta)*np.cos(vertheta),self.s_start,self.s_end))[0])

         elif int(self.k.split("|")[0])  % 2 == 0  and self.k.split("|")[1]== 'v' :
            return self.r*self.q*self.m*((quad(lambda vertheta: -np.cos(self.m*vertheta)*np.sin(vertheta)*np.cos(vertheta),self.s_start,self.s_end))[0])
         elif int(self.k.split("|")[0])  % 2 != 0  and self.k.split("|")[1]== 'v' :
            return self.r*self.q*self.m*((quad(lambda vertheta: -np.sin(self.m*vertheta)*np.sin(vertheta)*np.cos(vertheta),self.s_start,self.s_end))[0])

#         if self.k % 2 == 0 and self.k<=-2 :
#            return self.r*self.q*((quad(lambda vertheta: np.cos(self.m*vertheta)*np.sin(vertheta)*np.cos(vertheta)/self.m,self.s_start,self.s_end))[0])
#         if self.k % 2 != 0 and self.k<=-2 :
#            return self.r*self.q*((quad(lambda vertheta: -np.sin(self.m*vertheta)*np.sin(vertheta)*np.cos(vertheta)/self.m,self.s_start,self.s_end))[0])


    def proj_load_z(self):
         if self.k.split("|")[0]=='t':
            return 0  
       
         elif self.k.split("|")[0]=='a':
            return self.r*self.q*((quad(lambda vertheta:1*-np.cos(vertheta)*np.cos(vertheta),self.s_start,self.s_end))[0]) 
         elif self.k.split("|")[0]=='1':
            return 0     
        
         elif int(self.k.split("|")[0])  % 2 == 0  and self.k.split("|")[1]== 'c' :
            return self.r*self.q*self.m**2*((quad(lambda vertheta:-np.sin(self.m*vertheta)*-np.cos(vertheta)*np.cos(vertheta),self.s_start,self.s_end))[0])
         elif int(self.k.split("|")[0])  % 2 != 0  and self.k.split("|")[1]== 'c' :
            return self.r*self.q*self.m**2*((quad(lambda vertheta:np.cos(self.m*vertheta)*-np.cos(vertheta)*np.cos(vertheta),self.s_start,self.s_end))[0])
         elif int(self.k.split("|")[0])  % 2 == 0  and self.k.split("|")[1]== 'v' :
            return self.r*self.q*self.m**2*0*((quad(lambda vertheta:-np.sin(self.m*vertheta)*-np.cos(vertheta)*np.cos(vertheta),self.s_start,self.s_end))[0])
         elif int(self.k.split("|")[0])  % 2 != 0  and self.k.split("|")[1]== 'v' :
            return self.r*self.q*self.m**2*0*((quad(lambda vertheta:np.cos(self.m*vertheta)*-np.cos(vertheta)*np.cos(vertheta),self.s_start,self.s_end))[0])
#         if self.k % 2 == 0 and self.k<=-2 :
#            return self.r*self.q*((quad(lambda vertheta:np.sin(self.m*vertheta)*-np.cos(vertheta)*np.cos(vertheta)/self.m**2,self.s_start,self.s_end))[0])
#         if self.k % 2 != 0 and self.k<=-2 :
#            return self.r*self.q*((quad(lambda vertheta:np.cos(self.m*vertheta)*-np.cos(vertheta)*np.cos(vertheta)/self.m**2,self.s_start,self.s_end))[0])

    def press_load_z(self):
         func = lambdify((self.vertheta),self.i.warping_fun_w(), 'numpy')
         
         return self.r*self.q*((quad(lambda vertheta:func(vertheta),self.s_start,self.s_end))[0])
        
#          if self.k.split("|")[0]=='t':
#             return 0  
#          elif self.k.split("|")[0]=='a': 
#             return self.r*self.q*((quad(lambda vertheta:1,self.s_start,self.s_end))[0])    
#          elif self.k.split("|")[0]=='1':
#             return 0            
#          elif int(self.k.split("|")[0])  % 2 == 0  and self.k.split("|")[1]== 'c' :
            
#          elif int(self.k.split("|")[0])  % 2 != 0  and self.k.split("|")[1]== 'c' :
#             return self.r*self.q*self.m**2*((quad(lambda vertheta:func(vertheta),self.s_start,self.s_end))[0])
# #         if self.k % 2 == 0 and self.k<=-2  :
# #            return self.r*self.q*self.m**2*((quad(lambda vertheta:-np.sin(self.m*vertheta),self.s_start,self.s_end))[0])
# #         if self.k % 2 != 0 and self.k<=-2  :
# #            return self.r*self.q*self.m**2*((quad(lambda vertheta:np.cos(self.m*vertheta),self.s_start,self.s_end))[0])
#          elif int(self.k.split("|")[0])  % 2 == 0  and self.k.split("|")[1]== 'v' :
#              return self.r*self.q*((quad(lambda vertheta:func(vertheta),self.s_start,self.s_end))[0])
#          elif int(self.k.split("|")[0])  % 2 != 0  and self.k.split("|")[1]== 'v' :
#              return self.r*self.q*((quad(lambda vertheta:func(vertheta),self.s_start,self.s_end))[0])

 
    def vertical_load_x(self):
        return self.density*self.gravity*2*math.pi*self.r*self.l*self.t
    @staticmethod        
    def load_contr(l,r, t, q, s_start,s_end, num_modes, type_of_load,gravity,density,xi,R):    
        mat_csi=np.zeros((3*num_modes,5),object)
        kk=1
        for i in range(0,3,1) :
            if i==0: #torsion
               mat_csi[i,0]='t|t'
               m=0
            elif i==1: #axisymmetric
               mat_csi[i,0]='a|a'
               m=0
            elif i==2:  #axial
               mat_csi[i,0]='1|1'
               m=0
            elif i>=3:
                kk+=1 
                if kk % 2 == 0:
                    m=kk/2
                else:
                    m=(kk-1)/2
                mat_csi[i,0]=str(kk)+'|c'
                mat_csi[i+1,0]=str(kk)+'|v'
                mat_csi[i+2,0]=str(kk)+'|u'
                mat_csi[i:i+3,1]=m
                
            if type_of_load=='projected':
               mat_csi[i,2]=Load_shared_mode.proj_load_y(Load_shared_mode(l, mat_csi[i,0], m ,r, t, q, s_start,s_end, num_modes, type_of_load,gravity,density,xi,R))
               mat_csi[i+1,2]=Load_shared_mode.proj_load_y(Load_shared_mode(l, str(kk)+'|v', m ,r, t, q, s_start,s_end, num_modes, type_of_load,gravity,density,xi,R))
               mat_csi[i+2,2]=Load_shared_mode.proj_load_y(Load_shared_mode(l, str(kk)+'|v', m ,r, t, 0, s_start,s_end, num_modes, type_of_load,gravity,density,xi,R))

               mat_csi[i,3]=Load_shared_mode.proj_load_z(Load_shared_mode(l, mat_csi[i,0], m ,r, t, q, s_start,s_end, num_modes, type_of_load,gravity,density,xi,R))
               mat_csi[i+1,3]=Load_shared_mode.proj_load_z(Load_shared_mode(l, str(kk)+'|v', m ,r, t, q, s_start,s_end, num_modes, type_of_load,gravity,density,xi,R))
               mat_csi[i+2,3]=Load_shared_mode.proj_load_z(Load_shared_mode(l, str(kk)+'|v', m ,r, t, 0, s_start,s_end, num_modes, type_of_load,gravity,density,xi,R))

            else :
               mat_csi[i,3]=Load_shared_mode.press_load_z(Load_shared_mode(l ,mat_csi[i,0], m ,r, t, q, s_start,s_end, num_modes, type_of_load,gravity,density,xi,R))
               mat_csi[i+1,3]=Load_shared_mode.press_load_z(Load_shared_mode(l ,str(kk)+'|v', m ,r, t, 0, s_start,s_end, num_modes, type_of_load,gravity,density,xi,R))
               mat_csi[i+2,3]=Load_shared_mode.press_load_z(Load_shared_mode(l ,str(kk)+'|v', m ,r, t, 0, s_start,s_end, num_modes, type_of_load,gravity,density,xi,R))
        for i in range(3,3*num_modes,3):
            if i==0: #torsion
               mat_csi[i,0]='t|t'
               m=0
            elif i==1: #axisymmetric
               mat_csi[i,0]='a|a'
               m=0
            elif i==2:  #axial
               mat_csi[i,0]='1|1'
               m=0
            elif i>=3:
                kk+=1 
                if kk % 2 == 0:
                    m=kk/2
                else:
                    m=(kk-1)/2
                mat_csi[i,0]=str(kk)+'|c'
                mat_csi[i+1,0]=str(kk)+'|v'
                mat_csi[i+2,0]=str(kk)+'|u'
                mat_csi[i:i+3,1]=m
                
            if type_of_load=='projected':
               mat_csi[i,2]=Load_shared_mode.proj_load_y(Load_shared_mode(l, mat_csi[i,0], m ,r, t, q, s_start,s_end, num_modes, type_of_load,gravity,density,xi,R))
               mat_csi[i+1,2]=Load_shared_mode.proj_load_y(Load_shared_mode(l, str(kk)+'|v', m ,r, t, q, s_start,s_end, num_modes, type_of_load,gravity,density,xi,R))
               mat_csi[i+2,2]=Load_shared_mode.proj_load_y(Load_shared_mode(l, str(kk)+'|v', m ,r, t, 0, s_start,s_end, num_modes, type_of_load,gravity,density,xi,R))

               mat_csi[i,3]=Load_shared_mode.proj_load_z(Load_shared_mode(l, mat_csi[i,0], m ,r, t, q, s_start,s_end, num_modes, type_of_load,gravity,density,xi,R))
               mat_csi[i+1,3]=Load_shared_mode.proj_load_z(Load_shared_mode(l, str(kk)+'|v', m ,r, t, q, s_start,s_end, num_modes, type_of_load,gravity,density,xi,R))
               mat_csi[i+2,3]=Load_shared_mode.proj_load_z(Load_shared_mode(l, str(kk)+'|v', m ,r, t, 0, s_start,s_end, num_modes, type_of_load,gravity,density,xi,R))

            else :
               mat_csi[i,3]=Load_shared_mode.press_load_z(Load_shared_mode(l ,mat_csi[i,0], m ,r, t, q, s_start,s_end, num_modes, type_of_load,gravity,density,xi,R))
               mat_csi[i+1,3]=Load_shared_mode.press_load_z(Load_shared_mode(l ,str(kk)+'|v', m ,r, t, q, s_start,s_end, num_modes, type_of_load,gravity,density,xi,R))
               mat_csi[i+2,3]=Load_shared_mode.press_load_z(Load_shared_mode(l ,str(kk)+'|v', m ,r, t, 0, s_start,s_end, num_modes, type_of_load,gravity,density,xi,R))


        """ vertical load and self weight """ 
        #self weight... concentrated at node  
 
#        mat_csi[6,4]=-Load_shared_mode.vertical_load_x(Load_shared_mode(l, k, m ,r, t, q, s_start,s_end, num_modes, type_of_load,gravity,density))
        mode_list=[]
#         mat_csi=mat_csi.astype('float')
# #        if num_modes>9:
# #           mat_csi[8,4]=0.005
#         #mat_csi[12,4]=0.005
#         j=0
#         for i in range(0,num_modes): 
#             if  mat_csi[i,2] > 0.000001 or  mat_csi[i,4] != 0 or mat_csi[i,3] >0.000001  or mat_csi[i,3] < -0.000001 or mat_csi[i,2] < -0.000001  :
#                 mode_list.append(j)
#                 mode_list[j]=mat_csi[i,0]
#                 j+=1
       
        return mat_csi, mode_list


#_____________________________________________________________  
        
class loading:
    def __init__ (self,mode_list,mat_csi,l):
        self.mode_list_size=len(mode_list)
#        self.no_beam_elem=no_beam_elem
        self.mode_list=mode_list
        self.mat_csi=mat_csi
        self.l=l
    def element(self):
        f_ext_element=np.zeros((Dof.Element(Dof(self.mode_list,self.mat_csi,0))),float)
        count=0
        for j in range(self.mode_list_size):
            k=self.mode_list[j]
            if k.split("|")[0]=='t':
               f_ext_element[0]=self.mat_csi[0,2]+self.mat_csi[0,3] 
               f_ext_element[1]=0
               count+=4              
            elif k.split("|")[0]=='a':
               q_tot=(self.mat_csi[1,2]+self.mat_csi[1,3])
               f_ext_element[count:count+4]=[q_tot*self.l/2,q_tot*self.l*self.l/12,q_tot*self.l/2,(-self.l*self.l*q_tot)/12]
               count+=4
            elif k.split("|")[0]=='1':
               f_ext_element[count]=(self.mat_csi[2,2]+self.mat_csi[2,3])*self.l/2 
               f_ext_element[count+1]=(self.mat_csi[2,2]+self.mat_csi[2,3])*self.l/2
               count+=4
            elif k.split("|")[1]== 'c' :
               q_tot=self.mat_csi[3+(int(k.split("|")[0])-2)*3,2]+self.mat_csi[3+(int(k.split("|")[0])-2)*3,3]
               f_ext_element[count:count+4]=[q_tot*self.l/2,q_tot*self.l*self.l/12,q_tot*self.l/2,(-self.l*self.l*q_tot)/12]
               count+=4
            elif k.split("|")[1]== 'v' :
               q_w=self.mat_csi[4+(int(k.split("|")[0])-2)*3,3]+self.mat_csi[4+(int(k.split("|")[0])-2)*3,2]
               f_ext_element[count:count+4]=[q_w*self.l/2,q_w*self.l*self.l/12,q_w*self.l/2,(-self.l*self.l*q_w)/12] 
               count+=4
            elif k.split("|")[1]== 'u' :
               q_w=self.mat_csi[5+(int(k.split("|")[0])-2)*3,3]+self.mat_csi[5+(int(k.split("|")[0])-2)*3,2]
               f_ext_element[count:count+4]=[q_w*self.l/2,q_w*self.l*self.l/12,q_w*self.l/2,(-self.l*self.l*q_w)/12]
               count+=4
#               print f_ext_element
        return f_ext_element

#_____________________________________________________________  
        
class loading_cuved_local_w_loading:
    def __init__ (self,R,r,varphi,mode_list,q,LM_element,start_vertheta,end_vertheta,xi):
        self.mode_list_size=len(mode_list)
#        self.no_beam_elem=no_beam_elem
        self.mode_list=mode_list
        self.start_vertheta=start_vertheta
        self.end_vertheta=end_vertheta
        self.xi_c=xi
        self.q=q
        self.varphi=varphi
        self.R=R
        self.r=r
        self.LM_element=LM_element
        self.vertheta=Symbol('vertheta')
        
    def element(self):
        f_ext_element=np.zeros(len(self.LM_element),float)
        count=0
        for j in range(self.mode_list_size):
            k=self.mode_list[j]
            
            self.i=GBT_func_curve(k,self.r,self.vertheta,self.R) 
            
            V_verphi=np.mat([self.varphi * self.R / 2,self.R  * self.varphi ** 2 / 12,self.varphi * self.R / 2,-self.R * self.varphi ** 2 / 12])
            
            self.xi=(1+(self.xi_c*(self.r/self.R)*cos(self.vertheta))) 
            
         #   V_verphi=Matrix([[(self.R + self.r * cos(self.vertheta)) * self.varphi / 2,(self.R + self.r * cos(self.vertheta)) * self.varphi ** 2 / 12,(self.R + self.r * cos(self.vertheta)) * self.varphi / 2,-(self.R + self.r * cos(self.vertheta)) * self.varphi ** 2 / 12]])
            
            # umm not so elegant  
            list_4=self.q*self.i.warping_fun_w()*self.xi
            
            func = lambdify((self.vertheta),list_4, 'numpy') 
            q_0=(quad(lambda vertheta:func(vertheta),self.start_vertheta,self.end_vertheta)[0])*self.r*V_verphi
            

            f_ext_element[count:count+4]=q_0#[q_0[0],q_0[1],q_0[2],q_0[4]]
            count+=4
        return f_ext_element  
#_____________________________________________________________  
        
class loading_cuved_local_w_loading_dist: #dir mode 2 only carful!
    def __init__ (self,R,r,varphi,mode_list,q,LM_element,start_vertheta,end_vertheta):
        self.mode_list_size=len(mode_list)
#        self.no_beam_elem=no_beam_elem
        self.mode_list=mode_list
        self.start_vertheta=start_vertheta
        self.end_vertheta=end_vertheta
        
        self.q=q
        self.varphi=varphi
        self.R=R
        self.r=r
        self.LM_element=LM_element
        self.vertheta=Symbol('vertheta')
        
    def element(self):
        f_ext_element=np.zeros(len(self.LM_element),float)
        count=0
        for j in range(self.mode_list_size):
            k=self.mode_list[j]
            
            self.i=GBT_func_curve(k,self.r,self.vertheta,self.R) 
           # V_verphi=Matrix([[(self.R + self.r * cos(self.vertheta)) * self.varphi / 2,(self.R + self.r * cos(self.vertheta)) * self.varphi ** 2 / 12,(self.R + self.r * cos(self.vertheta)) * self.varphi / 2,-(self.R + self.r * cos(self.vertheta)) * self.varphi ** 2 / 12]])
            
            # umm not so elegant  
            list_4=self.q*sin(self.vertheta)*self.i.warping_fun_w()#*V_verphi
            
            func_0 = lambdify((self.vertheta),list_4, 'numpy')  
            q_1=(quad(lambda vertheta:func_0(vertheta),self.start_vertheta,self.end_vertheta)[0])*self.r
            f_ext_element[count:count+4]=[0,0,q_1,0]
            count+=4
        return f_ext_element        
class loading_cuved_local_v_loading_dist: #dir mode 2 only carful!
    def __init__ (self,R,r,varphi,mode_list,q,LM_element,start_vertheta,end_vertheta):
        self.mode_list_size=len(mode_list)
#        self.no_beam_elem=no_beam_elem
        self.mode_list=mode_list
        self.start_vertheta=start_vertheta
        self.end_vertheta=end_vertheta
        
        self.q=q
        self.varphi=varphi
        self.R=R
        self.r=r
        self.LM_element=LM_element
        self.vertheta=Symbol('vertheta')
        
    def element(self):
        f_ext_element=np.zeros(len(self.LM_element),float)
        count=0
        for j in range(self.mode_list_size):
            k=self.mode_list[j]
            
            self.i=GBT_func_curve(k,self.r,self.vertheta,self.R) 
           # V_verphi=Matrix([[(self.R + self.r * cos(self.vertheta)) * self.varphi / 2,(self.R + self.r * cos(self.vertheta)) * self.varphi ** 2 / 12,(self.R + self.r * cos(self.vertheta)) * self.varphi / 2,-(self.R + self.r * cos(self.vertheta)) * self.varphi ** 2 / 12]])
            
            # umm not so elegant  
            list_4=self.q*cos(self.vertheta)*self.i.warping_fun_v()#*V_verphi
           
            func_0 = lambdify((self.vertheta),list_4, 'numpy')  
            q_1=(quad(lambda vertheta:func_0(vertheta),self.start_vertheta,self.end_vertheta)[0])*self.r
            f_ext_element[count:count+4]=[0,0,q_1,0]
            count+=4
        return f_ext_element            

#_____________________________________________________________  
        
class support:        
    def __init__ (self,mode_list,mat_csi,no_beam_elem,torsion_dof,axisymmetric_dof,extension_dof,bending_dof,allocal_dof,LM_matrix):
        self.mode_list_size=len(mode_list)
        self.no_beam_elem=no_beam_elem
        self.mode_list=mode_list
        self.mat_csi=mat_csi
        self.torsion_dof=torsion_dof
        self.axisymmetric_dof=axisymmetric_dof
        self.extension_dof=extension_dof
        self.bending_dof=bending_dof
        self.allocal_dof=allocal_dof
        self.LM_matrix=LM_matrix
    def dof(self):
        store_delet_row=[]
        store_insert_row=[]
        count=0
        count_ins=0
        for j in range(self.mode_list_size):
            if self.mode_list[j].split("|")[0]=='t':
                if self.torsion_dof[0]=='fix':
                   store_delet_row = np.append(store_delet_row,self.LM_matrix[count:count+2,0] )
                   store_insert_row=np.append(store_insert_row,self.LM_matrix[count,0]-count_ins )
                   store_insert_row=np.append(store_insert_row,self.LM_matrix[count+1,0]-count_ins-1 )               
                if self.torsion_dof[1]=='fix':
                   store_delet_row = np.append(store_delet_row,self.LM_matrix[count+2:count+4,self.no_beam_elem-1] )
                   store_insert_row=np.append(store_insert_row,self.LM_matrix[count+2,self.no_beam_elem-1]-count_ins )
                   store_insert_row=np.append(store_insert_row,self.LM_matrix[count+2+1,self.no_beam_elem-1]-count_ins-1 )
                count_ins+=2
                count+=4
            elif self.mode_list[j].split("|")[0]=='a':
                if self.axisymmetric_dof[0]=='fix':
                   store_delet_row = np.append(store_delet_row,self.LM_matrix[count:count+2,0] )
                   store_insert_row=np.append(store_insert_row,self.LM_matrix[count,0]-count_ins )
                   store_insert_row=np.append(store_insert_row,self.LM_matrix[count+1,0]-count_ins-1 )               
                if self.axisymmetric_dof[1]=='fix':
                   store_delet_row = np.append(store_delet_row,self.LM_matrix[count+2:count+4,self.no_beam_elem-1] )
                   store_insert_row=np.append(store_insert_row,self.LM_matrix[count+2,self.no_beam_elem-1]-count_ins )
                   store_insert_row=np.append(store_insert_row,self.LM_matrix[count+2+1,self.no_beam_elem-1]-count_ins-1 )
                count_ins+=2
                count+=4
            elif self.mode_list[j].split("|")[0]=='1':
                if self.extension_dof[0]=='fix':
                   store_delet_row = np.append(store_delet_row,self.LM_matrix[count,0] )
                   store_insert_row=np.append(store_insert_row,self.LM_matrix[count,0]-count_ins )
                if self.extension_dof[1]=='fix':
                   store_delet_row = np.append(store_delet_row,self.LM_matrix[count+3,self.no_beam_elem-1] )
                   store_insert_row=np.append(store_insert_row,self.LM_matrix[count+3,self.no_beam_elem-1]-count_ins+2)
                count_ins+=-1
                count+=4               
            elif int(self.mode_list[j].split("|")[0])==2 or  int(self.mode_list[j].split("|")[0])==3 :
                if self.bending_dof[0]=='fix':
                   store_delet_row = np.append(store_delet_row,self.LM_matrix[count:count+2,0] )
                   store_insert_row=np.append(store_insert_row,self.LM_matrix[count,0]-count_ins )
                   store_insert_row=np.append(store_insert_row,self.LM_matrix[count+1,0]-count_ins-1 ) 
                if self.bending_dof[1]=='fix':
                   store_delet_row = np.append(store_delet_row,self.LM_matrix[count+2:count+4,self.no_beam_elem-1] )
                   store_insert_row=np.append(store_insert_row,self.LM_matrix[count+2,self.no_beam_elem-1]-count_ins )
                   store_insert_row=np.append(store_insert_row,self.LM_matrix[count+2+1,self.no_beam_elem-1]-count_ins-1 )
                count_ins+=2
                count+=4
            elif int(self.mode_list[j].split("|")[0])>3 :
                if self.allocal_dof[0]=='fix':
                   store_delet_row = np.append(store_delet_row,self.LM_matrix[count:count+2,0] )
                   store_insert_row=np.append(store_insert_row,self.LM_matrix[count,0]-count_ins )
                   store_insert_row=np.append(store_insert_row,self.LM_matrix[count+1,0]-count_ins-1 ) 
                if self.allocal_dof[1]=='fix':
                   store_delet_row = np.append(store_delet_row,self.LM_matrix[count+2:count+4,self.no_beam_elem-1] )
                   store_insert_row=np.append(store_insert_row,self.LM_matrix[count+2,self.no_beam_elem-1]-count_ins )
                   store_insert_row=np.append(store_insert_row,self.LM_matrix[count+2+1,self.no_beam_elem-1]-count_ins-1 )
                count_ins+=2
                count+=4           

        return (store_delet_row.astype('int') , store_insert_row.astype('int'))
#_____________________________________________________________  
        
class amplification:
    def __init__ (self,mode_list,mat_csi,no_beam_elem,u_previous,l):
        self.mode_list_size=len(mode_list)
#        self.no_beam_elem=no_beam_elem
        self.mode_list=mode_list
        self.mat_csi=mat_csi
        self.no_beam_elem=no_beam_elem
        self.u_previous=u_previous
        self.l=l
    def VV2(self):    
        V=np.zeros((self.no_beam_elem+1,self.mode_list_size+1),float)
        V2=np.zeros((self.no_beam_elem+1,self.mode_list_size+1),float)
        count=0
        countl=1
        for j in range(self.mode_list_size):
            if self.mode_list[j].split("|")[0]=='1' : 
                p=0
                for i in range(0,1+self.no_beam_elem*3,3):
                    V2[p,0]=p*self.l  
                    V[p,0]=p*self.l
                    V2[p,countl]=self.u_previous[count+i]
                    p+=1
                count+=self.no_beam_elem*3+1
                countl+=1        
            else:                
                p=0
                for i in range(0,2+self.no_beam_elem*2,2):
                    V2[p,0]=p*self.l  
                    V[p,0]=p*self.l
                    V2[p,countl]=self.u_previous[count+i+1]
                    V[p,countl]=self.u_previous[count+i]
                    
                    p+=1 
                countl+=1    
                count+=2+self.no_beam_elem*2
        return (V,V2) 


#_____________________________________________________________  
        
class Gbt_internal_forces_curved:
    def __init__ (self,mode_list,no_of_beam_elem,cross_sect_ref,u_previous,l,E,t,r,mu,R,xi,LM_matrix,varphi):
        self.mode_list_size=len(mode_list)
        self.mode_list=mode_list
        self.no_of_beam_elem=no_of_beam_elem-1
        self.u_previous=u_previous
        self.L=l
        self.LM_matrix=LM_matrix
        self.LM_matrix_s=LM_connectivity_array.LM_array(LM_connectivity_array(self.mode_list,1))
        self.E=E
        self.t=t
        self.r=r
        self.R=R
        self.varphi=varphi
        self.xi_c=xi
        self.mu=mu
        self.cross_sect_ref=cross_sect_ref
        self.K=self.E*self.t**3/(12.0*(1.0-self.mu**2))
        x=-varphi/2
 
        self.V=np.mat([2 * x ** 3 / varphi ** 3 + 0.1e1 / 0.2e1 - 0.3e1 / 0.2e1 * x / varphi,x ** 3 / varphi ** 2  - x ** 2 / varphi / 2 - x * 1.0 / 4 + varphi * 1.0 / 8,-2 * x ** 3 / varphi ** 3 + 0.1e1 / 0.2e1 + 0.3e1 / 0.2e1 * x / varphi,x ** 3 / varphi ** 2 * 1.0 + x ** 2 / varphi * 1.0 / 2 - x * 1.0 / 4 - varphi * 1.0 / 8])
        self.Vx=np.mat([6 * x ** 2 / varphi ** 3 - 0.3e1 / 0.2e1 / varphi,3 * x ** 2 / varphi ** 2 - x / varphi - 0.1e1 / 0.4e1,-6 * x ** 2 / varphi ** 3 + 0.3e1 / 0.2e1 / varphi,3 * x ** 2 / varphi ** 2 + x / varphi - 0.1e1 / 0.4e1])
        self.Vxx=np.mat([12 * x / varphi ** 3,6 * x / varphi ** 2 * 1.0 - 1 / varphi * 1.0,-12 * x / varphi ** 3,6 * x / varphi ** 2 * 1.0 + 1 / varphi * 1.0])
        self.Vxxx=np.mat([12 / varphi ** 3,6 / varphi ** 2 * 1.0,-12 / varphi ** 3,6 / varphi ** 2 * 1.0])
  #axial
        self.Va=np.mat([-0.9e1 / 0.2e1 * x ** 3 / varphi ** 3 + 0.9e1 / 0.4e1 * x ** 2 / varphi ** 2 + x / varphi / 8 - 0.1e1 / 0.16e2,0.27e2 / 0.2e1 * x ** 3 / varphi ** 3 - 0.9e1 / 0.4e1 * x ** 2 / varphi ** 2 - 0.27e2 / 0.8e1 * x / varphi + 0.9e1 / 0.16e2,-0.27e2 / 0.2e1 * x ** 3 / varphi ** 3 - 0.9e1 / 0.4e1 * x ** 2 / varphi ** 2 + 0.27e2 / 0.8e1 * x / varphi + 0.9e1 / 0.16e2,0.9e1 / 0.2e1 * x ** 3 / varphi ** 3 + 0.9e1 / 0.4e1 * x ** 2 / varphi ** 2 - x / varphi / 8 - 0.1e1 / 0.16e2])
        self.Vax=np.mat([-0.27e2 / 0.2e1 * x ** 2 / varphi ** 3 + 0.9e1 / 0.2e1 * x / varphi ** 2 + 0.1e1 / varphi / 8,0.81e2 / 0.2e1 * x ** 2 / varphi ** 3 - 0.9e1 / 0.2e1 * x / varphi ** 2 - 0.27e2 / 0.8e1 / varphi,-0.81e2 / 0.2e1 * x ** 2 / varphi ** 3 - 0.9e1 / 0.2e1 * x / varphi ** 2 + 0.27e2 / 0.8e1 / varphi,0.27e2 / 0.2e1 * x ** 2 / varphi ** 3 + 0.9e1 / 0.2e1 * x / varphi ** 2 - 0.1e1 / varphi / 8])
        self.Vaxx=np.mat([-27 * x / varphi ** 3 + 0.9e1 / 0.2e1 / varphi ** 2,81 * x / varphi ** 3 - 0.9e1 / 0.2e1 / varphi ** 2,-81 * x / varphi ** 3 - 0.9e1 / 0.2e1 / varphi ** 2,27 * x / varphi ** 3 + 0.9e1 / 0.2e1 / varphi ** 2])
        self.G=self.E/(2.0*(1.0+self.mu))
 

#        u_element=np.zeros((len(LM_matrix_s) ),float)
    def N_x(self):

        n_x=np.zeros((self.cross_sect_ref),float)
        u_element=self.u_previous[self.LM_matrix[:,self.no_of_beam_elem]]
        
        vertheta=(2*math.pi/(self.cross_sect_ref-1))/2 #0#-2*math.pi/(cross_sect_ref-1)
        for theta in range (0,self.cross_sect_ref):
            count_a=0
            for a in range(self.mode_list_size):
                k=self.mode_list[a]
                GBT=GBT_func_numpy_curve(k,self.r,vertheta,self.R)
                self.xi=(1+(self.xi_c*(self.r/self.R)*cos(vertheta)))
                dim_1=self.LM_matrix_s[count_a:count_a+4,0]
                count_a+=4                   

                if k.split("|")[0]=='1' :
                   n_x[theta]+= np.dot(self.Vax,u_element[dim_1])* GBT.warping_fun_u() *self.E*self.t/(self.R*self.xi*(1-self.mu**2))                    
                else: 
                   n_x[theta]+=(self.E*self.t/(1-self.mu**2))*(np.dot(self.Vxx,u_element[dim_1])* GBT.warping_fun_u()/(self.R*self.xi)+\
                                                              (np.dot(self.V,u_element[dim_1])*((GBT.warping_fun_w()*cos(vertheta)-GBT.warping_fun_v()*sin(vertheta))/(self.R*self.xi) + self.mu* (GBT.warping_fun_v_1xdiff()+GBT.warping_fun_w() )/self.r)))
            vertheta+=(2*math.pi/(self.cross_sect_ref-1))
        return (n_x) 
    def N_theta(self):

        n_theta=np.zeros((self.cross_sect_ref),float)
        u_element=self.u_previous[self.LM_matrix[:,self.no_of_beam_elem]]
        
        vertheta=(2*math.pi/(self.cross_sect_ref-1))/2
        for theta in range (0,self.cross_sect_ref):
            count_a=0
            for a in range(self.mode_list_size):
                k=self.mode_list[a]
                GBT=GBT_func_numpy_curve(k,self.r,vertheta,self.R)
                self.xi=(1+(self.xi_c*(self.r/self.R)*cos(vertheta)))
                dim_1=self.LM_matrix_s[count_a:count_a+4,0]
                count_a+=4                   

                if k.split("|")[0]=='1' :
                   n_theta[theta]+=self.mu* np.dot(self.Vax,u_element[dim_1])* GBT.warping_fun_u() *self.E*self.t/(self.R*self.xi*(1-self.mu**2))                    
                else: 
                   n_theta[theta]+=(self.E*self.t/(1-self.mu**2))*(self.mu*np.dot(self.Vxx,u_element[dim_1])* GBT.warping_fun_u()/(self.R*self.xi)+\
                                                              (np.dot(self.V,u_element[dim_1])*( self.mu*(GBT.warping_fun_w()*cos(vertheta)-GBT.warping_fun_v()*sin(vertheta))/(self.R*self.xi) + (GBT.warping_fun_v_1xdiff()+GBT.warping_fun_w() )/self.r)))
            vertheta+=(2*math.pi/(self.cross_sect_ref-1))
        return (n_theta) 

    def N_x_theta(self):

        n_x_theta=np.zeros((self.cross_sect_ref),float)
        u_element=self.u_previous[self.LM_matrix[:,self.no_of_beam_elem]]
        
        vertheta=(2*math.pi/(self.cross_sect_ref-1))/2
        for theta in range (0,self.cross_sect_ref):
            count_a=0
            for a in range(self.mode_list_size):
                k=self.mode_list[a]
                GBT=GBT_func_numpy_curve(k,self.r,vertheta,self.R)
                self.xi=(1+(self.xi_c*(self.r/self.R)*cos(vertheta)))
                dim_1=self.LM_matrix_s[count_a:count_a+4,0]
                count_a+=4                   

                if k.split("|")[0]=='1' :
                   n_x_theta[theta]+=self.G*self.t*np.dot(self.Va,u_element[dim_1])*( GBT.warping_fun_u_1xdiff()/self.r + (GBT.warping_fun_v()+GBT.warping_fun_u()*sin(vertheta))/(self.R*self.xi))              
                else: 
                   n_x_theta[theta]+=self.G*self.t*np.dot(self.Vx,u_element[dim_1])*( GBT.warping_fun_u_1xdiff()/self.r + (GBT.warping_fun_v()+GBT.warping_fun_u()*sin(vertheta))/(self.R*self.xi))              
                   # print (u_element[dim_1],( GBT.warping_fun_u_1xdiff()/self.r + (GBT.warping_fun_v()+GBT.warping_fun_u()*sin(vertheta))/(self.R*self.xi)),self.Vx,self.G,self.t)
            vertheta+=(2*math.pi/(self.cross_sect_ref-1))
        return (n_x_theta)     

    def M_x(self):

        M_x=np.zeros((self.cross_sect_ref),float)
        u_element=self.u_previous[self.LM_matrix[:,self.no_of_beam_elem]]
        
        vertheta=(2*math.pi/(self.cross_sect_ref-1))/2
        for theta in range (0,self.cross_sect_ref):
            count_a=0
            for a in range(self.mode_list_size):
                k=self.mode_list[a]
                GBT=GBT_func_numpy_curve(k,self.r,vertheta,self.R)
                self.xi=(1+(self.xi_c*(self.r/self.R)*cos(vertheta)))
                dim_1=self.LM_matrix_s[count_a:count_a+4,0]
                count_a+=4                   

                if k.split("|")[0]=='1' :
                   M_x[theta]+= self.K*(np.dot(self.Vax,u_element[dim_1])*(GBT.warping_fun_u()*cos(vertheta)-GBT.warping_fun_w() )/(self.R*self.xi)**2)                     
                else: 
                   M_x[theta]+=self.K*((np.dot(self.Vxx,u_element[dim_1])*(GBT.warping_fun_u()*cos(vertheta)-GBT.warping_fun_w() )/(self.R*self.xi)**2)+  np.dot(self.V,u_element[dim_1])*(sin(vertheta)*(GBT.warping_fun_w_1xdiff()-GBT.warping_fun_v())/(self.R*self.r*self.xi)+self.mu*(GBT.warping_fun_v_1xdiff()-GBT.warping_fun_w_2xdiff())/self.r**2 ))
            vertheta+=(2*math.pi/(self.cross_sect_ref-1))
        return (M_x) 
 
    def M_theta(self):

        M_theta=np.zeros((self.cross_sect_ref),float)
        u_element=self.u_previous[self.LM_matrix[:,self.no_of_beam_elem]]
        
        vertheta=(2*math.pi/(self.cross_sect_ref-1))/2
        for theta in range (0,self.cross_sect_ref):
            count_a=0
            for a in range(self.mode_list_size):
                k=self.mode_list[a]
                GBT=GBT_func_numpy_curve(k,self.r,vertheta,self.R)
                self.xi=(1+(self.xi_c*(self.r/self.R)*cos(vertheta)))
                dim_1=self.LM_matrix_s[count_a:count_a+4,0]
                count_a+=4                   

                if k.split("|")[0]=='1' :
                   M_theta[theta]+= self.K*(self.mu*np.dot(self.Vax,u_element[dim_1])*(GBT.warping_fun_u()*cos(vertheta)-GBT.warping_fun_w() )/(self.R*self.xi)**2)                     
                else: 
                   M_theta[theta]+= self.K*(self.mu*(np.dot(self.Vxx,u_element[dim_1])*(GBT.warping_fun_u()*cos(vertheta)-GBT.warping_fun_w() )/(self.R*self.xi)**2)+  np.dot(self.V,u_element[dim_1])*(self.mu*sin(vertheta)*(GBT.warping_fun_w_1xdiff()-GBT.warping_fun_v())/(self.R*self.r*self.xi)+(GBT.warping_fun_v_1xdiff()-GBT.warping_fun_w_2xdiff())/self.r**2 ))
            vertheta+=(2*math.pi/(self.cross_sect_ref-1))
        return (M_theta) 
    
    
    def M_x_theta(self):

        M_x_theta=np.zeros((self.cross_sect_ref),float)
        u_element=self.u_previous[self.LM_matrix[:,self.no_of_beam_elem]]
        
        vertheta=(2*math.pi/(self.cross_sect_ref-1))/2
        for theta in range (0,self.cross_sect_ref):
            count_a=0
            for a in range(self.mode_list_size):
                k=self.mode_list[a]
                GBT=GBT_func_numpy_curve(k,self.r,vertheta,self.R)
                self.xi=(1+(self.xi_c*(self.r/self.R)*cos(vertheta)))
                dim_1=self.LM_matrix_s[count_a:count_a+4,0]
                count_a+=4                   

                if k.split("|")[0] =='1' :
                   M_x_theta[theta]+=(self.G*self.t**3/6)*np.dot(self.Va,u_element[dim_1])*( -GBT.warping_fun_w_1xdiff()/(self.R*self.r*self.xi)-GBT.warping_fun_w()*sin(vertheta)/(self.R*self.xi)**2+\
                                                         GBT.warping_fun_v()*(1+2*self.xi)/(4*self.R*self.r*self.xi**2)+GBT.warping_fun_u_1xdiff()*cos(vertheta)/(2*self.R*self.r*self.xi)-\
                                                             GBT.warping_fun_u_1xdiff()/(4*self.r*self.r*self.xi) + GBT.warping_fun_u()*(2*self.r**2*sin(2*vertheta)+(1-2*self.xi)*self.R*self.r*sin(vertheta))/(2*self.R*self.r*self.xi)**2)                    
                else: 
                   M_x_theta[theta]+=(self.G*self.t**3/6)*np.dot(self.Vx,u_element[dim_1])*( -GBT.warping_fun_w_1xdiff()/(self.R*self.r*self.xi)-GBT.warping_fun_w()*sin(vertheta)/(self.R*self.xi)**2+\
                                                         GBT.warping_fun_v()*(1+2*self.xi)/(4*self.R*self.r*self.xi**2)+GBT.warping_fun_u_1xdiff()*cos(vertheta)/(2*self.R*self.r*self.xi)-\
                                                             GBT.warping_fun_u_1xdiff()/(4*self.r*self.r*self.xi) + GBT.warping_fun_u()*(2*self.r**2*sin(2*vertheta)+(1-2*self.xi)*self.R*self.r*sin(vertheta))/(2*self.R*self.r*self.xi)**2)                    

            vertheta+=(2*math.pi/(self.cross_sect_ref-1))
        return (M_x_theta) 
 
    def Q_x(self):

        Q_x=np.zeros((self.cross_sect_ref),float)
        u_element=self.u_previous[self.LM_matrix[:,self.no_of_beam_elem]]
    #    vertheta=Symbol('vertheta')
        ver=Symbol('ver')
        vertheta=(2*math.pi/(self.cross_sect_ref-1))/2
        for theta in range (0,self.cross_sect_ref):
            count_a=0
            for a in range(self.mode_list_size):
                k=self.mode_list[a]
                
                GBT=GBT_func_numpy_curve(k,self.r,vertheta,self.R)            
                self.xi=(1+(self.xi_c*(self.r/self.R)*cos(vertheta))) 
                
                # GBT_sym=GBT_func_curve(k,self.r,vertheta,self.R)
                # M_x_theta_theta=diff((-GBT_sym.warping_fun_w_1xdiff()/(self.R*self.r*self.xi)-GBT_sym.warping_fun_w()*sin(vertheta)/(self.R*self.xi)**2+\
                #                                          GBT_sym.warping_fun_v()*(1+2*self.xi)/(4*self.R*self.r*self.xi**2)+GBT_sym.warping_fun_u_1xdiff()*cos(vertheta)/(2*self.R*self.r*self.xi)-\
                #                                              GBT_sym.warping_fun_u_1xdiff()/(4*self.r*self.r*self.xi) + GBT_sym.warping_fun_u()*(2*self.r**2*sin(2*vertheta)+(1-2*self.xi)*self.R*self.r*sin(vertheta))/(2*self.R*self.r*self.xi)**2),vertheta)                               
                GBT_sym=GBT_func_curve(k,self.r,ver,self.R)
                M_x_theta_theta=diff((-GBT_sym.warping_fun_w_1xdiff()/(self.R*self.r*self.xi)-GBT_sym.warping_fun_w()*sin(ver)/(self.R*self.xi)**2+\
                                                         GBT_sym.warping_fun_v()*(1+2*self.xi)/(4*self.R*self.r*self.xi**2)+GBT_sym.warping_fun_u_1xdiff()*cos(ver)/(2*self.R*self.r*self.xi)-\
                                                             GBT_sym.warping_fun_u_1xdiff()/(4*self.r*self.r*self.xi) + GBT_sym.warping_fun_u()*(2*self.r**2*sin(2*ver)+(1-2*self.xi)*self.R*self.r*sin(ver))/(2*self.R*self.r*self.xi)**2),ver)                               
                M_x_theta_theta_func=lambdify(ver, M_x_theta_theta, 'numpy')
                dim_1=self.LM_matrix_s[count_a:count_a+4,0]
                count_a+=4                   

                if k.split("|")[0] =='1' :
                   Q_x[theta]+=(self.G*self.t**3/(6*self.r))*np.dot(self.Va,u_element[dim_1])*M_x_theta_theta_func(vertheta)+\
                              (1/(self.R*self.xi))*(self.K*(np.dot(self.Vaxx,u_element[dim_1])*(GBT.warping_fun_u()*cos(vertheta)-GBT.warping_fun_w() )/(self.R*self.xi)**2))-\
                               (2*sin(vertheta)/(self.R*self.xi))*((self.G*self.t**3/6)*np.dot(self.Va,u_element[dim_1])*( -GBT.warping_fun_w_1xdiff()/(self.R*self.r*self.xi)-GBT.warping_fun_w()*sin(vertheta)/(self.R*self.xi)**2+\
                                                         GBT.warping_fun_v()*(1+2*self.xi)/(4*self.R*self.r*self.xi**2)+GBT.warping_fun_u_1xdiff()*cos(vertheta)/(2*self.R*self.r*self.xi)-\
                                                             GBT.warping_fun_u_1xdiff()/(4*self.r*self.r*self.xi) + GBT.warping_fun_u()*(2*self.r**2*sin(2*vertheta)+(1-2*self.xi)*self.R*self.r*sin(vertheta))/(2*self.R*self.r*self.xi)**2))   
                
                
                
                
                else: 
                   Q_x[theta]+=(self.G*self.t**3/(6*self.r))*np.dot(self.Vx,u_element[dim_1])*M_x_theta_theta_func(vertheta)+\
                              (1/(self.R*self.xi))*self.K*((np.dot(self.Vxxx,u_element[dim_1])*(GBT.warping_fun_u()*cos(vertheta)-GBT.warping_fun_w() )/(self.R*self.xi)**2)+  np.dot(self.Vx,u_element[dim_1])*(sin(vertheta)*(GBT.warping_fun_w_1xdiff()-GBT.warping_fun_v())/(self.R*self.r*self.xi)+self.mu*(GBT.warping_fun_v_1xdiff()-GBT.warping_fun_w_2xdiff())/self.r**2 ))-\
                               (2*sin(vertheta)/(self.R*self.xi))*((self.G*self.t**3/6)*np.dot(self.Vx,u_element[dim_1])*( -GBT.warping_fun_w_1xdiff()/(self.R*self.r*self.xi)-GBT.warping_fun_w()*sin(vertheta)/(self.R*self.xi)**2+\
                                                         GBT.warping_fun_v()*(1+2*self.xi)/(4*self.R*self.r*self.xi**2)+GBT.warping_fun_u_1xdiff()*cos(vertheta)/(2*self.R*self.r*self.xi)-\
                                                             GBT.warping_fun_u_1xdiff()/(4*self.r*self.r*self.xi) + GBT.warping_fun_u()*(2*self.r**2*sin(2*vertheta)+(1-2*self.xi)*self.R*self.r*sin(vertheta))/(2*self.R*self.r*self.xi)**2))   

            vertheta+=(2*math.pi/(self.cross_sect_ref-1))
        return (Q_x)    

    def Q_theta(self):

        Q_theta=np.zeros((self.cross_sect_ref),float)
        u_element=self.u_previous[self.LM_matrix[:,self.no_of_beam_elem]]
    #    vertheta=Symbol('vertheta')
        ver=Symbol('ver')
        vertheta=(2*math.pi/(self.cross_sect_ref-1))/2
        for theta in range (0,self.cross_sect_ref):
            count_a=0
            for a in range(self.mode_list_size):
                k=self.mode_list[a]
                
                GBT=GBT_func_numpy_curve(k,self.r,vertheta,self.R)            
                self.xi=(1+(self.xi_c*(self.r/self.R)*cos(vertheta))) 
                
                GBT_sym=GBT_func_curve(k,self.r,ver,self.R)
                M_theta_theta_1=diff( self.mu*((GBT_sym.warping_fun_u()*cos(ver)-GBT_sym.warping_fun_w() )/(self.R*self.xi)**2),ver) 
                M_theta_theta_2= diff((self.mu*sin(ver)*(GBT_sym.warping_fun_w_1xdiff()-GBT_sym.warping_fun_v())/(self.R*self.r*self.xi)+(GBT_sym.warping_fun_v_1xdiff()-GBT_sym.warping_fun_w_2xdiff())/self.r**2 ),ver)                               
                M_theta_theta_func_1=lambdify(ver, M_theta_theta_1, 'numpy')
                M_theta_theta_func_2=lambdify(ver, M_theta_theta_2, 'numpy')                
                dim_1=self.LM_matrix_s[count_a:count_a+4,0]
                count_a+=4                   

                if k.split("|")[0] =='1' :
                   Q_theta[theta]+=(self.K/self.r)*(np.dot(self.Vax,u_element[dim_1])*M_theta_theta_func_1(vertheta)+np.dot(self.V,u_element[dim_1])*M_theta_theta_func_2(vertheta) )+\
                               (1/(self.R*self.xi))*((self.G*self.t**3/6)*np.dot(self.Vax,u_element[dim_1])*( -GBT.warping_fun_w_1xdiff()/(self.R*self.r*self.xi)-GBT.warping_fun_w()*sin(vertheta)/(self.R*self.xi)**2+\
                                                         GBT.warping_fun_v()*(1+2*self.xi)/(4*self.R*self.r*self.xi**2)+GBT.warping_fun_u_1xdiff()*cos(vertheta)/(2*self.R*self.r*self.xi)-\
                                                             GBT.warping_fun_u_1xdiff()/(4*self.r*self.r*self.xi) + GBT.warping_fun_u()*(2*self.r**2*sin(2*vertheta)+(1-2*self.xi)*self.R*self.r*sin(vertheta))/(2*self.R*self.r*self.xi)**2))+\
                               (sin(vertheta)/(self.R*self.xi))* ((self.K*((np.dot(self.Vax,u_element[dim_1])*(GBT.warping_fun_u()*cos(vertheta)-GBT.warping_fun_w() )/(self.R*self.xi)**2)+  np.dot(self.V,u_element[dim_1])*(sin(vertheta)*(GBT.warping_fun_w_1xdiff()-GBT.warping_fun_v())/(self.R*self.r*self.xi)+self.mu*(GBT.warping_fun_v_1xdiff()-GBT.warping_fun_w_2xdiff())/self.r**2 )))-\
                                                                  (self.K*(self.mu*(np.dot(self.Vax,u_element[dim_1])*(GBT.warping_fun_u()*cos(vertheta)-GBT.warping_fun_w() )/(self.R*self.xi)**2)+  np.dot(self.V,u_element[dim_1])*(self.mu*sin(vertheta)*(GBT.warping_fun_w_1xdiff()-GBT.warping_fun_v())/(self.R*self.r*self.xi)+(GBT.warping_fun_v_1xdiff()-GBT.warping_fun_w_2xdiff())/self.r**2 ))) )  
                
                
                
                
                else: 
                   Q_theta[theta]+=(self.K/self.r)*(np.dot(self.Vxx,u_element[dim_1])*M_theta_theta_func_1(vertheta)+np.dot(self.V,u_element[dim_1])*M_theta_theta_func_2(vertheta) )+\
                               (1/(self.R*self.xi))*((self.G*self.t**3/6)*np.dot(self.Vxx,u_element[dim_1])*( -GBT.warping_fun_w_1xdiff()/(self.R*self.r*self.xi)-GBT.warping_fun_w()*sin(vertheta)/(self.R*self.xi)**2+\
                                                         GBT.warping_fun_v()*(1+2*self.xi)/(4*self.R*self.r*self.xi**2)+GBT.warping_fun_u_1xdiff()*cos(vertheta)/(2*self.R*self.r*self.xi)-\
                                                             GBT.warping_fun_u_1xdiff()/(4*self.r*self.r*self.xi) + GBT.warping_fun_u()*(2*self.r**2*sin(2*vertheta)+(1-2*self.xi)*self.R*self.r*sin(vertheta))/(2*self.R*self.r*self.xi)**2))+\
                               (sin(vertheta)/(self.R*self.xi))* ((self.K*((np.dot(self.Vxx,u_element[dim_1])*(GBT.warping_fun_u()*cos(vertheta)-GBT.warping_fun_w() )/(self.R*self.xi)**2)+  np.dot(self.V,u_element[dim_1])*(sin(vertheta)*(GBT.warping_fun_w_1xdiff()-GBT.warping_fun_v())/(self.R*self.r*self.xi)+self.mu*(GBT.warping_fun_v_1xdiff()-GBT.warping_fun_w_2xdiff())/self.r**2 )))-\
                                                                  (self.K*(self.mu*(np.dot(self.Vxx,u_element[dim_1])*(GBT.warping_fun_u()*cos(vertheta)-GBT.warping_fun_w() )/(self.R*self.xi)**2)+  np.dot(self.V,u_element[dim_1])*(self.mu*sin(vertheta)*(GBT.warping_fun_w_1xdiff()-GBT.warping_fun_v())/(self.R*self.r*self.xi)+(GBT.warping_fun_v_1xdiff()-GBT.warping_fun_w_2xdiff())/self.r**2 ))) )  
                

            vertheta+=(2*math.pi/(self.cross_sect_ref-1))
        return (Q_theta)       
#_____________________________________________________________  
class Results_comparison_num2:
    def __init__ (self, list_values,minValue):
        #start = timer()
        #self.total_time = 0
        self.list_values=list_values
        self.shape=np.shape(self.list_values)
        self.nValue=self.shape[1]
        self.mSD = np.empty(self.nValue)
        self.maxAbsValues=abs(max(np.amax(self.list_values),np.amin(self.list_values),key=abs))
        self.tolerance=minValue*self.maxAbsValues
        # Creating a Mask of all vaules inside the interval
        self.list_values=np.ma.masked_inside(self.list_values,-self.tolerance,self.tolerance)
        print (self.list_values)
        # Applying a Or Mask of the two vectors
        self.list_values.mask=np.ma.mask_or(self.list_values.mask[0,:],self.list_values.mask[1,:])
        
        self.diffs=self.list_values[0,:].compressed()-self.list_values[1,:].compressed()
        #end = timer()
        #self.total_time += end - start
        #print(self.total_time) 
    def Mean_signed_deviation(self):
        #start = timer()
        self.mSD = abs(np.divide(self.diffs,self.list_values[1,:].compressed())*100)
        print('Mean Signed Deviation - MSD - and its standard deviation (in %)')
        #end = timer()
        #self.total_time += end - start
        #print(self.total_time)
        return (np.mean(self.mSD),np.std(self.mSD))    
#_____________________________________________________________      
class Results_comparison_num:
    def __init__ (self, list_values,minValue):
        #start = timer()
        #self.total_time = 0
        self.maxAbsValues=np.amax(abs(list_values))
        self.tolerance=minValue*self.maxAbsValues
        self.list_values=list_values#[:,(list_values[1] <= -self.tolerance) & (list_values[1] < self.tolerance)]
      #  print(self.list_values)
        self.diffs=(self.list_values[0,:]-self.list_values[1,:])
        #self.total_time += end - start
        #print(self.total_time)
        
    def Mean_signed_deviation(self):
        #start = timer()
        self.mSD =abs( np.divide(self.diffs,self.list_values[1,:])*100)
        print('Mean Signed Deviation - MSD - and its standard deviation (in %)')
        #end = timer()
        #self.total_time += end - start
        #print(self.total_time)
        return (np.mean(self.mSD),np.std(self.mSD))
#_____________________________________________________________      
class Multi_results_comparison_num:
    def __init__ (self, list_values,minValue):
        #start = timer()
        #self.total_time = 0
        self.list_values=list_values
        self.shape=np.shape(self.list_values)
        self.nlist=self.shape[0]
        self.nValue=self.shape[1]
        self.maxAbsValues=abs(max(np.amax(self.list_values),np.amin(self.list_values),key=abs))
        self.tolerance=minValue*self.maxAbsValues
        # Eliminating all vectors that values are under the tolerance
        
        # All difference between the two vectors are saved in a 3D matrix
        # The number of rows and colums are the same of the number of input lists
        # The level present the MSD from each value among the lists. The first level is reserved to the final results
        self.diffs=np.zeros((self.nlist,self.nlist,self.nValue+1))
        for i in range(self.nlist-1):
            for j in range(i+1,self.nlist):
                for k in range(self.nValue):
                    if -self.tolerance < self.list_values[i,k] < self.tolerance:
                       self.diffs[i,j,k+1]=np.nan
                    else:
                       self.diffs[i,j,k+1]=(self.list_values[j,k]-self.list_values[i,k])/self.list_values[i,k]
                self.diffs[i,j,0]=np.nanmean(self.diffs[i,j,1:self.nValue+1])
                self.diffs[j,i,0]=np.nanstd(self.diffs[i,j,1:self.nValue+1])
        #end = timer()
        #self.total_time += end - start
        #print(self.total_time)        
                
    def Mean_signed_deviation(self):
        print('Mean Signed Deviation - MSD - and its standard deviation (in %)')
        return (self.diffs[:,:,0])       
#_____________________________________________________________  
                
class plot_internal_forces:
    def __init__ (self,cross_section,c_s_1,c_s_2,force,comp_1,comp_2,r, scale,title):
        self.cross_section=cross_section
        self.c_s_1=c_s_1

        
        self.c_s_2=c_s_2
        self.r=r
        
        
        self.scale=scale/max(np.max(force),np.max(comp_1),np.max(comp_2))
        self.force=force*self.scale
        self.comp_1=comp_1*self.scale#np.fliplr([comp_1])[0]*100
        self.comp_2=comp_2*self.scale       
        self.title=title
    """ load participation"""
    def plt(self):
        
        fig = plt.figure(frameon=False)
        pos_signal = self.force.copy()
        neg_signal = self.force.copy()
        pos_signal[pos_signal <= 0] = np.NaN
        neg_signal[neg_signal >= 0] = np.NaN         
        plotxy_p=np.zeros((len(self.force),2),float)
        plotxy_p[:,0]=self.cross_section
        plotxy_p[:,1]=pos_signal
        plotxy_p = plotxy_p[pd.notnull(plotxy_p[:,1])]
        
        plotxy_n=np.zeros((len(self.force),2),float)
        plotxy_n[:,0]=self.cross_section
        plotxy_n[:,1]=neg_signal
        plotxy_n = plotxy_n[pd.notnull(plotxy_n[:,1])]
       
        print(self.title,self.scale)   
    
      #  plt.polar(self.cross_section, self.force, c= 'k',linewidth=2)
        if self.c_s_1!=0:
           self.r_d = np.linspace(0,2*math.pi,self.c_s_1+1)
           self.c_shell = self.r_d[0:len(self.r_d)-1]            
           plt.polar(self.c_shell, self.r+self.comp_1,'x', c= 'k',markevery=3,linewidth=6,markersize=10)
#           plt.polar(np.linspace(-1.0/2.0*math.pi,-5.0/2.0*math.pi,self.c_s_1), self.comp_1,'x', c= 'k',linewidth=3,markersize=6)

        if self.c_s_2!=0:
           plt.polar(np.linspace(1.0/2.0*math.pi,5.0/2.0*math.pi,self.c_s_2), self.comp_2,'--', c= 'k',linewidth=2)            

        try:
            plt.polar(self.cross_section,self.r+pos_signal, '-',c= 'r',linewidth=4)
            plt.polar(self.cross_section,self.r+neg_signal, '-',c= 'b',linewidth=4)
        except (RuntimeError, TypeError, NameError, RuntimeWarning):
            pass
   #     plt.title(self.title)
    #    text= "x={:.3f}, y={:.3f}".format(np.amax(N_teta[0,:], np.amin(N_teta[0,:])
    #    plt.annotate(text, xy=(0, 0), xytext=(0.94,0.96), **kw)
#        plt.text(0, np.amin(self.force*self.scale)-(self.r*1.5),' Max= %5.2f \n Min= %5.2f' % (float(np.amax(self.force)),float(np.amin(self.force))),{'color': 'k', 'fontsize': 10,'weight':'bold'})

        plt.text(1.1*math.pi,-(self.r*0.25),r'Max. =  %5.2f' % (float(np.amax(self.force)/self.scale))+'\n ',{'color': 'r', 'fontsize': 23,'weight':'bold','fontname':'Times New Roman'})
        plt.text(1.1*math.pi,-(self.r*0.25),r'Min. =  %5.2f' % (float(np.amin(self.force)/self.scale)),{'color': 'b', 'fontsize': 23,'weight':'bold','fontname':'Times New Roman'})

     #   plt.text(0, np.amin(N_teta[0,:]*a)-(r*1.5), r'Max='+np.amax(N_teta[0,:],{'color': 'b', 'fontsize': 20})
       # print np.size(u), np.size(N_teta[0,:]),np.size(np.ones((cross_sect_ref,1)))
        plt.polar(self.cross_section, self.r+(self.cross_section*0), 'k--')
    #    y1=np.array(np.zeros((cross_sect_ref,1)))
    #    y2=np.array(N_teta[0,:])
     #   y1,y2, where=y2 >= y1
       # plt.fill_between(u, y1,y2, where=y2 >= y1,facecolor='red', interpolate=True)   
       # plt.fill_between(plotxy_p[:,0],0,plotxy_p[:,1],  hatch='\\')  

     #   plt.fill_between( 0 , plotxy_p[:,1],color="b", hatch='//', edgecolor="r", linewidth=0.0)  
        plt.fill_between(self.cross_section, self.r,self.r+pos_signal, hatch='//', edgecolor="r",facecolor="none", linewidth=0.0)  
        
        plt.fill_between(self.cross_section, self.r,self.r+neg_signal, hatch='//', edgecolor="b",facecolor="none", linewidth=0.0)  
    
     #   plt.yticks([np.amin(N_teta[0,:]*a)-(r*1), 0 ,np.amax(N_teta[0,:]*a)+(r*1)])
        plt.ylim(-self.r*2.15,self.r*2.15)
    #    plt.xlim(-self.r*1.5,self.r*1.5)
      #  print np.amax(N_teta[0,:]), np.amin(N_teta[0,:])
        ntext=self.title.split(' ')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('%s.pdf'% str(ntext[0]), format='pdf', dpi=400)        
        plt.show()

        compare=np.zeros((2,self.c_s_1),float)
        compare[0,:]=self.force[0:self.c_s_1]/self.scale
        compare[1,:]=self.comp_1/self.scale
        compare_S_G=Results_comparison_num(compare,0.10)
        print ( '%s'% str(ntext[0]), compare_S_G.Mean_signed_deviation())          

        return 'ok'

