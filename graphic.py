#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 13:48:00 2018

@author: iiitnr
"""

#necessary imports
################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
################################################################################

#path to the data
path = "adult.data"
################################################################################
col = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","nativecountry","result"]
df = pd.read_csv(path,header=None)
# print df.head()
# print df.shape
# print df.isnull().any()
# df = df.as_matrix()
data = pd.DataFrame()
i=0
for o in col:
    data[o] = df[i]
    i = i+1

################################################################################
#inspecting the data

inspect =  data.head()
print (data.head())


dataUS=data[data.nativecountry==" United-States"]
dataIndia=data[data.nativecountry==" India"]

dataIndia_50kp=dataIndia[dataIndia.result==" >50K"]
dataIndia_50kl=dataIndia[dataIndia.result== " <=50K"]

dataUS_50kp=dataUS[dataUS.result==" >50K"]
dataUS_50kl=dataUS[dataUS.result== " <=50K"]
################################################################################

import matplotlib.pyplot as plt
total = float(len(dataUS) + len(dataIndia))
 
# Data to plot
labels = 'India','USA'
sizes =  [len(dataIndia)/total,len(dataUS)/total]
explode = (0.1, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.title("UnitedStates vs India") 
plt.axis('equal')
plt.show()

################################################################################
 
total = float(len(dataUS))
# Data to plot
labels = '50k-','50k+'
sizes =  [len(dataUS_50kl)/total,len(dataUS_50kp)/total]
explode = (0.1, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels,
        autopct='%1.1f%%', shadow=True, startangle=140)
 plt.title("UnitedStates")
plt.axis('equal')
plt.show()
################################################################################
 
total = float(len(dataIndia))
# Data to plot
labels = '50k-','50k+'
sizes =  [len(dataIndia_50kl)/total,len(dataIndia_50kp)/total]
explode = (0.1, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.title("India") 
plt.axis('equal')
plt.show()


