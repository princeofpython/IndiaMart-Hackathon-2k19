# -*- coding: utf-8 -*-
"""Copy of phase1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/nik299/IndiaMart-Hackathon-2k19/blob/master/phase1.ipynb

# Phase-1 IndiaMART Hackathon
This notebook is our a working prototype solution to gauge the appropriate unit wise price range for the 3 categories(Gloves,Kurtas,Drills) based on their units by removing outliers from the data.

**Instructions for running this notebook**


*   Jupyter notebook is needed to run this notebook if it is not available Please use Colab from google  to run it.
*   required libraries to run this notebook are pandas,numpy,seaborn and scipy.
*   Incase you are running this notebook please make sure all **.csv** files are uploaded
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from scipy.signal import argrelextrema
from scipy import stats

"""we can import the required data either from the folder or our git repository.
The given .xlsx file is split into Three parts and and converted to .csv file which is easy to handle.
"""

try:
  drilldf = pd.read_csv('./phaseone-drill.csv')
  glovedf = pd.read_csv('./phaseone-gloves.csv')
  kurtadf = pd.read_csv('./phaseone-kurta.csv')
except:
  try:
    drilldf = pd.read_csv('https://raw.githubusercontent.com/nik299/IndiaMart-Hackathon-2k19/master/phaseone-drill.csv')
    glovedf = pd.read_csv('https://raw.githubusercontent.com/nik299/IndiaMart-Hackathon-2k19/master/phaseone-gloves.csv')
    kurtadf = pd.read_csv('https://raw.githubusercontent.com/nik299/IndiaMart-Hackathon-2k19/master/phaseone-kurta.csv')
  except:
    print("no file found and no internet connection")

"""the following piece of code prints out all unique units of the three item given"""

drill_units=drilldf['Unit'].unique()
print(drill_units,'drills')
glove_units=glovedf['Unit'].unique()
print(glove_units,'gloves')
kurta_units=kurtadf['Unit'].unique()
print(kurta_units,'kurtas')

"""The following three cells prints out the count of each unique unts."""

drilldf['Unit'].value_counts()

glovedf['Unit'].value_counts()

kurtadf['Unit'].value_counts()

"""# Calculating Z-score
Z-score is meausure of how much given sample is deviating compared to Standaed deviation.
In python the function is available in scipy.stats which we are using in the following cells

**note:**we are calculating only abolsolute values as sign of z-score doesn't matter in finding outliers
"""

drill_z = np.abs(stats.zscore(drilldf['Price']))
glove_z = np.abs(stats.zscore(glovedf.loc[glovedf['Unit']=='Pair']['Price']))
kurta_z = np.abs(stats.zscore(kurtadf.loc[((kurtadf['Unit']=='Piece') | (kurtadf['Unit']=='Piece(s)'))]['Price']))

"""After obtaing z-score we are not considering all entries whse z-score is greater than 3 or less than -3 and also we are considering only some units values as unit conversion is not as indicative as the sugar example given."""

drill_ol=drilldf[(drill_z < 3)]
npa=drilldf[(drill_z < 3)]['Price']
npa1=glovedf.loc[glovedf['Unit']=='Pair'][(glove_z < 3)]['Price']
npa2=kurtadf.loc[((kurtadf['Unit']=='Piece') | (kurtadf['Unit']=='Piece(s)'))][(kurta_z < 3)]['Price'][:-1]

"""The following 3 cells describe the data taken for consideration"""

drilldf[(drill_z < 3)].describe()

glovedf.loc[glovedf['Unit']=='Pair'][(glove_z < 3)].describe()

kurtadf.loc[((kurtadf['Unit']=='Piece') | (kurtadf['Unit']=='Piece(s)'))][(kurta_z < 3)].describe()

"""The following cell is used to store standard deviation which is very important to calculate the bandwidth in later part"""

stdev=np.std(npa)
stdev1=np.std(npa1)
stdev2=np.std(npa2)
print(stdev,stdev1,stdev2)

"""# Kernel Density estimation method for calculation of Probabilty Disribution Function
we are using probability distribution function for find ing require range as pdf is more suitable for finding relationship in general population.

kernel Density Estimation is useful method for calculating pdf from discret samples.
"""

from scipy.stats import gaussian_kde
def kde_scipy(x, x_grid, bandwidth, **kwargs):
    """Kernel Density Estimation with Scipy"""
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)

y_grid=np.linspace(np.amin(npa)-(np.amax(npa)-np.amin(npa))*0.4,np.amax(npa)+(np.amax(npa)-np.amin(npa))*0.4,10000)
ideal=1.06*(np.std(npa))*((len(npa)**(-1/5)))
pdf = kde_scipy(npa, y_grid, bandwidth=ideal)
plt.plot(y_grid, pdf, color='blue', alpha=1, lw=1)
plt.title('PDF for drill')
plt.xlabel('Price')
plt.show()

y_grid1=np.linspace(np.amin(npa1)-(np.amax(npa1)-np.amin(npa1))*0.4,np.amax(npa1)+(np.amax(npa1)-np.amin(npa1))*0.4,10000)
ideal1=1.06*(np.std(npa1))*((len(npa1)**(-1/5)))
pdf1 = kde_scipy(npa1, y_grid1, bandwidth=ideal1)
plt.plot(y_grid1, pdf1, color='blue', alpha=1, lw=1)
plt.title('PDF for gloves')
plt.xlabel('Price')
plt.show()

y_grid2=np.linspace(-(np.amax(npa2)-np.amin(npa2))*0.05,np.amax(npa2)*1.05,10000)
ideal2=1.06*(np.std(npa2))*((len(npa2)**(-1/5)))
pdf2 = kde_scipy(npa2, y_grid2, bandwidth=ideal2)
plt.plot(y_grid2, pdf2, color='blue', alpha=1, lw=1)
plt.title('PDF for kurta')
plt.xlabel('Price')
plt.show()

w=((np.amax(npa)-np.amin(npa))*1.8)/10000
w1=((np.amax(npa1)-np.amin(npa1))*1.8)/10000
w2=((np.amax(npa2)-np.amin(npa2))*0.05+np.amax(npa2)*1.05)/10000
print(np.sum(pdf)*w,np.sum(pdf1)*w1,np.sum(pdf2)*w2)

pdf_area=np.zeros(len(pdf)-1)
for a in range(len(pdf_area)):
  pdf_area[a]=(pdf[a]+pdf[a+1])*(w/2)
print(np.sum(pdf_area))

"""# Range calculation from PDF 
we used 2 methods for calculating a range they are 

1.   finding the peak and finding a range which covers 50% of the area under pdf and having peak as mean point of that range(it means there is 50% chance for a product to be in that range) this ensures price which has maximum entries lies in that region

2.   finding the smallest range which covers majority of the region(we can use range which covers 50% of the region)the advantage with this method is it ensures ranges is minimum.
"""

f_peak=argrelextrema(pdf, np.greater)[0][0]
for q in range(f_peak):
  if np.sum(pdf[f_peak-q:f_peak+q])*w > 0.5:
    print(np.sum(pdf[f_peak-q:f_peak+q])*w,q)
    break
print('This is the range obtained for drill ',np.round((f_peak-q)*w+np.amin(npa)-(np.amax(npa)-np.amin(npa))*0.4),np.round((f_peak+q)*w+np.amin(npa)-(np.amax(npa)-np.amin(npa))*0.4))

f_peak1=argrelextrema(pdf1, np.greater)[0][0]
q=0
for q in range(f_peak1):
  if np.sum(pdf1[f_peak1-q:f_peak1+2*q])*w1 > 0.5:
    print(np.sum(pdf1[f_peak1-q:f_peak1+2*q])*w1,q)
    break
print('This is the range obtained for gloves ',np.round((f_peak1-q)*w1+np.amin(npa1)-(np.amax(npa1)-np.amin(npa1))*0.4),np.round((f_peak1+2*q)*w1+np.amin(npa1)-(np.amax(npa1)-np.amin(npa1))*0.4))

f_peak2=argrelextrema(pdf2, np.greater)[0][0]
for q in range(f_peak2):
  if np.sum(pdf2[f_peak2-q:f_peak2+q])*w2 > 0.5:
    print(np.sum(pdf2[f_peak2-q:f_peak2+q])*w2,q)
    break
print('This is the range obtained for Kurta ',np.round((f_peak2-q)*w2-(np.amax(npa2)-np.amin(npa2))*0.05),np.round((f_peak2+q)*w2-(np.amax(npa2)-np.amin(npa2))*0.05))

qq1=len(pdf)
for q in range(len(pdf)):
  for p in range(len(pdf)-(q+1)):
     if np.sum(pdf[p:p+q+1])*w > 0.5:
        print(np.sum(pdf[p:p+q+1])*w,p,q)
        break
  if np.sum(pdf[p:p+q+1])*w > 0.5:
    break
print('This is the range obtained for drill ',np.round((p)*w+np.amin(npa)-(np.amax(npa)-np.amin(npa))*0.4),np.round((p+q+1)*w+np.amin(npa)-(np.amax(npa)-np.amin(npa))*0.4))

qq2=len(pdf1)
for q in range(len(pdf1)):
  for p in range(len(pdf1)-(q+1)):
     if np.sum(pdf1[p:p+q+1])*w1 > 0.5:
        print(np.sum(pdf1[p:p+q+1])*w1,p,q)
        break
  if np.sum(pdf1[p:p+q+1])*w1 > 0.5:
    break
print('This is the range obtained for gloves ',np.round((p)*w1+np.amin(npa1)-(np.amax(npa1)-np.amin(npa1))*0.4),np.round((p+q+1)*w1+np.amin(npa1)-(np.amax(npa1)-np.amin(npa1))*0.4))

qq3=len(pdf2)
for q in range(len(pdf2)):
  for p in range(len(pdf2)-(q+1)):
     if np.sum(pdf2[p:p+q+1])*w2 > 0.6:
        print(np.sum(pdf2[p:p+q+1])*w2,p,q)
        break
  if np.sum(pdf2[p:p+q+1])*w2 > 0.6:
    break
print('This is the range obtained for Kurta ',np.round((p)*w2-(np.amax(npa2)-np.amin(npa2))*0.05),np.round((p+q+1)*w2-(np.amax(npa2)-np.amin(npa2))*0.05))