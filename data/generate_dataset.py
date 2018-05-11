# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 11:56:00 2018

@author: nprad
"""

import numpy as np
import scipy.special as sp


# Reactor data - Success or failure

# Features: Range

# 1. temperature: 400 - 700 Kelvin
temperature_lower = 400
temperature_upper = 700

# 2. pressure: 1-50 bar
pressure_lower = 1
pressure_upper = 50

# 3. feed flow rate: 50-200 kmol/hr
fflowrate_lower = 50
fflowrate_upper = 200

# 4. coolant flow rate: 1000-3600 L/hr
cflowrate_lower = 1000
cflowrate_upper = 3600

# 5. inlet concentration of reactant: 0.1-0.5 mol fraction
molfraction_lower = 0.1
molfraction_upper = 0.5

# number of data points
n = 1000

temperature_data = np.random.uniform(temperature_lower,temperature_upper,n)
temperature_data = np.ceil(temperature_data*100)/float(100)

pressure_data = np.random.uniform(pressure_lower,pressure_upper,n)
pressure_data = np.ceil(pressure_data*100)/float(100)

fflowrate_data = np.random.uniform(fflowrate_lower,fflowrate_upper,n)
fflowrate_data = np.ceil(fflowrate_data*100)/float(100)

cflowrate_data = np.random.uniform(cflowrate_lower,cflowrate_upper,n)
cflowrate_data = np.ceil(cflowrate_data*100)/float(100)

molfraction_data = np.random.uniform(molfraction_lower,molfraction_upper,n)
molfraction_data = np.ceil(molfraction_data*10000)/float(10000)

dataset = np.transpose(np.array([temperature_data,pressure_data,fflowrate_data,cflowrate_data,molfraction_data]))

# Parameters of the logistic regression model

beta = 3.5
w = np.array([0.002,0.025,0.01,-0.0033,1])

class_label = np.floor(2*sp.expit(np.dot(dataset,w)+beta))

for i in range(0,class_label.size):
    
    randnum = np.random.uniform(0,1)
    
    if randnum < 0.05:
        
        class_label[i] = not class_label[i]
        
print np.sum(class_label)

np.savetxt('q1_data_matrix.csv',dataset,delimiter = ',')
np.savetxt('q1_labels.csv',class_label,delimiter = ',')


# Credit Card Fraud - Fraud or not fraud

# Features: Range

# 1. Age: 18-100 years
age_lower = 18
age_upper = 100

# 2. Transaction Amount: 0-5000 dollars
amount_lower = 0
amount_upper = 5000

# 3. Total Monthly Transactions: 0-50000 dollars
monthly_lower = 0
monthly_upper = 50000

# 4. Annual Income: 30000-1000000 dollars
income_lower = 30000
income_upper = 1000000

# 5. Gender: 0/1 Male/Female
gender_lower = 0.01
gender_upper = 2

# number of data points
n = 1000

age_data = np.floor(np.random.uniform(age_lower,age_upper,n))

amount_data = np.floor(np.random.uniform(amount_lower,amount_upper,n))

monthly_data = np.floor(np.random.uniform(monthly_lower,monthly_upper,n))

income_data = np.floor(np.random.uniform(income_lower,income_upper,n)/100)*100

gender_data = np.floor(np.random.uniform(gender_lower,gender_upper,n))

dataset2 = np.transpose(np.array([age_data,amount_data,monthly_data,income_data,gender_data]))

# add quadratic terms: x1x2, x2x3, x3x4, x1x4, x1^2, x2^2, x3^2, x4^2

extensions = np.transpose(np.array([age_data*amount_data,amount_data*monthly_data,monthly_data*income_data,age_data*income_data,age_data*age_data,amount_data*amount_data,monthly_data*monthly_data,income_data*income_data]))

extended_set = np.concatenate((dataset2,extensions),axis=1)

# Parameters of the logistic regression model

beta = -25
w = np.array([0.02,0.00025,0.00004,0.00001,0.5,1.5*0.00005,1.3*0.0000001,1.7*0.000000004,2*0.000002,-0.004,0.000000625,0.000000016,-0.000000001])

# print np.dot(extended_set,w)+beta
print np.mean(np.dot(extended_set,w)+beta)

class_label2 = 0.5*(np.sign(np.dot(extended_set,w)+beta)+1)

for i in range(0,class_label2.size):
    
    randnum = np.random.uniform(0,1)
    
    if randnum < 0.05:
        
        class_label2[i] = not class_label2[i]
        
print np.sum(class_label2)

np.savetxt('q2_data_matrix.csv',dataset2,delimiter = ',')
np.savetxt('q2_labels.csv',class_label2,delimiter = ',')