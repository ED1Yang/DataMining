# -*- coding: utf-8 -*-

# INFOSYS 722
# DATA MINING AND BIG DATA
#
# Iteration III - OSAS
#
# Meteorological effect on Air quality in Beijing
#
# By Tianyi Yang
# Tyan227


#import packages:
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold #For K−fold cross validation from sklearn .ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import r2_score


data = pd.read_csv("/Users/edward/Desktop/Master-S1/INFOSYS 722/assignments/Iteration Resubmit/Iteration3-OSAS/raw-data.csv")


# 2 Data verification
data.shape
data.describe()
display(data)

# Function to check the missing values:
def num_missing(x):
  return sum(x.isnull())

# Function to check the '-':
def num_empty(x):
  return sum(x=='-')

# Applying per column:
print ("Missing values per column:")
print (data.apply(num_missing, axis=0)) #axis=0 defines that function is to be applied on each column

print ("Empty values per column:")
print (data.apply(num_empty, axis=0)) 

# ---------------
# 3 Data Preparation
# change "-" to null:
data.replace('-',np.NaN,inplace=True)
print (data.apply(num_empty, axis=0)) 
display(data)
# modify binary rows:
data.RA.replace(np.NaN,0,inplace=True)
data.RA.replace('o',1,inplace=True)

data.SN.replace(np.NaN,0,inplace=True)
data.SN.replace('o',1,inplace=True)

data.FG.replace(np.NaN,0,inplace=True)
data.FG.replace('o',1,inplace=True)

data.TS.replace(np.NaN,0,inplace=True)
data.TS.replace('o',1,inplace=True)

display(data[['RA','TS','SN','FG']])
# quality check again:
print ("Missing values per column:")
print (data.apply(num_missing, axis=0)) #axis=0 defines that function is to be applied on each column

print ("Empty values per column:")
print (data.apply(num_empty, axis=0)) 

# outliers check
data.boxplot()
data[['AQI']].boxplot()

# alternative way using seaborn
f,ax=plt.subplots(figsize=(10,8))
sns.boxplot(y='AQI',data=data,ax=ax)
plt.show()


# ---------------
# 4. Data transformation

# delete columns:
df=data.drop(['TM','Tm','VM','VG','SLP','VV','PP','Month','Day','AQI Ranking'],axis=1)
df.head(10)

# remove rows with null values:
df=df.dropna()
df.head(10)
df.shape

# change orders
df=df[['Date','T','H','V','RA','SN','TS','FG','PM2.5','PM10','So2','No2','Co','O3','AQI','AQI Quality']]
df.head(10)
# projec the data
df.hist()

# change data types
df.dtypes

df['T'] = df['T'].astype('float64')
df['H'] = df['H'].astype('float64')
df['V'] = df['V'].astype('float64')


df['RA'] = df['RA'].astype(bool)
df['SN'] = df['SN'].astype(bool)
df['TS'] = df['TS'].astype(bool)
df['FG'] = df['FG'].astype(bool)

df.dtypes
df.info()

# ---------------
# 6. Data Mining algorithms selection:


#Regressions:
X = df[['T','H','V','PM2.5','PM10','So2','No2','Co','O3']]
X.head()
Y=df[['AQI']]
#split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=1)

linreg = LinearRegression()
linreg.fit(X_train, Y_train)
# get intercept and coefficients
print (linreg.intercept_)
print (linreg.coef_)

Y_pred = linreg.predict(X_test)

print ("MSE:",metrics.mean_squared_error(Y_test, Y_pred))
print ("RMSE:",np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
print ("R2:",r2_score(Y_test, Y_pred))

df['AQI'].describe()

# visualized prediction result:
fig, ax = plt.subplots()
ax.scatter(Y_test, Y_pred)
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=9)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

#Classifications: 


def cmodel(model,data,predictors,outcome):
    #split data:
    p_train, p_test, o_train, o_test = train_test_split(data[predictors], data[outcome], test_size = 0.3, random_state=1)
    #Fit the model:
    model.fit(p_train,o_train)
    #Make predictions on training set :
    predictions = model.predict(p_test)
    #Print accuracy
    accuracy = metrics.accuracy_score(predictions,o_test) 
    print ("Accuracy:%s" % "{0:.3%}".format(accuracy))
    #Perform k−fold cross−validation with 5 folds 
    kf = KFold(data.shape[0], n_folds=5)
    error = []
    # Continued ..
    for train, test in kf: 
        # Filter training data
        train_predictors = (data[predictors].iloc[train,:])
        # The target we’re using to train the algorithm .
        train_target = data[outcome].iloc[train]
        # Training the algorithm using the predictors and target .
        model.fit(train_predictors,train_target)
        #Record error from each cross−validation run
        error.append(model.score(data[predictors].iloc[test,:],data[outcome].iloc[test]))
    print ("Cross−Validation Score : %s" % "{0:.3%}".format(np.mean(error)))
      
# try different models:
predictor=['RA','SN','TS','FG']
df[df['AQI']>300][predictor].describe()

cmodel(LogisticRegression(), df,predictor,'AQI')
cmodel(DecisionTreeClassifier(), df,predictor,'AQI')
cmodel(RandomForestClassifier(), df,predictor,'AQI')

# ---------------
# 8 Interpretation
# visualization:
# linear:

# get scatter plot of AQI and other factors:
def draw(name):
    fig, ax = plt.subplots()
    ax.scatter(df[name], df['AQI'])
    ax.set_xlabel(name)
    ax.set_ylabel('AQI')
    plt.show()

draw('PM2.5')
draw('PM10')
draw('V')
draw('H')

# classifications:
g3 = pd.crosstab(df['RA'], df['AQI Quality']) 
g3.plot(kind='bar', stacked=True, grid=False)

g4 = pd.crosstab(df['TS'], df['AQI Quality']) 
g4.plot(kind='bar', stacked=True, grid=False)

g1 = pd.crosstab(df['SN'], df['AQI Quality']) 
g1.plot(kind='bar', stacked=True, grid=False)

g2 = pd.crosstab(df['FG'], df['AQI Quality']) 
g2.plot(kind='bar', stacked=True, grid=False)

# 8.5 Iterate prior steps:
# check meteorological factors effect on AQI:

X1 = df[['T','H','V']]
X1.head()
Y1=df[['AQI']]
#split data
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size = 0.3, random_state=1)

linreg1 = LinearRegression()
linreg1.fit(X1_train, Y1_train)
# get intercept and coefficients
print (linreg1.intercept_)
print (linreg1.coef_)

Y1_pred = linreg1.predict(X1_test)

print ("MSE:",metrics.mean_squared_error(Y1_test, Y1_pred))
print ("RMSE:",np.sqrt(metrics.mean_squared_error(Y1_test, Y1_pred)))
print ("R2:",r2_score(Y1_test, Y1_pred))

df['AQI'].describe()
