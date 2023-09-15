#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#read csv, look at it
penguin_data = pd.read_csv("penguin_data.csv")
print(penguin_data.head())
print(penguin_data.info())
print(penguin_data.describe())

#drop unecessary rowid col:
penguin_data.drop(["rowid"],axis=1,inplace=True)
print(penguin_data.info())

#visualise NaN values: 
sns.heatmap(penguin_data.isnull(),yticklabels=False,cbar=False,cmap = 'viridis')#can see nan values acorss alot of columns, so drop them 
#visualise hwo many nan:
print(penguin_data.isnull().sum())

#drop nan:
penguin_data.dropna(inplace=True)
print(penguin_data.isnull().sum())
sns.heatmap(penguin_data.isnull(),yticklabels=False,cbar=False,cmap = 'viridis')

#dummies for sex column 
sex = pd.get_dummies(penguin_data['sex'], drop_first=True)
print(sex.head())#boolean

#turn true false into 0,1
sex = sex.astype(int)
print(sex.head(2))

#drop sex col from penguin data frame
penguin_data.drop(['sex'], inplace=True, axis=1)
print(penguin_data.head(2))

#concatenated sex and penguin_data
penguin_d = pd.concat([penguin_data, sex],axis=1) #no inplace here.
print(penguin_d.head(2))

#change species column to 0,1,2. #going to use label encoder:
from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder() #instantiate??
penguin_d["species"]=label_encoder.fit_transform(penguin_d["species"])
print(penguin_d.head(3))#check
penguin_d["species"].unique()#check 

#change island col to 0,1,2
from sklearn.preprocessing import LabelEncoder
label_encod=LabelEncoder() #instantiate??
penguin_d["island"]=label_encod.fit_transform(penguin_d["island"])
print(penguin_d.head(3))#check
penguin_d["island"].unique()#check

#now all data is numerical. #year col might not give much info so drop.
penguin_d.drop(["year"], axis=1, inplace=True)
print(penguin_d.head(3))

#interactive plotly plot 
import os 
import plotly.express as px 
import plotly.io as pio 
pio.renderers.default='browser'

fig = px.scatter_3d(penguin_data, x='bill_length_mm', y='flipper_length_mm', z='body_mass_g')
fig.show()

fig = px.scatter_3d(penguin_d, x='bill_length_mm', y='flipper_length_mm', z='body_mass_g')
fig.show()

#split up data = 0.2
from sklearn.model_selection import train_test_split
X=penguin_d.drop("species",axis=1)
y=penguin_d["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#SVM/Random forests:
    
#RF 1st:
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators = 200)
rfc.fit(X_train,y_train)
y_pred=rfc.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix 
print(confusion_matrix(y_test, y_pred))
print('\n')
print(classification_report(y_test, y_pred))
# accuracy of 1.00 means model made correct predictons 100% fo the time.

#SVM: 
from sklearn.svm import SVC
svc_model=SVC()#instantiate
svc_model.fit(X_train, y_train)
y_predSVM = svc_model.predict(X_test) 

#model evaluation
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_predSVM))
print('\n')
print(classification_report(y_test, y_predSVM))
#result= accuracy is 0.67

#grdisearch - hyperparameter tuning
from sklearn.model_selection import GridSearchCV
grid=GridSearchCV (SVC(),param_grid,verbose = 2)
grid.fit(X_train, y_train)#gridsearch object and fit it to data
param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
grid_pred = grid.predict(X_test)

#model evaluation 2.0
print(confusion_matrix(y_test,grid_pred))
print('\n')
print(classification_report(y_test, grid_pred))
#result = accuracy is 0.88.this is an improvement from previous model

#Overall Random Forest model performed better in comaprison to SVM.