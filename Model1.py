import pandas as pd

train_fea_names=pd.read_csv('train_labels.csv')
train_vals=pd.read_csv('train_values.csv')
test_set=pd.read_csv('test_values.csv')

#Data Cleaning / Pre-processing
#Train_set

cols=train_vals.iloc[:,[8,9,10,11,12,13,14,26]]
dumdum=pd.get_dummies(cols,columns=cols.columns)
d_cols=dumdum.iloc[:,[0,3,8,11,16,20,24,34]]
dumdum=dumdum.drop(axis=1,columns=d_cols.columns)

x=train_vals.drop(axis=1,columns=cols.columns)
xtrain=x.join(dumdum)
ytrain=train_fea_names.iloc[:,1]

#Scaling
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
xtrain=ss.fit_transform(xtrain)

#Test_set
dumdumt=pd.get_dummies(test_set.iloc[:,[8,9,10,11,12,13,14,26]],columns=cols.columns)
dumdumt=dumdumt.drop(axis=1,columns=d_cols.columns)
xtest=test_set.drop(axis=1,columns=cols.columns)
xtest=xtest.join(dumdumt)

xtest=ss.fit_transform(xtest)

#PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=20)
xtrain=pca.fit_transform(xtrain)
xtest=pca.fit_transform(xtest)

#Model
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100)
model.fit(xtrain,ytrain)

#Prediction
pred=model.predict(xtest)

#Submission
sub=pd.DataFrame(test_set.iloc[:,0])
sub['damage_grade']=pred
sub.to_csv('Submission1.csv')