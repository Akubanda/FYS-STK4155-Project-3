from DataPreProcess import *

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
mse = []
r2 = []
r2score = []
newMse=[]
#add more degrees in the list
#polyDegree = [1,2,3,4,5]
polyDegree = [1,2]
for deg in polyDegree:
    poly = PolynomialFeatures(degree = deg)
    X_train = poly.fit_transform(X_train)
    X_test = poly.fit_transform(X_test)
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_tilde= model.predict(X_train)
    y_pred = model.predict(X_test)
    #train_error = mean_squared_error(y_train,y_tilde)
    #test_error= mean_squared_error(y_test,y_pred)
    mse.append(mean_squared_error(y_test,y_pred))
    r2.append(r2_score(y_test,y_pred))
    '''
    r2=cross_val_score(model,y_test,y_pred,cv=5,scoring='r2')
    a =np.mean(r2)
    r2score.append()
    mseScore=cross_val_score(model,y_test,y_pred,cv=5,scoring='neg_mean_squared_error')
    newMse.append(np.mean(mseScore))'''
print('without cross validation:')
print('Test mse:',mse,'r2 score:',r2)
