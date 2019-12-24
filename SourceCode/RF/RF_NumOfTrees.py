from DataPreProcess import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
#numberOfTrees = [1,2,3,4,5,6,7,8,9,10]
sco = []
mse = []
r2 = []
numberOfTrees = [1,5,1, 50, 100,150,200,250,300]
for num in numberOfTrees:
    rf = RandomForestRegressor(n_estimators=num, random_state=0,)
    rf.fit(X_train, y_train)
    sco.append(rf.score)
    y_pred = rf.predict(X_test)
    score = cross_val_score(rf,X_train,y_train,cv=5,scoring='r2')
    r2.append(np.mean(score))
    mseScore=cross_val_score(rf,X_train,y_train,cv=5,scoring='neg_mean_squared_error')
    mse.append(np.mean(mseScore))

print('mse are:', mse)
print('r2 scores are:',r2)

def bestMSE(mse):
    best_mse= min(mse)

    ind = mse.index(min(mse))
    return best_mse,ind

def bestR2Score(r2):
    best_r2= min(r2)

    ind = r2.index(max(r2))
    return best_r2,ind
best_mse,ind=bestMSE(mse)
a=ind
print('The best mse is',  best_mse, 'for ',numberOfTrees[a], 'number of trees')
bestR2,index=bestR2Score(r2)
j = index
print('The best r2 score is', bestR2, 'for ',numberOfTrees[j], 'number of trees')
newMse = mse
for i, j in enumerate(mse):
    newMse[i] =j * -1

print('new mse is:,',newMse)

plt.plot(numberOfTrees, newMse,label='mse')
plt.plot(numberOfTrees, r2, label='r2 score')
#plt.plot(x, y ,'ro')
#plt.axis([0,2.0,0, 15.0])
plt.xlabel('Number of trees')
plt.ylabel('r2 score and MSE')
plt.title(' Random Forest Regression with Different n_estimators ')
plt.legend()
plt.show()
