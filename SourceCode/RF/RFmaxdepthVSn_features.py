from DataPreProcess import *

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
#max features list has alll options available for mac_features parameter
max_featureL=['auto','sqrt','log2',19]
max_feat_names=['auto=n_features','sqrt[n_features]','log2[n_features]','n_features']
max_depthL = [None,1,2,3,4,5,6,7,8,9,10,11,]
i = len(max_depthL)
testerror = np.empty((4, i))
trainerror = np.empty((4, i))
mse = np.empty((4, i))
r2 = np.empty((4, i))
for f,feature in enumerate(max_featureL):
    for s,dep in enumerate(max_depthL):
        rf = RandomForestRegressor( n_estimators=100,max_features=feature,
                                    max_depth=dep,random_state=0)
        rf.fit(X_train, y_train)
        '''
        trainAccuracy = rf.score(X_train, y_train)
        testAccuracy = rf.score(X_test, y_test)
        ypred= rf.predict(y_test)
        ytilde=rf.predict(y_train)
        testerror[f,s] = mean_squared_error(y_test, ypred)
        trainerror[f,s] = mean_squared_error(y_train,ytilde)
        '''
        score = cross_val_score(rf,X_train,y_train,cv=5,scoring='r2')
        r2[f,s] = np.mean(score)
        mseScore=cross_val_score(rf,X_train,y_train,cv=5,scoring='neg_mean_squared_error')
        mse[f,s] = np.mean(mseScore)

newMse = mse
for i, j in enumerate(mse):
    newMse[i] =j * -1

print('new mse is:,',newMse)

sns.set()
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(mse, annot=True, ax=ax, cmap="viridis")
ax.set_title("Training MSE error for Different Maximum depths")
ax.set_ylabel("Maximum Features")
ax.set_xlabel("Max Depth")
ax.set_yticklabels(max_featureL)
ax.set_xticklabels(max_depthL)
plt.show()

sns.set()
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(r2, annot=True, ax=ax, cmap="viridis")
ax.set_title("Training r2 score for Different Maximum depths")
ax.set_ylabel("Maximum Features")
ax.set_xlabel("Max Depth")
ax.set_yticklabels(max_featureL)
ax.set_xticklabels(max_depthL)
plt.show()
