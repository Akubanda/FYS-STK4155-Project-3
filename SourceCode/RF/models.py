
#from NN_class import N_Network
from DataPreProcess import *
#from importData import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
#model = RandomForestClassifier(n_estimators=40)
#model = tree.DecisionTreeClassifier()

def fit(modelX):
    model = modelX
    model.fit(X_train, y_train)
    train_score = model.score(X_train,y_train)
    test_score = model.score(X_test,y_test)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_pred,y_test)
    r2 = r2_score(y_pred,y_test)
    return train_score, test_score, mse, r2



models = [LinearRegression(),Ridge(alpha = 1,random_state = None),linear_model.Lasso(alpha= 0.0001),
        RandomForestRegressor(n_estimators=100, max_depth = 8),DecisionTreeRegressor(random_state=0),
            AdaBoostRegressor(n_estimators=100, random_state=0)]
model_names = ['OLS','Ridge','Lasso', 'Random Forest', 'Decision Trees','AdaBoost']
train_score= []
test_score = []
MSEs = []
r2Scores =[]
for m, mod in enumerate(models):
    train_s, test_s,MSE,r2S = fit(mod)

    train_score.append(train_s)
    test_score.append(test_s)
    MSEs.append(MSE)
    r2Scores.append(r2S)

print(train_score)
print(test_score)
print('MSE scores:', MSEs)
print('r2 scores:',r2Scores)
fig, ax = plt.subplots(figsize=(10,10))
sns.set()
sns.set_style("whitegrid")

sns.axes_style("whitegrid")
sns.barplot( x= model_names, y= MSEs    )
ax.set_title("MSEs for Models")
ax.set_ylabel("MSe")
ax.set_xlabel("Model")
plt.show()

sns.set()
sns.set_style("whitegrid")

sns.axes_style("whitegrid")
sns.barplot( x= model_names, y= r2Scores    )
ax.set_title("R2 scores for Models")
ax.set_ylabel("R2 score")
ax.set_xlabel("Model")
plt.show()
