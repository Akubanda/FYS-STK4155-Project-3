from DataPreProcess import *

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor( n_estimators=100,max_features=19,
                    random_state=0)
rf.fit(X_train, y_train)
feature_importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(feature_importances)[::-1]
# Print the feature ranking
print("Feature ranking:")
for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], feature_importances[indices[f]]))

y_ticks = np.arange(0, len(feature_importances))
fig, ax = plt.subplots()
ax.barh(y_ticks, feature_importances[indices])
ax.set_yticklabels(indices)
ax.set_yticks(y_ticks)
ax.set_title("Random Forest Feature Importances ")
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
fig.tight_layout()
plt.show()
