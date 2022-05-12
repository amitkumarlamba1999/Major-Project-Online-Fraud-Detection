import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix

# importing dataset
data = pd.read_csv("datasets/phishcoop.csv")
data = data.drop('id', 1)

# features, labels
x = data.iloc[ :, :-1].values
y = data.iloc[ :, -1:].values

# splitting into training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# ____________________ finding best parameters____________________________________________
parameters = [{'n_estimators' : [100, 70],
               'max_features' : ['sqrt', 'log2'],
               'criterion' : ['gini', 'entropy']}]
grid_search = GridSearchCV(RandomForestClassifier(), parameters, cv=5, n_jobs=-1)
grid_search.fit(x_train, y_train)
print("best accuracy = "+ str(grid_search.best_score_))
print("best parameters = "+ str(grid_search.best_params_))
# ________________________________________________________________________________________

# fitting RandomForest classifier with best parameters
classifier = RandomForestClassifier(n_estimators=100, criterion='gini', max_features='log2', random_state=0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
# print(classifier.predict_proba(x_test))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# pickle file joblib
joblib.dump(classifier, 'rf_final.pkl')

# -------------Features Importance random forest
names = data.iloc[:, :-1].columns
importances = classifier.feature_importances_
sorted_importances = sorted(importances, reverse=True)
indices = np.argsort(-importances)
var_imp = pd.DataFrame(sorted_importances, names[indices], columns=['importance'])



# -------------plotting variable importance
plt.title("Variable Importances")
plt.barh(np.arange(len(names)), sorted_importances, height=0.6)
plt.yticks(np.arange(len(names)), names[indices], fontsize=7)
plt.xlabel('Relative Importance')
plt.show()
plt.savefig(fname='fig.png')