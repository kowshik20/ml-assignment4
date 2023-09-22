#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

data = pd.read_excel('Custom_CNN_Features1.xlsx')

X = data.drop(columns=['Filename', 'Label'])
y = data['Label']

Tr_X, Te_X, Tr_y, Te_y = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()

model.fit(Tr_X, Tr_y)

train_accuracy = model.score(Tr_X, Tr_y)
print("Training Set Accuracy:", train_accuracy)

test_accuracy = model.score(Te_X, Te_y)
print("Test Set Accuracy:", test_accuracy)

class_names = [str(label) for label in class_names]

plt.figure(figsize=(70, 20))
plot_tree(model, filled=True, feature_names=X.columns.tolist(), class_names=class_names)
plt.show()


# In[10]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

data = pd.read_excel('Custom_CNN_Features1.xlsx')

X = data.drop(columns=['Filename', 'Label'])
y = data['Label'].astype(str)  # Convert class labels to strings

Tr_X, Te_X, Tr_y, Te_y = train_test_split(X, y, test_size=0.2, random_state=42)

class_names = data['Label'].unique().astype(str)  # Convert unique class labels to strings

model = DecisionTreeClassifier(max_depth=5)

model.fit(Tr_X, Tr_y)

train_accuracy = model.score(Tr_X, Tr_y)
print("Training Set Accuracy with Max Depth Constraint:", train_accuracy)

test_accuracy = model.score(Te_X, Te_y)
print("Test Set Accuracy with Max Depth Constraint:", test_accuracy)

plt.figure(figsize=(70, 20))
plot_tree(model, filled=True, feature_names=X.columns.tolist(), class_names=class_names.tolist())
plt.show()


# In[11]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

data = pd.read_excel('Custom_CNN_Features1.xlsx')

X = data.drop(columns=['Filename', 'Label'])
y = data['Label'].astype(str)

Tr_X, Te_X, Tr_y, Te_y = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(criterion="entropy", max_depth=5)

model.fit(Tr_X, Tr_y)

train_accuracy = model.score(Tr_X, Tr_y)
print("Training Set Accuracy with Entropy Criterion:", train_accuracy)

test_accuracy = model.score(Te_X, Te_y)
print("Test Set Accuracy with Entropy Criterion:", test_accuracy)

plt.figure(figsize=(70, 20))
plot_tree(model, filled=True, feature_names=X.columns.tolist(), class_names=class_names.tolist())
plt.show()


# In[13]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Read data from the Excel file
data = pd.read_excel('Custom_CNN_Features1.xlsx')

# Drop non-numeric columns
data = data.select_dtypes(include=[float, int])

X = data.drop(columns=['Label'])
y = data['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

y_pred_decision_tree = decision_tree.predict(X_test)
decision_tree_accuracy = accuracy_score(y_test, y_pred_decision_tree)

y_pred_random_forest = random_forest.predict(X_test)
random_forest_accuracy = accuracy_score(y_test, y_pred_random_forest)

print("Decision Tree Accuracy:", decision_tree_accuracy)
print("Random Forest Accuracy:", random_forest_accuracy)

print("Decision Tree Classification Report:\n", classification_report(y_test, y_pred_decision_tree))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_random_forest))

print("Decision Tree Confusion Matrix:\n", confusion_matrix(y_test, y_pred_decision_tree))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_random_forest))


# In[ ]:




