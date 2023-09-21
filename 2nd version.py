#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
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

plt.figure(figsize=(70, 20))
plot_tree(model, filled=True, feature_names=X.columns.tolist(), class_names=[str(label) for label in model.classes_])
plt.show()


# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

data = pd.read_excel('Custom_CNN_Features1.xlsx')

X = data.drop(columns=['Filename', 'Label'])
y = data['Label'].astype(str)  

Tr_X, Te_X, Tr_y, Te_y = train_test_split(X, y, test_size=0.2, random_state=42)

class_names = data['Label'].unique().astype(str)  

model = DecisionTreeClassifier(max_depth=5)

model.fit(Tr_X, Tr_y)

train_accuracy = model.score(Tr_X, Tr_y)
print("Training Set Accuracy with Max Depth Constraint:", train_accuracy)

test_accuracy = model.score(Te_X, Te_y)
print("Test Set Accuracy with Max Depth Constraint:", test_accuracy)

plt.figure(figsize=(70, 20))
plot_tree(model, filled=True, feature_names=X.columns.tolist(), class_names=class_names.tolist())
plt.show()


# In[ ]:




