#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load data from an Excel file named 'Custom_CNN_Features1.xlsx'
data = pd.read_excel('Custom_CNN_Features1.xlsx')

# Separate the features (X) and the target labels (y)
X = data.drop(columns=['Filename', 'Label'])
y = data['Label'].astype(str)  # Convert class labels to strings

# Split the dataset into training and testing sets using a 80-20 split ratio
Tr_X, Te_X, Tr_y, Te_y = train_test_split(X, y, test_size=0.2, random_state=42)

# Get unique class labels and convert them to strings
class_names = data['Label'].unique().astype(str)

# Create a Decision Tree Classifier with a maximum depth of 5
model = DecisionTreeClassifier(max_depth=5)

# Fit the model on the training data
model.fit(Tr_X, Tr_y)

# Calculate and print the training set accuracy
train_accuracy = model.score(Tr_X, Tr_y)
print("Training Set Accuracy with Max Depth Constraint:", train_accuracy)

# Calculate and print the test set accuracy
test_accuracy = model.score(Te_X, Te_y)
print("Test Set Accuracy with Max Depth Constraint:", test_accuracy)

# Create a large figure for plotting the decision tree
plt.figure(figsize=(70, 20))

# Plot the decision tree with filled nodes and labels
plot_tree(model, filled=True, feature_names=X.columns.tolist(), class_names=class_names.tolist())

# Display the decision tree plot
plt.show()


# In[3]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load data from an Excel file named 'Custom_CNN_Features1.xlsx'
data = pd.read_excel('Custom_CNN_Features1.xlsx')

# Separate the features (X) and the target labels (y)
X = data.drop(columns=['Filename', 'Label'])
y = data['Label'].astype(str)

# Split the dataset into training and testing sets using an 80-20 split ratio and a fixed random seed
Tr_X, Te_X, Tr_y, Te_y = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree Classifier with entropy as the splitting criterion and a maximum depth of 5
model = DecisionTreeClassifier(criterion="entropy", max_depth=5)

# Fit the model on the training data
model.fit(Tr_X, Tr_y)

# Calculate and print the training set accuracy with entropy criterion
train_accuracy = model.score(Tr_X, Tr_y)
print("Training Set Accuracy with Entropy Criterion:", train_accuracy)

# Calculate and print the test set accuracy with entropy criterion
test_accuracy = model.score(Te_X, Te_y)
print("Test Set Accuracy with Entropy Criterion:", test_accuracy)

# Create a large figure for plotting the decision tree
plt.figure(figsize=(70, 20))

# Plot the decision tree with filled nodes and labels
plot_tree(model, filled=True, feature_names=X.columns.tolist(), class_names=class_names.tolist())
plt.show()
# Display the decision tree plot


# In[4]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Read data from the Excel file 'Custom_CNN_Features1.xlsx'
data = pd.read_excel('Custom_CNN_Features1.xlsx')

# Drop non-numeric columns to keep only numeric features
data = data.select_dtypes(include=[float, int])

# Separate features (X) and target labels (y)
X = data.drop(columns=['Label'])
y = data['Label']

# Split the dataset into training and testing sets with an 80-20 split ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree Classifier with a fixed random state
decision_tree = DecisionTreeClassifier(random_state=42)

# Fit the Decision Tree model on the training data
decision_tree.fit(X_train, y_train)

# Create a Random Forest Classifier with 100 trees and a fixed random state
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the Random Forest model on the training data
random_forest.fit(X_train, y_train)

# Make predictions using the Decision Tree model on the test data
y_pred_decision_tree = decision_tree.predict(X_test)

# Calculate the accuracy of the Decision Tree model
decision_tree_accuracy = accuracy_score(y_test, y_pred_decision_tree)

# Make predictions using the Random Forest model on the test data
y_pred_random_forest = random_forest.predict(X_test)

# Calculate the accuracy of the Random Forest model
random_forest_accuracy = accuracy_score(y_test, y_pred_random_forest)

# Print the accuracy of both models
print("Decision Tree Accuracy:", decision_tree_accuracy)
print("Random Forest Accuracy:", random_forest_accuracy)

# Print classification reports for both models
print("Decision Tree Classification Report:\n", classification_report(y_test, y_pred_decision_tree))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_random_forest))

# Print confusion matrices for both models
print("Decision Tree Confusion Matrix:\n", confusion_matrix(y_test, y_pred_decision_tree))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_random_forest))


# In[6]:


# Import the math module for logarithmic calculations
import math

# Define a function to calculate the entropy of a set of probabilities
def entropy(probabilities):
    return -sum(p * math.log2(p) if p > 0 else 0 for p in probabilities)

# Define a function to calculate the information gain for a given attribute
def information_gain(data, attribute_index, target_index):
    # Calculate the total number of instances in the dataset
    total_instances = len(data)
    
    # Get unique target values in the dataset
    target_values = set(data[i][target_index] for i in range(total_instances))
    
    # Calculate the probabilities of each unique target value
    target_probabilities = [sum(1 for row in data if row[target_index] == value) / total_instances for value in target_values]
    
    # Calculate the entropy of the target variable before the split
    entropy_before = entropy(target_probabilities)
    
    # Get unique values of the attribute being considered
    attribute_values = set(data[i][attribute_index] for i in range(total_instances))
    
    # Initialize the weighted entropy after the split
    weighted_entropy_after = 0
    
    # Calculate the weighted entropy for each attribute value
    for value in attribute_values:
        subset = [row for row in data if row[attribute_index] == value]
        subset_size = len(subset)
        
        # Calculate the probabilities of each unique target value within the subset
        subset_target_probabilities = [sum(1 for row in subset if row[target_index] == target_value) / subset_size for target_value in target_values]
        
        # Calculate the weighted entropy after the split
        weighted_entropy_after += (subset_size / total_instances) * entropy(subset_target_probabilities)
    
    # Calculate the information gain as the difference between entropy before and after the split
    information_gain = entropy_before - weighted_entropy_after
    return information_gain

# Define the dataset as a list of lists
data = [
    ["<=30", "high", "no", "fair", "no"],
    ["<=30", "high", "no", "excellent", "no"],
    ["31…40", "high", "no", "fair", "yes"],
    [">40", "medium", "no", "fair", "yes"],
    [">40", "low", "yes", "fair", "yes"],
    [">40", "low", "yes", "excellent", "no"],
    ["31…40", "low", "yes", "excellent", "yes"],
    ["<=30", "medium", "no", "fair", "no"],
    ["<=30", "low", "yes", "fair", "yes"],
    [">40", "medium", "yes", "fair", "yes"],
    ["<=30", "medium", "yes", "excellent", "yes"],
    ["31…40", "medium", "no", "excellent", "yes"],
    ["31…40", "high", "yes", "fair", "yes"],
    [">40", "medium", "no", "excellent", "no"]
]
# Specify the index of the target variable in the dataset
target_index = 4

# Specify the attributes being considered
attributes = ["age", "income", "student", "credit_rating"]

# Initialize a dictionary to store information gains for each attribute
information_gains = {}

# Calculate information gains for each attribute and store them in the dictionary
for attribute_index, attribute in enumerate(attributes):
    gain = information_gain(data, attribute_index, target_index)
    information_gains[attribute] = gain

# Find the root attribute with the maximum information gain
root_attribute = max(information_gains, key=information_gains.get)
root_information_gain = information_gains[root_attribute]

# Print the root attribute and its information gain
print(f"The root node is '{root_attribute}' with Information Gain of {root_information_gain:.3f}")


# In[7]:


# Import necessary libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Define the dataset as a list of lists
data = [
    ["<=30", "high", "no", "fair", "no"],
    ["<=30", "high", "no", "excellent", "no"],
    ["31…40", "high", "no", "fair", "yes"],
    [">40", "medium", "no", "fair", "yes"],
    [">40", "low", "yes", "fair", "yes"],
    [">40", "low", "yes", "excellent", "no"],
    ["31…40", "low", "yes", "excellent", "yes"],
    ["<=30", "medium", "no", "fair", "no"],
    ["<=30", "low", "yes", "fair", "yes"],
    [">40", "medium", "yes", "fair", "yes"],
    ["<=30", "medium", "yes", "excellent", "yes"],
    ["31…40", "medium", "no", "excellent", "yes"],
    ["31…40", "high", "yes", "fair", "yes"],
    [">40", "medium", "no", "excellent", "no"]
]

# Define the column names for the dataset
columns = ["age", "income", "student", "credit_rating", "buys_computer"]

# Create a DataFrame from the data with column names
df = pd.DataFrame(data, columns=columns)

# Separate the features (X) and the target labels (y)
X = df.drop("buys_computer", axis=1)
y = df["buys_computer"]

# Define the list of categorical features
categorical_features = ["age", "income", "student", "credit_rating"]

# Create a preprocessor that applies one-hot encoding to categorical features
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(), categorical_features)],
    remainder="passthrough"
)

# Create a machine learning pipeline that includes preprocessing and a decision tree classifier
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier())
])

# Fit the pipeline on the data to train the decision tree classifier
pipeline.fit(X, y)

# Get the depth of the decision tree in the pipeline
tree_depth = pipeline.named_steps["classifier"].get_depth()

# Print the depth of the decision tree
print(f"Tree depth: {tree_depth}")


# In[8]:


# Import necessary libraries
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Define the dataset as a list of lists
data = [
    ["<=30", "high", "no", "fair", "no"],
    ["<=30", "high", "no", "excellent", "no"],
    ["31…40", "high", "no", "fair", "yes"],
    [">40", "medium", "no", "fair", "yes"],
    [">40", "low", "yes", "fair", "yes"],
    [">40", "low", "yes", "excellent", "no"],
    ["31…40", "low", "yes", "excellent", "yes"],
    ["<=30", "medium", "no", "fair", "no"],
    ["<=30", "low", "yes", "fair", "yes"],
    [">40", "medium", "yes", "fair", "yes"],
    ["<=30", "medium", "yes", "excellent", "yes"],
    ["31…40", "medium", "no", "excellent", "yes"],
    ["31…40", "high", "yes", "fair", "yes"],
    [">40", "medium", "no", "excellent", "no"]
]

# Define the column names for the dataset
columns = ["age", "income", "student", "credit_rating", "buys_computer"]

# Create a DataFrame from the data with column names
df = pd.DataFrame(data, columns=columns)

# Separate the features (X) and the target labels (y)
X = df.drop("buys_computer", axis=1)
y = df["buys_computer"]

# Define the list of categorical features
categorical_features = ["age", "income", "student", "credit_rating"]

# Create a preprocessor that applies one-hot encoding to categorical features
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(), categorical_features)],
    remainder="passthrough"
)

# Create a machine learning pipeline that includes preprocessing and a decision tree classifier
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier())
])

# Fit the pipeline on the data to train the decision tree classifier
pipeline.fit(X, y)

# Get the names of the transformed features
feature_names = list(pipeline.named_steps["preprocessor"].get_feature_names_out(input_features=categorical_features)) + list(X.columns.drop(categorical_features))

# Create a large figure for plotting the decision tree
plt.figure(figsize=(70, 20))

# Plot the decision tree with filled nodes and feature names
plot_tree(pipeline.named_steps["classifier"], filled=True, feature_names=feature_names, class_names=['no', 'yes'])

# Display the decision tree plot
plt.show()


# In[9]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Read data from the Excel file 'Custom_CNN_Features1.xlsx'
data = pd.read_excel('Custom_CNN_Features1.xlsx')

# Separate the features (X) and the target labels (y)
X = data.drop(columns=['Filename', 'Label'])
y = data['Label']

# Split the dataset into training and testing sets with an 80-20 split ratio and a fixed random seed
Tr_X, Te_X, Tr_y, Te_y = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree Classifier
model = DecisionTreeClassifier()

# Fit the model on the training data
model.fit(Tr_X, Tr_y)

# Calculate and print the training set accuracy
train_accuracy = model.score(Tr_X, Tr_y)
print("Training Set Accuracy:", train_accuracy)

# Calculate and print the test set accuracy
test_accuracy = model.score(Te_X, Te_y)
print("Test Set Accuracy:", test_accuracy)

# Create a large figure for plotting the decision tree
plt.figure(figsize=(70, 20))

# Plot the decision tree with filled nodes, feature names, and class names
plot_tree(model, filled=True, feature_names=X.columns.tolist(), class_names=[str(label) for label in model.classes_])

# Display the decision tree plot
plt.show()


# In[ ]:




