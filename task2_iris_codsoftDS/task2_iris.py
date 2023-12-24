#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv('iris.csv')


# In[4]:


print(data.columns)


# In[5]:


sns.pairplot(data, hue='Species')
plt.show()


# In[6]:


# Separate features and target variable
X = data.iloc[:, 1:5]  # Features: sepal length, sepal width, petal length, petal width
y = data['Species']    # Target variable: species

# Encoding the categorical target variable into numerical values
le = LabelEncoder()
y = le.fit_transform(y)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


svm_model = SVC(kernel='linear', C=1.0, random_state=42)

# Train the model
svm_model.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_model.predict(X_test)


# In[8]:


accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_)


# In[9]:


print(f"Accuracy: {accuracy}")
print("\nClassification Report:")
print(report)


# In[ ]:





# In[ ]:




