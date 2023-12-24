#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# In[2]:


data = pd.read_csv('tested.csv')


# In[3]:


sns.pairplot(data, hue='Survived')
plt.show()


# In[4]:


for column in data.columns:
    sns.histplot(data[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()


# In[5]:


data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
data.dropna(inplace=True)


# In[6]:


data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)


# In[7]:


X = data.drop('Survived', axis=1)
y = data['Survived']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# In[9]:


y_pred = model.predict(X_test)


# In[10]:


accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("\nClassification Report:")
print(report)


# In[11]:


accuracy = accuracy_score(y_test, y_pred)
percentage_accuracy = accuracy * 100
print(f"Accuracy: {percentage_accuracy:.2f}%")


# In[ ]:




