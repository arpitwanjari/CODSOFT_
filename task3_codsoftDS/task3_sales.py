#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


data = pd.read_csv('Advertising.csv')


# In[3]:


sns.pairplot(data)
plt.show()


# In[4]:


for column in data.columns:
    sns.histplot(data[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()


# In[5]:


X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[8]:


y_pred = model.predict(X_test)


# In[9]:


mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)


# In[10]:


print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared: {r_squared}")


# In[11]:


metrics = ['Mean Squared Error (MSE)', 'R-squared']
values = [mse, r_squared]

plt.figure(figsize=(8, 6))
plt.bar(metrics, values, color=['blue', 'green'])
plt.title('Model Evaluation Metrics')
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.show()


# In[ ]:




