#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score


# In[6]:


data=pd.read_csv('Iris.csv')


# Visualization of Data

# In[7]:


data.head()


# In[8]:


data.head(20)


# In[9]:


data.tail()


# In[10]:


data.describe()


# In[11]:


data.size


# In[12]:


data.shape


# In[14]:


data.dtypes


# In[15]:


data.columns


# In[17]:


data.groupby('PetalWidthCm').size()


# In[19]:


data['PetalWidthCm'].unique().tolist()


# In[20]:


data['Species'].unique().tolist()


# In[21]:


data.isnull()


# In[22]:


data.dropna()


# In[23]:


data.info()


# In[24]:


data.isnull().sum()


# # DATA VISUALIZATION
# 

# In[25]:


plt.boxplot(df['SepalLengthCm'])


# In[26]:


plt.boxplot(df['PetalWidthCm'])


# In[27]:


sns.heatmap(df.corr())


# # EXPLORATORY DATA ANALYSIS

# In[29]:


nameplot = data['Species'].value_counts().plot.bar(title='Flower Species Classification')
nameplot.set_xlabel('Flower',size=30)
nameplot.set_ylabel('count',size=30)


# In[30]:


sns.pairplot(data, hue='Species')


# In[32]:


data.hist()


# # splitting of data

# In[35]:


x= data.drop("Species", axis=1)
y= data["Species"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size= 0.2, random_state=0)


# In[36]:


print("X_train.shape:", x_train.shape)
print("X_test.shape:", x_test.shape)
print("Y_train.shape:", y_train.shape)
print("Y_test.shape:", y_test.shape)


# # Model Selection and Prediction

# In[37]:


from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
x_new= np.array([[151, 5, 2.9, 1, 0.2]])
prediction= knn.predict(x_new)
print("Prediction: {}".format(prediction))


# In[38]:


model= KNeighborsClassifier()
model.fit(x_train,y_train)
model.score(x_train, y_train)


# In[39]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[41]:


# reusable function to test our model
def test_model(model):
    model.fit(x_train, y_train) # train the whole training set
    predictions = model.predict(x_test) # predict on test set
    
    # output model testing results
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("Classification Report:")
    print(classification_report(y_test, predictions))


# In[42]:


test_model(model)


# In[ ]:




