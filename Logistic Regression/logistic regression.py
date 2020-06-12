#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.linear_model import LogisticRegression


# In[3]:


titanic = sns.load_dataset('titanic')


# In[4]:


titanic.head()


# In[5]:


titanic.describe()


# ## Understanding Data

# In[6]:


titanic.isnull().sum()


# In[7]:


sns.heatmap(titanic.isnull(), cbar=False)


# In[8]:


titanic['age'].isnull().sum()/titanic.shape[0]*100


# In[9]:


ax = titanic['age'].hist(bins=30, density=True, stacked=True, color='teal', alpha=0.7, figsize=(16,5))
titanic['age'].plot(kind='density', color='red')
ax.set_label("Age")
plt.show()


# In[10]:


s='survived'
ns='not survived'

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
women = titanic[titanic['sex']=='female']
men = titanic[titanic['sex']=='male']

ax = sns.distplot(women[women[s]==1].age.dropna(), bins=18, label=s, ax=axes[0], kde=False)
ax = sns.distplot(women[women[s]==0].age.dropna(), bins=40, label=ns, ax=axes[0], kde=False)
ax.legend()
ax.set_title('Female')

ax2 = sns.distplot(men[men[s]==1].age.dropna(), bins=18, label=s, ax=axes[1], kde=False)
ax2 = sns.distplot(men[men[s]==0].age.dropna(), bins=40, label=ns, ax=axes[1], kde=False)
ax2.legend()
ax2.set_title('Male')


# In[11]:


titanic['sex'].value_counts()


# In[12]:


sns.catplot(x='pclass', y='age', data=titanic, kind='box')


# In[13]:


sns.catplot(x='pclass', y='fare', data=titanic, kind='box')


# In[14]:


titanic[titanic['pclass']==1]['age'].mean()


# In[15]:


titanic[titanic['pclass']==2]['age'].mean()


# In[16]:


titanic[titanic['pclass']==3]['age'].mean()


# In[17]:


def impute_age(cols):
    age=cols[0]
    pclass=cols[1]
    
    if pd.isnull(age):
        if pclass==1:
            return titanic[titanic['pclass']==1]['age'].mean()
        elif pclass==2:
            return titanic[titanic['pclass']==2]['age'].mean()
        elif pclass==3:
            return titanic[titanic['pclass']==3]['age'].mean()
        
    else:
        return age


# In[18]:


titanic['age'] = titanic[['age', 'pclass']].apply(impute_age, axis = 1)


# In[19]:


sns.heatmap(titanic.isnull(), cbar=False, cmap='viridis')


# ## Analysis Embarked

# In[20]:


f = sns.FacetGrid(titanic, row='embarked', height=2.5, aspect=3)
f.map(sns.pointplot, 'pclass', 'survived', 'sex', order=None, hue_order=None)
f.add_legend()


# In[21]:


titanic['embarked'].isnull().sum()


# In[22]:


titanic['embark_town'].value_counts()


# In[23]:


common = 'S'
titanic['embarked'].fillna(common, inplace=True)


# In[24]:


sns.heatmap(titanic.isnull(), cbar=False)


# In[25]:


titanic.drop(labels=['deck', 'embark_town', "alive"], inplace=True, axis=1)


# In[26]:


sns.heatmap(titanic.isnull(), cbar=False)


# In[27]:


titanic.info()


# In[28]:


titanic.head()


# In[29]:


titanic['fare'] = titanic['fare'].astype('int')
titanic['age'] = titanic['age'].astype('int')
titanic['pclass'] = titanic['pclass'].astype('int')
titanic.info()


# ## Convert catagorical data to numerical data

# In[30]:


genders = {'male': 0, 'female': 1}
titanic['sex'] = titanic['sex'].map(genders)


# In[31]:


titanic.head()


# In[32]:


who = {'man':0, 'woman':1, "child":2}
titanic["who"] = titanic['who'].map(who)


# In[33]:


admale = {True:1, False:0}
titanic['adult_male'] = titanic['adult_male'].map(admale)


# In[34]:


alone = {True:0, False:1}
titanic['alone'] = titanic['alone'].map(alone)


# In[35]:


ports = {'S':0, 'C':1, 'Q':2}
titanic['embarked'] = titanic['embarked'].map(ports)


# In[36]:


titanic.head()


# In[37]:


titanic.drop(labels = ['class', 'who'], axis=1, inplace=True)


# In[38]:


titanic.head()


# ### Build logistic regression model

# In[39]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[40]:


X = titanic.drop('survived', axis=1)
y = titanic['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[41]:


X_train.shape


# In[43]:


model = LogisticRegression(solver='lbfgs', max_iter = 400)
model.fit(X_train, y_train)
y_predict= model.predict(X_test)


# In[44]:


model.score(X_test, y_test)

