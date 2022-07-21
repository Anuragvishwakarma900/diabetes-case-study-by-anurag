#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as snb
import pandas as pd
import matplotlib.pyplot as plt


# # Read DATASET 

# In[2]:


df = pd.read_csv('dataset_diabetes_diabetic_data.csv')
df.head()


# # Dataset Analysis

# In[3]:


df.isnull().sum()


# In[4]:


for col in df.columns:
    print(col,":",df[col].nunique())


# In[5]:


df.drop('patient_nbr',axis='columns',inplace=True)
df.head()


# In[6]:


df[df['diag_1']=='?']['diag_1'].count()


# In[7]:



df[df['diag_2']=='?']['diag_2'].count()


# In[8]:


df[df['diag_3']=='?']['diag_3'].count()


# In[9]:


for i in df.columns:
    if (df[df[i]=='?'][i].count())>0:
        print(i,":",df[df[i]=='?'][i].count())


# In[10]:


count=0
for i,row in df.iterrows():
    if(row['race']=='?')&(row['diag_1']=='?')&(row['diag_2']=='?')&(row['diag_3']=='?'):
        count+=1
print(count)


# # Data Filtering

# In[13]:


df.drop(['diag_1','diag_2','diag_3'],axis='columns',inplace=True)
df.head()


# In[14]:


df.shape[0]


# In[15]:


df.drop_duplicates(keep = "first",inplace = True)


# In[16]:



df.shape[0]


# # Seperating Numerical data and Catagoorical data

# In[17]:


df_cat = pd.DataFrame()
df_num = pd.DataFrame()


# In[18]:


for i in df.columns:
    if df[i].dtype == object:
        df_cat[i] = df[i]
    else:
        df_num[i] = df[i]


# In[19]:


df_cat.head()


# In[20]:


df_num.head()


# # Removing Null Value

# In[21]:


from sklearn.preprocessing import LabelEncoder
lr = LabelEncoder()


# In[22]:


for i in df_cat.columns:
    df[i] = lr.fit_transform(df[i])
df.head()


# # Training and Testing

# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


inputs = df.drop('diabatic',axis='columns')
target = df.diabatic


# In[26]:


xtrain,xtest,ytrain,ytest = train_test_split(inputs,target,test_size=0.3,random_state=10)


# # Decision Tree

# In[27]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# In[28]:


model.fit(xtrain,ytrain)


# In[40]:


ypred = grid_search_cv.predict(xtest)


# In[41]:


model.score(xtest,ytest)


# # GridSearch Cross-Validation using decision tree 

# In[42]:


import numpy as np


# In[43]:


parameter = { 'criterion':['gini','entropy'],'max_depth': np.arange(3, 10)}


# In[44]:


from sklearn.model_selection import GridSearchCV
grid_search_cv = GridSearchCV(model, parameter,cv=3)


# In[45]:


grid_search_cv.fit(xtrain,ytrain)


# In[46]:


grid_search_cv.best_params_


# In[47]:


grid_search_cv.score(xtest,ytest)


# In[48]:


score = grid_search_cv.cv_results_
score['mean_test_score']


# In[49]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(ytest,ypred)
snb.heatmap(cm,annot=True)


# # Random Forest

# In[50]:


from sklearn.ensemble import RandomForestClassifier
rfs = RandomForestClassifier(max_depth=2)


# In[51]:


rfs.fit(xtrain,ytrain)


# In[52]:


ypred = rfs.predict(xtest)


# In[53]:


rfs.score(xtest,ytest)


# In[54]:


accuracy_score(ypred,ytest)


# In[55]:


cm = confusion_matrix(ytest,ypred)


# In[57]:


import seaborn as sns


# In[58]:


sns.heatmap(cm,annot=True)


# # AUC and ROC Curve
# 

# # Auc-Roc Curve in Decision Tree

# In[59]:


probs = grid_search_cv.predict_proba(xtest)
probs = probs[:,1]
probs


# In[60]:


from sklearn.metrics import roc_curve,roc_auc_score


# In[61]:


ind_prob = [0 for i in range(len(ytest))]
ns_auc = roc_auc_score(ytest, ind_prob)
lr_auc = roc_auc_score(ytest, probs)


# In[62]:


print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Decision Tree: ROC AUC=%.3f' % (lr_auc))


# In[63]:


ns_fpr, ns_tpr, _ = roc_curve(ytest, ind_prob)
lr_fpr, lr_tpr, _ = roc_curve(ytest, probs)


# In[64]:


plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Decision tree')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# # Auc-Roc Curve in Random Forest

# In[65]:


probs = rfs.predict_proba(xtest)
probs = probs[:,1]
probs


# In[66]:


ind_prob = [0 for i in range(len(ytest))]
ns_auc = roc_auc_score(ytest, ind_prob)
lr_auc = roc_auc_score(ytest, probs)


# In[67]:


print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Random Forest: ROC AUC=%.3f' % (lr_auc))


# In[68]:


ns_fpr, ns_tpr, _ = roc_curve(ytest, ind_prob)
lr_fpr, lr_tpr, _ = roc_curve(ytest, probs)


# In[69]:


plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Ramdom Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# In[ ]:




