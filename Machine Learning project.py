#!/usr/bin/env python
# coding: utf-8

# In[3]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


get_ipython().system('wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv')


# In[5]:


df = pd.read_csv('loan_train.csv')
df.head()


# In[6]:


df.shape


# In[7]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# In[8]:


df['loan_status'].value_counts()


# In[9]:



get_ipython().system('conda install -c anaconda seaborn -y')


# In[9]:


import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[10]:


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[11]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# In[12]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# In[13]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# In[14]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# In[15]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# In[16]:


df[['Principal','terms','age','Gender','education']].head()


# In[17]:


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# In[18]:


X = Feature
X[0:5]


# In[19]:


y = df['loan_status'].values
y[0:5]


# In[20]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# In[21]:


#testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[22]:


#KNN
from sklearn.neighbors import KNeighborsClassifier
k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh


# In[23]:


yhat = neigh.predict(X_test)
X[0:5]


# In[24]:


#KNN accuracy
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# In[25]:


#best K
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc


# In[26]:


#plot best K
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()


# In[27]:


#which K is the best?
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 


# In[28]:


#f1 score for KNN
from sklearn.metrics import f1_score
f1_score(y_test, yhat, average='weighted') 


# In[29]:


#jaccard for KNN
from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat)


# In[30]:


#decisiontree
from sklearn.tree import DecisionTreeClassifier
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
Tree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
Tree 


# In[31]:


Tree.fit(X_trainset,y_trainset)


# In[32]:


predTree = Tree.predict(X_testset)


# In[33]:


print (predTree [0:5])
print (y_testset [0:5])


# In[34]:


#decisiontree accuracy
from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))


# In[30]:


#f1 score for decisiontree
from sklearn.metrics import f1_score
f1_score(y_test, yhat, average='weighted') 


# In[31]:


#jaccard for decisiontree
from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat)


# In[36]:


#SVM
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[37]:


from sklearn import svm
df = svm.SVC(kernel='rbf')
df.fit(X_train, y_train) 


# In[38]:


yhat = df.predict(X_test)
yhat [0:5]


# In[39]:


from sklearn.metrics import classification_report, confusion_matrix
import itertools


# In[40]:


#f1 score for SVM
from sklearn.metrics import f1_score
f1_score(y_test, yhat, average='weighted') 


# In[41]:


#jaccard for SVM
from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat)


# In[43]:


#logisticregression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[44]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR


# In[45]:


yhat = LR.predict(X_test)
yhat


# In[46]:


yhat_prob = LR.predict_proba(X_test)
yhat_prob


# In[47]:


#jaccard for LR
from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat)


# In[32]:


#f1 for LR
from sklearn.metrics import f1_score
f1_score(y_test, yhat, average='weighted') 


# In[49]:


#logloss for LR
from sklearn.metrics import log_loss
log_loss(y_test, yhat_prob)


# In[ ]:


#thank you for reading

