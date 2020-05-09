#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Load libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.impute import MissingIndicator
import seaborn as sns
from random import randrange, uniform
from sklearn.impute import SimpleImputer
from scipy.stats import chi2_contingency
from fancyimpute import KNN


# In[2]:


import os
os.chdir("E:/Study Material/Data Science/Learning Data Science edWisor/Projects/Bank Loan Default Case/Python code")


# In[3]:


data=pd.read_csv("bank-loan.csv", sep=',')
data_test=data.iloc[700:849
                    ,:]
data=data.iloc[:700,:]


# In[6]:


data.dtypes


# In[7]:


data['employ']=data['employ'].astype('category')
data['address']=data['address'].astype('category')
data['ed']=data['ed'].astype('category')
data['default']=data['default'].astype('category')

#data['employ']=data['employ'].astype('object')
#data['address']=data['address'].astype('object')
#data['ed']=data['ed'].astype('object')
#data['default']=data['default'].astype('object')


# # Missing Value Analysis

# In[8]:


data.head(60)


# In[9]:


data=data.replace({'ed':0,'employ':0,'address':0,'income':0,'debtinc':0,'creddebt':0,'othdebt':0},np.nan)


# In[10]:


data.dtypes


# In[11]:


data.head(60)


# In[12]:


#Create dataframe with missing percentage
missing_val = pd.DataFrame(data.isnull().sum())

#Reset index
missing_val = missing_val.reset_index()

#Rename variable
missing_val = missing_val.rename(columns = {'index': 'Variables', 0: 'Missing_percentage'})

#Calculate percentage
missing_val['Missing_percentage'] = (missing_val['Missing_percentage']/len(data))*100

#descending order
missing_val = missing_val.sort_values('Missing_percentage', ascending = False).reset_index(drop = True)


# In[13]:


data.dtypes 


# In[14]:


data.ed


# In[15]:


#Creating missing value 
#Actual value of Employ at observation 16 = 13
data['employ'].loc[41]=np.nan


# In[227]:


data.head(60)
#Impute with mode
#imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
#data=pd.DataFrame(imp.fit_transform(data),columns=data.columns,index=data.index)


# In[16]:


data.dtypes


# In[17]:


#Impute with KNN

data= pd.DataFrame(KNN(k = 5).fit_transform(data), columns = data.columns)



# In[18]:


data.head(60)


# In[19]:


catenames= ["ed","address","employ","default"]
for i in catenames:
    data.loc[:,i] = data.loc[:,i].round()
    data.loc[:,i] = data.loc[:,i].astype('category')
    


# In[232]:


data.head(60)


# In[20]:


data.head(60)


# In[21]:


data.dtypes


# In[22]:


#Converting into correct Datatypes
data['age']=data['age'].astype('int64')
#data['ed']=data['ed'].astype('category')
#data['employ']=data['employ'].astype('category')
#data['address']=data['address'].astype('category')
data['income']=data['income'].astype('int64')
data['debtinc']=data['debtinc'].astype('float64')
data['creddebt']=data['creddebt'].astype('float64')
data['othdebt']=data['othdebt'].astype('float64')
#data['default']=data['default'].astype('category')


# In[23]:


data.dtypes


# # Outlier Analysis

# In[24]:


df=data.copy()
#data=df.copy()


# In[25]:


# #Plot boxplot to visualize Outliers
get_ipython().run_line_magic('matplotlib', 'inline')
plt.boxplot(data['address'])


# In[26]:


#save All Numeric Variables  for Outlier analysis
cnames= ["age", "income", "debtinc", "creddebt", "othdebt"]


# In[27]:


# #Detect and delete outliers from data
for i in cnames:
    print(i)
    q75, q25 = np.percentile(data.loc[:,i], [75 ,25])
    iqr = q75 - q25

    min = q25 - (iqr*1.5)
    max = q75 + (iqr*1.5)
    print(min)
    print(max)
     
    data = data.drop(data[data.loc[:,i] < min].index)
    data = data.drop(data[data.loc[:,i] > max].index)


# In[21]:


#Detect and replace with NA
#for i in cnames:
 #   print(i)
  #  q75, q25 = np.percentile(data.loc[:,i], [75 ,25])
   # iqr = q75 - q25

    #min = q25 - (iqr*1.5)
 #   max = q75 + (iqr*1.5)
 #   print(min)
 #   print(max)

  #  marketing_train.loc[data[data.loc[:,i] < min,:'custAge'] = np.nan
  #  marketing_train.loc[data[data.loc[:,i] > max,:'custAge'] = np.nan

# #Calculate missing value
# missing_val = pd.DataFrame(data.isnull().sum())

# #Impute with KNN
# data = pd.DataFrame(KNN(k = 3).complete(data), columns = data.columns)


# In[28]:


data.shape


# # Feature Selection 

# In[29]:


##Correlation analysis
df_corr = data.loc[:,cnames]

#Set the width and hieght of the plot
f, ax = plt.subplots(figsize=(7, 5))

#Generate correlation matrix
corr = df_corr.corr()

#Plot using seaborn library
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# In[30]:


#Chisquare test of independence
#Save categorical variables
cat_names = ["ed", "employ","address"]

#loop for chi square values
for i in cat_names:
    print(i)
    chi2, p, dof, ex = chi2_contingency(pd.crosstab(data['default'], data[i]))
    print(p)


# In[31]:


data = data.drop(['ed'], axis=1)


# # Feature Scaling

# In[32]:


df = data.copy()
#data= df.copy()


# In[33]:


#Checking Normality of variables
get_ipython().run_line_magic('matplotlib', 'inline')
for i in cnames:
    print(i)
    plt.hist(data[i], bins='auto')


# In[34]:


#Nomalisation
for i in cnames:
    print(i)
    data[i] = (data[i] - (data[i]).min())/((data[i].max()) - (data[i].min()))


# In[35]:


data.head(10)


# # Model Deployment

# In[36]:


#Logistic Regression 
data_logit=pd.DataFrame(data['default'])


# In[37]:


data_logit=data_logit.join(data[cnames])


# In[38]:


data_logit.head()


# In[39]:


#Creating dummies for categorical variables
cate_names=["employ","address"]
for i in cate_names:
    temp = pd.get_dummies(data[i], prefix = i)
    data_logit = data_logit.join(temp)


# In[40]:


data_logit.shape


# In[41]:


Sample_Index = np.random.rand(len(data_logit)) < 0.8

train = data_logit[Sample_Index]
test = data_logit[~Sample_Index]


# In[42]:


#select column indexes for independent variables
data_cols = train.columns[1:72]


# In[43]:


train.shape


# In[44]:


data_cols


# In[46]:


train.dtypes


# In[254]:


#Building Logistic Regression
import statsmodels.api as sm

logit = sm.Logit(train['default'], train[data_cols]).fit()


# # KNN Implementation

# In[135]:



from sklearn.model_selection import train_test_split   # sklearn cross validation has been deprecated


# In[136]:



#Divide data into train and test
X = data.values[:, 0:7]
Y = data.values[:,7]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2)


# In[137]:


#X_train=X_train.astype('O')
y_train=y_train.astype(int)
#X_test=X_test.astype('O')
#y_test=y_test.astype('O')


# In[138]:


#KNN implementation
from sklearn.neighbors import KNeighborsClassifier

KNN_model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)


# In[94]:


#predict test cases
KNN_Predictions = KNN_model.predict(X_test)


# In[96]:


#build confusion matrix
CM = pd.crosstab(y_test, KNN_Predictions)

print(CM)

#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

print(TN)
print(FN)
print(TP)
print(FP)

#check accuracy of model
#accuracy_score(y_test, y_pred)*100
#((TP+TN)*100)/(TP+TN+FP+FN)

#False Negative rate 
(FN*100)/(FN+TP)


# In[104]:


#Accuracy:77.67%
#FNR: 72


# In[97]:


KNN_Predictions


# # Naive Bayes

# In[98]:


#Naive Bayes
from sklearn.naive_bayes import GaussianNB

#Naive Bayes implementation
NB_model = GaussianNB().fit(X_train, y_train)


# In[99]:


#predict test cases
NB_Predictions = NB_model.predict(X_test)


# In[104]:


#Build confusion matrix
CM = pd.crosstab(y_test, NB_Predictions)

print(CM)
#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

#check accuracy of model
#accuracy_score(y_test, y_pred)*100
#((TP+TN)*100)/(TP+TN+FP+FN)

#False Negative rate 
(FN*100)/(FN+TP)

#Accuracy: 70
#FNR: 48


# In[ ]:


#Decision Tree


# In[111]:


#Import Libraries for decision tree
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 


# In[112]:


#replace target categories with Yes or No
data['default'] = data['default'].replace(0, 'No')
data['default'] = data['default'].replace(1, 'Yes')


# In[113]:


#Divide data into train and test
X = data.values[:, 0:7]
Y = data.values[:,7]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2)


# In[122]:


#Decision Tree
C50_model = tree.DecisionTreeClassifier(criterion='entropy').fit(X_train, y_train)


# In[123]:


#predict new test cases
C50_Predictions = C50_model.predict(X_test)


# In[124]:


data.columns[0:7]


# In[125]:


X_train=pd.DataFrame(X_train[cnames])


# In[119]:


C50_model


# In[126]:


#Create dot file to visualise tree  #http://webgraphviz.com/
dotfile = open("pt1.dot", 'w')
df = tree.export_graphviz(C50_model, out_file=dotfile, feature_names = data.columns[0:7])


# In[127]:


#build confusion matrix
# from sklearn.metrics import confusion_matrix 
# CM = confusion_matrix(y_test, y_pred)
CM = pd.crosstab(y_test, C50_Predictions)

#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

#check accuracy of model
#accuracy_score(y_test, y_pred)*100
((TP+TN)*100)/(TP+TN+FP+FN)

#False Negative rate 
#(FN*100)/(FN+TP)

#Results
#Accuracy: 84.49
#FNR: 63


# In[ ]:


############Random Forest######


# In[128]:


from sklearn.ensemble import RandomForestClassifier

RF_model = RandomForestClassifier(n_estimators = 20).fit(X_train, y_train)


# In[129]:


RF_Predictions = RF_model.predict(X_test)


# In[134]:


# Confusion matrix
# from sklearn.metrics import confusion_matrix 
# CM = confusion_matrix(y_test, y_pred)
CM = pd.crosstab(y_test, RF_Predictions)
print(CM)
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

#check accuracy of model
#accuracy_score(y_test, y_pred)*100
((TP+TN)*100)/(TP+TN+FP+FN)

#False Negative rate 
#(FN*100)/(FN+TP)

#Accuracy: 75.89
#FNR: 64


# In[ ]:




