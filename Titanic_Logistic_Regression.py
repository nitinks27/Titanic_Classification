#!/usr/bin/env python
# coding: utf-8

# #### Overview about the data:
# 
# The data has been split into two groups:
# 
# - training set (train.csv)
# - test set (test.csv)
# 
# The **training set** should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use feature engineering to create new features.
# 
# The **test set** should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.
# 
# **Columns:**
# 
# - survival	 Survival	 0 = No, 1 = Yes
# - pclass	 Ticket class	 1 = 1st, 2 = 2nd, 3 = 3rd
# - sex	 Sex	
# - Age	 Age in years	
# - sibsp	 # of siblings / spouses aboard the Titanic	
# - parch	 # of parents / children aboard the Titanic	
# - ticket	 Ticket number	
# - fare	 Passenger fare	
# - cabin	 Cabin number	
# - embarked	 Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton
# 
# **Variable notes:**
# 
# - pclass: A proxy for socio-economic status (SES)
# - 1st = Upper
# - 2nd = Middle
# - 3rd = Lower
# 
# - age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# - sibsp: The dataset defines family relations in this way...
# - Sibling = brother, sister, stepbrother, stepsister
# - Spouse = husband, wife (mistresses and fiancés were ignored)
# 
# - parch: The dataset defines family relations in this way...
# - Parent = mother, father
# - Child = daughter, son, stepdaughter, stepson
# - Some children travelled only with a nanny, therefore parch=0 for them.

# `Task: Task is to predict survival of the Passengers in the test data.`

# **Logistic Regression**: Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable, although many more complex extensions exist. In regression analysis, logistic regression (or logit regression) is estimating the parameters of a logistic model (a form of binary regression).

# ![image.png](attachment:image.png)

# ###### Importing and preview of data

# In[2]:


#importing basic modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# %matplotlib inline: This is known as magic inline function.
#When using the 'inline' backend, our matplotlib graphs will be included in our notebook, next to the code. 


# Let's dive into our dataset.

# In[3]:


train_data = pd.read_csv("train.csv") #loading train data
test_data = pd.read_csv("test.csv") #loading test data


# In[4]:


train_df = train_data.copy() #copy of train_data as train_df
train_df.head() #train_df preview


# In[5]:


test_df = test_data.copy()#copy of train_data as train_df
test_df.head()


# As you can also notice that Suvived column is missing from the test data. As, this is out outcome variable. Otherwise I would have merged these two csv. Well, no issues we'll first perfrom all the operations with train_df and similar changes to test_df.
# 
# So, we'll train train our data with train_df and then predict the test data. 

# ##### Understanding our data

# In[6]:


train_df.shape


# In[7]:


test_df.shape


# In[8]:


train_df.isnull().sum() #isnull is used for checking missing values.


# In[9]:


test_df.isnull().sum()


# **Analysis**: 
# - train_df has three columns which is having missing values: age, cabin and embarked
# - test_df has two columns which is having missing values: age and cabin and Fare
# 
# We can use many funtions like fillna (backward or forward) or replacing it by using central tendency. If we notice carefully, for train_df Age column has 177 missing values out of 891 entries which is ~19-20% of the overall data and test_df has 86 out of 418 which is ~20-21% of the data, which can be handled but the issue is with Cabin column. In cabin column of both test_df and train_df ~77-78% of the data is missing which is quite huge. So, we will ignore them and there is missing value in Embarked column only for train_df which is just 2 entries, ~0.2-0.3% of the data. So, can easily be treated. So we left with Fare of test_df only 1 entry has null value.
# 
# Let's first replace Fare null value with mean.

# In[10]:


test_df.Fare.fillna(test_df["Fare"].mean(skipna=True), inplace=True)


# I'm going with the central tendency capping. But before choosing between mean median or mode let's check for the distribution of Age column.

# In[11]:


plt.figure(figsize=(15,6))
ax = sns.distplot(train_df.Age)
ax.set_title('Age', fontsize=14)


# **Understanding**: Age raging between 25-35 has the maximun number of people travelled on the Titanic. Let's check mean and median of the entries without missing values.

# In[12]:


train_df.Age.mean(skipna = True)


# In[13]:


train_df.Age.median(skipna = True)


# I'll go with median as it is by default in round figure so no need to work on it.

# In[14]:


train_df.Age.fillna(train_df["Age"].median(skipna=True), inplace=True)


# In[15]:


train_df['Embarked'].value_counts()


# So, most of the people have Port of Embarkation is S which stands for Southampton. Let's quickly fill the na with S.

# In[16]:


train_df.Embarked.fillna(train_df['Embarked'].value_counts().idxmax(), inplace=True)


# As Cabin has a lot of missing values we have ignored it and here we're done with missing values in train_df. Similar step in test_df for Age column.

# In[17]:


test_df.Age.fillna(test_df["Age"].median(skipna=True), inplace=True)


# Let's drop cabin column from both train_df and test_df

# In[18]:


train_df.drop(["Cabin"], axis=1, inplace=True)
train_df.head()


# In[19]:


test_df.drop(['Cabin'], axis=1, inplace=True)
test_df.head()


# Dealing with cateorical Terms.

# In[20]:


train_df['PassengerId'].nunique()


# In[21]:


train_df['Name'].nunique()


# In[22]:


train_df['Ticket'].nunique()


# In[23]:


train_df['Pclass'].nunique()


# In[24]:


train_df['SibSp'].nunique()


# In[25]:


train_df['Parch'].nunique()


# In[26]:


train_df['Fare'].nunique()


# **Analysis**: Now, PassengerId and name of every entries are unique, as well as ticket number except those who came with some person. So these features are not going to help us in further process. So, let's drop it.

# In[27]:


train_df.drop(['Name','PassengerId','Ticket'], axis=1, inplace=True)


# In[28]:


#Similarly with test_df
test_df.drop(['Name','PassengerId','Ticket'], axis=1, inplace=True)


# In[29]:


plt.figure(figsize=(15,6))
ax = sns.barplot(x=train_df['Pclass'], y=train_df['Fare'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_title('Pclass Vs Fare', fontsize=14)


# **Understanding**: So clearly, these are ordinal categories as 1st class passengers have paid more than 2nd trailing by 3rd class passengers.

# In[30]:


plt.figure(figsize=(15,6))
ax = sns.barplot(x=train_df['Pclass'], y=train_df['Survived'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_title('Pclass Vs Survived', fontsize=14)


# **Understanding**: Ah! so peoples who paid more had more secured life than those who paid less. So, it's clear that these are orinal categories so it's not needed to encode it using One Hot Encoding.

# In[31]:


ax = sns.catplot(x="Sex", y="Survived", hue="Pclass", kind="bar", data=train_df)


# In[32]:


ax = sns.catplot(x="Sex", y="Age", hue="Survived", kind="violin", split=True, data=train_df)


# **Understanding**: 
# - 1. Females have survived more than Males.
# - 2. Superior Passenger class people survived in higher number than there preceeding class.
# - 3. Age has a little differnce but noticible point is childrens survived more than died.
# - 4. Death in male is higher than survival among age ranges ~20-35.
# 

# In[32]:


print(train_df.Age.max())
print(train_df.Age.min())

age_gap = []

for i in range(0,81,5):
    age_gap.append(i)
    
train_df['Age'].groupby(pd.cut(train_df['Age'], age_gap)).count()


# In[33]:


plt.figure(figsize=(18,10))
ax = sns.barplot(x='Age', y='Survived', data=train_df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right");


# **Understanding**: 
# - 1. 0-16 aged kids have survived more than the people belonging to group 17-48.
# - 2. Again, 48-80 aged pople have survived more.
# 
# So they preffered to save kids, olds and women first. Good work!! 
# 
# But, I'll divide the age between two groups below 17 & above 48 in 1 and 17-48 in 0.

# In[34]:


train_df['Age'] = np.where((train_df['Age'] > 17) & (train_df['Age'] < 49),0,1)


# In[35]:


#similar changes with test data
test_df['Age'] = np.where((test_df['Age'] > 17) & (test_df['Age'] < 49),0,1)


# In[36]:


plt.figure(figsize=(15,6))
ax = sns.barplot(x=train_df['Age'], y=train_df['Survived'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_title('Age Vs Survived', fontsize=14)


# As you can see 0 has less chace for survival incomparison to 1

# In[37]:


plt.figure(figsize=(15,6))
ax = sns.kdeplot(train_df["Fare"][train_df.Survived == 1],shade= True)
ax1 = sns.kdeplot(train_df["Fare"][train_df.Survived == 0], shade= True)
plt.legend(['Survived', 'Died'])
plt.xlim(-20,200);


# **Understanding**:
# 
# As we discussed above the more you pay the more your life is secured. Here, is the same thing like the feature Pclass.

# In[38]:


plt.figure(figsize=(15,6))
ax = sns.barplot(x=train_df['Embarked'], y=train_df['Survived'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_title('Embarked Vs Survived', fontsize=14)


# So. people from Cheobarg survived more than Queensland and Southampton. One of the reason I assume of S being low because majority of people embark at S or may be more males embark at S or may be Pclass plays any role.

# Now, we are left with columns like Travelling with Sibiling or Parents.

# In[39]:


train_df.SibSp.value_counts()


# In[40]:


train_df.Parch.value_counts()


# **Analysis**:
# 
# Both column have a similarity that if 0 means they all are travelling alone and if any other value, they are either with parents or sibilings or say they aren't travelling alone. So, let's make it little simple.
# 
# Pepople travelling Alone would be 0, or travelling with someone would be 1. Let's make these changes.

# In[41]:


# Combining result of SibSp and Parch and storing result in new column travel
train_df['Travel'] = train_df['SibSp'] + train_df['Parch']


# In[42]:


#Changing values according to our analysis.
train_df['Travel'] = np.where((train_df.Travel == 0 ), 0, 1)


# In[43]:


plt.figure(figsize=(15,6))
ax = sns.barplot(x=train_df['Travel'], y=train_df['Survived'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_title('Travel Vs Survived', fontsize=14)


# **Analysis**:
# 
# So, people who were travelling alone had less chance to surive in comparison with travelling with someone.

# Now, will drop SibSp and Parch because they are left with no use.

# In[44]:


train_df.drop(["SibSp",'Parch'], axis=1, inplace=True)
train_df.head()


# In[45]:


#Similar chnages to test data

test_df['Travel'] = test_df['SibSp'] + test_df['Parch']
test_df['Travel'] = np.where((test_df.Travel == 0 ), 0, 1)
test_df.drop(["SibSp",'Parch'], axis=1, inplace=True)
test_df.head()


# Now its time to convert Embarked caterogical using One Hot Encoding or get_dummines, and label encode Sex column.

# In[46]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

train_df['Sex'] = le.fit_transform(train_df['Sex'])
train_df.head()


# In[47]:


#Similarly with test data.
test_df['Sex'] = le.fit_transform(test_df['Sex'])
test_df.head()


# In[48]:


train_df = pd.get_dummies(train_df, columns=['Embarked'], prefix = ['Embarked'])
train_df.head()


# In[49]:


#Similarly with test data
test_df = pd.get_dummies(test_df, columns=['Embarked'], prefix = ['Embarked'])
test_df.head()


# These are the final Columns of test and train data.

# Checking for multicollinearity among the continuous columns using VIF methods.
# 
# **Multicollinearity**: Multicollinearity occurs when two or more independent variables are highly correlated with one another in a regression model.
# 
# **Why not Multicollinearity?**: Multicollinearity can be a problem in a regression model because we would not be able to distinguish between the individual effects of the independent variables on the dependent variable.
# 
# **Detection of Multicollinearity**: Multicollinearity can be detected via various methods. One of the popular method is using VIF.
# 
# **VIF**: VIF stands for Variable Inflation Factors. VIF determines the strength of the correlation between the independent variables. It is predicted by taking a variable and regressing it against every other variable.

# In[50]:


X1 = train_df.drop(['Embarked_C','Embarked_Q', 'Embarked_S'],axis=1)


# In[51]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

X_vif = add_constant(X1)

pd.Series([variance_inflation_factor(X_vif.values, i) 
               for i in range(X_vif.shape[1])], 
              index=X_vif.columns)


# **Analsis**:
# 
# None of the columns have high VIF. Hence, less multicolinearity. Great!!

# In[52]:


#Checking for correlation.

plt.figure(figsize=(15,6))
ax = sns.heatmap(train_df.corr(),annot = True)
ax.set_title('CORRELATION MATRIX', fontsize=14)


# In[53]:


X = train_df.drop(["Survived"], axis=1)
y = train_df['Survived']


# In[54]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

# instantiate the model (using the default parameters)
logreg = LogisticRegression(max_iter=1000)

# fit the model with data
logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_test)


# In[55]:


# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[56]:


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[57]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[58]:


y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[59]:


X1 = ['Pclass','Sex','Age','Fare','Travel','Embarked_C','Embarked_Q','Embarked_S']

test_df['Survived'] = logreg.predict(test_df[X1])
test_df['PassengerId'] = test_data['PassengerId']

Predicted_outcome=  test_df[['PassengerId','Survived']]

Predicted_outcome.to_csv("Predicted_outcome.csv", index=False)

Predicted_outcome.head()


# In[ ]:




