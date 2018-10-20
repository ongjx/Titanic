
# coding: utf-8

# # Importing libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
import seaborn as sns
# get_ipython().run_line_magic('matplotlib', 'inline')


# # Reading csv data and displaying first 5 rows

# In[2]:


df = pd.read_csv('train.csv')
df.head()


# In[3]:


print(df.head())
print(df.describe())


# ## Separation of dataset into Male and Female

# In[4]:


#change to numeric value
label, unique = pd.factorize(df.Sex)
df['Sex'] = label
male = df[df['Sex'] == 0]
female = df[df['Sex']== 1]

print(df.head())
df.Sex.value_counts().plot(kind='bar',rot=0)


# ## Survival Rate for Male and Female

# In[5]:


total_male_survive = male['Survived'].value_counts()[1]
total_male_died = male['Survived'].value_counts()[0]
total_male = male['Sex'].value_counts()
# print(total_male,total_male_survive,total_male_died)
percentage_male_survived = total_male_survive/total_male

total_female_survive = female['Survived'].value_counts()[1]
total_female_died = female['Survived'].value_counts()[0]
total_female = female['Sex'].value_counts()
percentage_female_survived = total_female_survive/total_female

print('Percentage of females survived: ', percentage_female_survived[1])
print('Percentage of males survived: ', percentage_male_survived[0])

print(df[['Sex','Survived']].groupby('Sex',as_index=False).mean())


# ## Separating by class

# In[6]:


total_class3 = df[df['Pclass'] == 3]['Pclass'].value_counts()
total_class2 = df[df['Pclass'] == 2]['Pclass'].value_counts()
total_class1 = df[df['Pclass'] == 1]['Pclass'].value_counts()
# print(total_class1[1],total_class2[2],total_class3[3])
print('Total Class1:',total_class1[1])
print('Total Class2:',total_class2[2])
print('Total Class3:',total_class3[3])


# What is the proportion of survivors in the different classes?

# In[7]:


class3 = df[df['Pclass']==3]
class3_survive = class3[class3['Survived'] == 1]['Survived'].value_counts()
class2 = df[df['Pclass']==2]
class2_survive = class2[class2['Survived'] == 1]['Survived'].value_counts()
class1 = df[df['Pclass']==1]
class1_survive = class1[class1['Survived'] == 1]['Survived'].value_counts()

print('Proportion of class 3 surviving:',class3_survive[1]/total_class3[3])
print('Proportion of class 2 surviving:',class2_survive[1]/total_class2[2])
print('Proportion of class 1 surviving:',class1_survive[1]/total_class1[1])


df[['Pclass','Survived']].groupby('Pclass', as_index=False).mean()


# ## Separating by survival/death

# In[8]:


male_survived = male[male['Survived'] == 1]
male_died = male[male['Survived']==0]
female_survived = female[female['Survived'] == 1]
female_died = female[female['Survived']==0]


# ### Male survival rates by class

# In[22]:


class3_male = male[male['Pclass']==3]['Pclass'].value_counts()[3]
class2_male = male[male['Pclass']==2]['Pclass'].value_counts()[2]
class1_male = male[male['Pclass']==1]['Pclass'].value_counts()[1]

male_survived_byclass = pd.DataFrame(male_survived['Pclass'].value_counts()).reset_index().sort_values(by='index')
male_survived_byclass.columns = ['class','survived']

class3_male_survive = male_survived_byclass[male_survived_byclass['class']==3]
class3_male_survival_rate = class3_male_survive['survived']/class3_male

class2_male_survive = male_survived_byclass[male_survived_byclass['class']==2]
class2_male_survival_rate = class2_male_survive['survived']/class2_male

class1_male_survive = male_survived_byclass[male_survived_byclass['class']==1]
class1_male_survival_rate = class1_male_survive['survived']/class1_male
print("Class1 male survival rate: ",class1_male_survival_rate[1])
print("Class2 male survival rate: ",class2_male_survival_rate[2])
print("Class3 male survival rate: ",class3_male_survival_rate[0])

male_survived_byclass['rates'] = [class1_male_survival_rate[1],class2_male_survival_rate[2],class3_male_survival_rate[0]]


male_died_byclass = pd.DataFrame(male_died['Pclass'].value_counts()).reset_index().sort_values(by='index')
male_died_byclass.columns = ['class','died']

class3_male_died = male_died_byclass[male_died_byclass['class']==3]
class3_male_died_rate = class3_male_died['died']/class3_male

class2_male_died = male_died_byclass[male_died_byclass['class']==2]
class2_male_died_rate = class2_male_died['died']/class2_male

class1_male_died= male_died_byclass[male_died_byclass['class']==1]
class1_male_died_rate = class1_male_died['died']/class1_male
print("Class1 male death rate: ",class1_male_died_rate[2])
print("Class2 male death rate: ",class2_male_died_rate[1])
print("Class3 male death rate: ",class3_male_died_rate[0])

male_died_byclass['rates'] = [class1_male_died_rate[2],class2_male_died_rate[1],class3_male_died_rate[0]]


male_summary_df = pd.DataFrame()
male_summary_df['class'] = [1,2,3]
male_summary_df['survival_rate'] = [class1_male_survival_rate[1],class2_male_survival_rate[2],class3_male_survival_rate[0]]
male_summary_df['death_rate'] = [class1_male_died_rate[2],class2_male_died_rate[1],class3_male_died_rate[0]]
male_summary_df['total'] = male_summary_df['survival_rate']+male_summary_df['death_rate']
print(male_summary_df)


# ## Female survival rates by class

# In[19]:


class3_female = female[female['Pclass']==3]['Pclass'].value_counts()[3]
print('Number of class3 females:',class3_female)
class2_female = female[female['Pclass']==2]['Pclass'].value_counts()[2]
print('Number of class2 females:',class2_female)
class1_female = female[female['Pclass']==1]['Pclass'].value_counts()[1]
print('Number of class1 females:',class1_female)

female_survived_byclass = pd.DataFrame(female_survived['Pclass'].value_counts()).reset_index().sort_values(by='index')
female_survived_byclass.columns = ['class','survived']

class3_female_survive = female_survived_byclass[female_survived_byclass['class']==3]
class3_female_survival_rate = class3_female_survive['survived']/class3_female

class2_female_survive = female_survived_byclass[female_survived_byclass['class']==2]
class2_female_survival_rate = class2_female_survive['survived']/class2_female

class1_female_survive = female_survived_byclass[female_survived_byclass['class']==1]
class1_female_survival_rate = class1_female_survive['survived']/class1_female


print("Class3 female survival rate: ",class3_female_survival_rate[1])
print("Class2 female survival rate: ",class2_female_survival_rate[2])
print("Class1 female survival rate: ",class1_female_survival_rate[0])
female_survived_byclass['rates'] = [class1_female_survival_rate[0],class2_female_survival_rate[2],class3_female_survival_rate[1]]


female_died_byclass = pd.DataFrame(female_died['Pclass'].value_counts()).reset_index().sort_values(by='index')
female_died_byclass.columns = ['class','died']

class3_female_died = female_died_byclass[female_died_byclass['class']==3]
class3_female_died_rate = female_died_byclass['died']/class3_female

class2_female_died = female_died_byclass[female_died_byclass['class']==2]
class2_female_died_rate = female_died_byclass['died']/class2_female

class1_female_died= female_died_byclass[female_died_byclass['class']==1]
class1_female_died_rate = female_died_byclass['died']/class1_female
print("Class3 female death rate: ",class3_female_died_rate[0])
print("Class2 female death rate: ",class2_female_died_rate[1])
print("Class1 female death rate: ",class1_female_died_rate[2])
female_died_byclass['rates'] = [class1_female_died_rate[2],class2_female_died_rate[1],class3_female_died_rate[0]]

female_summary_df = pd.DataFrame()
female_summary_df['class'] = [1,2,3]
female_summary_df['survival_rate'] = [class1_female_survival_rate[0],class2_female_survival_rate[2],class3_female_survival_rate[1]]
female_summary_df['death_rate'] = [class1_female_died_rate[2],class2_female_died_rate[1],class3_female_died_rate[0]]
female_summary_df['total'] = female_summary_df['survival_rate']+female_summary_df['death_rate']
print(female_summary_df)


# ## Visualising Results
# 
# How does the class affect the survival and death rates of Male and Females?

# In[23]:


fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(10,10))
male_survived_byclass.plot(kind = 'bar', x='class', y='rates', ax=axes[0,0],rot=0,title='male survivors',ylim=[0,1])
female_survived_byclass.plot(kind = 'bar', x='class', y='rates', ax=axes[1,0],rot=0, title='female survivors',ylim=[0,1])
male_died_byclass.plot(kind = 'bar', x='class', y='rates', ax=axes[0,1],rot=0,title='dead males',ylim=[0,1])
female_died_byclass.plot(kind = 'bar', x='class', y='rates', ax=axes[1,1],rot=0,title='dead females',ylim=[0,1])
plt.show()


# ### Deduction
# 
# It can be seen that more females survived than compared to males and when looking at the class level, passengers from the higher class were able to survive than compared to those from the lower class

# # Creating title column

# In[24]:


df['title'] = df['Name'].str.extract(' ([A-Za-z]+)\.',expand=False)
df['title'] = pd.factorize(df['title'])[0]
pd.factorize(df['title'])
df.loc[df['title']>2, 'title'] = 3


# ## Subdividing into Age group<br>
# <li>Noticed that some Age values are <code>Null</code>. Need to solve that.</li>
# 
# <li>By making use of the median age group of each name title to assign the age for NA values</li>
# 
# <li>Display the spread of all ages</li>

# In[25]:


df['Age'].fillna(df.groupby('title')['Age'].transform("median"),inplace=True)

facet = sns.FacetGrid(df, hue='Survived', aspect = 4)
facet.map(sns.kdeplot,'Age',shade = True)
facet.set(xlim=(0, df['Age'].max()))
facet.add_legend()
plt.show()
male['Age'].hist()
plt.show()


# ## Converting age into categories
# <br>
# <li>Children - 0</li>
# <li>Teen - 1</li>
# <li>Adult - 2</li>
# <li>Mid age - 3</li>
# <li>Elderly - 4</li>

# In[26]:


df.loc[df['Age'] <= 16, 'Age'] = 0
df.loc[(df['Age'] > 16) & (df['Age'] <= 21), 'Age'] = 1,
df.loc[(df['Age'] > 21) & (df['Age'] <= 40), 'Age'] = 2,
df.loc[(df['Age'] > 40) & (df['Age'] <= 60), 'Age'] = 3,
df.loc[df['Age'] > 60, 'Age'] = 4

df.head()


# ## Combining parent, children, sibling, spouse into 1

# In[27]:


df['family_size'] = (df['SibSp'] + df['Parch'] + 1)/5
df['family_size'].unique()
print(df.head(5))


facet = sns.FacetGrid(df, hue='Survived', aspect = 4)
facet.map(sns.kdeplot,'family_size',shade = True)
facet.set(xlim=(0, df['family_size'].max()))
facet.add_legend()
plt.show()


# ## Dropping SibSp & Parch

# In[28]:


df.drop(columns=['SibSp','Parch'],inplace=True)
df.head()


# ## Embarked
# 
# Checking the spread

# In[29]:


pclass1 = df[df['Pclass']==1]['Embarked'].value_counts()
pclass2 = df[df['Pclass']==2]['Embarked'].value_counts()
pclass3 = df[df['Pclass']==3]['Embarked'].value_counts()
embarked_df = pd.DataFrame([pclass1,pclass2,pclass3])
embarked_df.index = ['Class 1', 'Class 2', 'Class 3']
embarked_df.plot(kind='bar',stacked=True,rot=0)


# ## Converting to numeric category

# In[30]:


label,unique = pd.factorize(df.Embarked)
df["Embarked"] = pd.DataFrame(label)
df.head()


# ## Converting Fare to categories
# 
# <li>First, fill all the NA values </li>
# <li>Filter less than equal 7.91 = 0</li>
# <li>Filter less than equal 14.454 = 1</li>
# <li>Filter less than equal 31 = 2</li>
# <li>Filter more than equal 7.91 = 3</li>

# In[31]:


df['Fare'].fillna(df['Fare'].median(),inplace=True)
df.loc[df['Fare'] <= 7.91, 'Fare'] = 0
df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1,
df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare'] = 2,
df.loc[df['Fare'] > 31, 'Fare'] = 3

df.head()


# # Removing the columns that are deemed to be unimportant
# 
# I felt that the place at which the passengers embarked from, the fare/ticket isn't very important. Fare itself can be disregarded as the <code>Pclass</code> column provides more information hence dropping <code>Cabin,Ticket,Name</code>

# In[32]:


df.drop(columns=['Ticket','Name','Cabin'],inplace=True)


# # Final dataset

# In[33]:


df.head(10)


# # Machine Learning portion

# In[34]:


import numpy as np
label = df['Survived']
features = df.drop(columns=['PassengerId','Survived'])


# ### Cross Validation (K-fold)

# In[35]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
kfold = KFold(n_splits=10,shuffle=True,random_state=0)


# ### Random Forest Regressor

# In[36]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestRegressor(n_estimators=13,random_state=42)
clf = RandomForestClassifier(n_estimators=13)
# label = df['Survived']
# features = df.drop(columns=['PassengerId','Survived'])
xtrain, xtest, ytrain, ytest = train_test_split(features,label,test_size=0.20)
rf.fit(features,label)
# print(rf.score(xtest,ytest))
predictions = rf.predict(xtest)
x = pd.DataFrame([predictions,ytest]).T
x.columns = ['predict','True']
rf.score(xtest,ytest)



# ### Random Forest Classifier

# In[37]:


scoring = 'accuracy'
score = cross_val_score(clf,features,label,cv=kfold,n_jobs=1,scoring=scoring)
print(round(np.mean(score)*100,2))


# ### LinearSVC

# In[38]:


from sklearn import svm
clf = svm.LinearSVC(random_state=0,tol=1e-5)
clf.fit(xtrain,ytrain)
clf.score(xtest,ytest)


# ### SVM

# In[39]:


from sklearn.svm import SVC
clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf,features,label,cv=kfold,n_jobs=1,scoring=scoring)
print(score)
print(round(np.mean(score)*100,2))


# ### KNearestNeighbor

# In[40]:


from sklearn.neighbors import KNeighborsClassifier
# print(df.info())
clf = KNeighborsClassifier(n_neighbors=14)
scoring = 'accuracy'
score = cross_val_score(clf,features,label,cv=kfold,n_jobs=1,scoring=scoring)
print(score)
print(round(np.mean(score)*100,2))


# ### Decision Tree

# In[41]:


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(criterion='entropy')
scoring = 'accuracy'
score = cross_val_score(clf,features,label,cv=kfold,n_jobs=1,scoring=scoring)
print(score)
print(round(np.mean(score)*100,2))


# # Testing

# ## Cleaning up test file

# In[42]:


test = pd.read_csv('test.csv')
test['title'] = test['Name'].str.extract(' ([A-Za-z]+)\.',expand=False)
test['title'] = pd.factorize(test['title'])[0]
pd.factorize(test['title'])
test.loc[test['title']>2, 'title'] = 3
test['Age'].fillna(test.groupby('title')['Age'].transform("median"),inplace=True)
test.loc[test['Age'] <= 16, 'Age'] = 0
test.loc[(test['Age'] > 16) & (test['Age'] <= 21), 'Age'] = 1,
test.loc[(test['Age'] > 21) & (test['Age'] <= 40), 'Age'] = 2,
test.loc[(test['Age'] > 40) & (test['Age'] <= 60), 'Age'] = 3,
test.loc[test['Age'] > 60, 'Age'] = 4
test['family_size'] = (test['SibSp'] + test['Parch'] + 1)/5
test['family_size'].unique()
test.drop(columns=['SibSp','Parch'],inplace=True)
l,unique = pd.factorize(test.Embarked)
test["Embarked"] = pd.DataFrame(l)
test['Fare'].fillna(test['Fare'].median(),inplace=True)
test.loc[test['Fare'] <= 7.91, 'Fare'] = 0
test.loc[(test['Fare'] > 7.91) & (test['Fare'] <= 14.454), 'Fare'] = 1,
test.loc[(test['Fare'] > 14.454) & (test['Fare'] <= 31), 'Fare'] = 2,
test.loc[test['Fare'] > 31, 'Fare'] = 3
l, unique = pd.factorize(test.Sex)
test['Sex'] = l
male = test[test['Sex'] == 0]
female = test[test['Sex']== 1]
test['Age'].fillna(test.groupby('title')['Age'].transform("median"),inplace=True)
test.head()


# ### Creating test features and doing the predictions

# In[43]:


test_features = test.drop(columns=['Name','Ticket','Cabin','PassengerId']).copy()


# In[44]:


clf = SVC()
clf.fit(features,label)
prediction = clf.predict(test_features)


# In[45]:


submission = pd.DataFrame({
    "PassengerId": test['PassengerId'],
    "Survived": prediction})

submission.to_csv('submission.csv',index=False)


# In[46]:


sub = pd.read_csv('submission.csv')
submission.head()


# In[47]:


rclf = RandomForestClassifier(n_estimators=100)
rclf.fit(features,label)
predict = rclf.predict(test_features)
submission = pd.DataFrame({
    "PassengerId": test['PassengerId'],
    "Survived": prediction})

submission.to_csv('submission1.csv',index=False)


# In[48]:


sub = pd.read_csv('submission1.csv')
submission.head()
rclf.score(features,label)

