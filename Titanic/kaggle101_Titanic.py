#kaggle101_Titanic:myself+reference
#Project site:https://www.kaggle.com/c/titanic
#data analysis package
#kernel:https://www.kaggle.com/startupsci/titanic-data-science-solutions
import pandas as pd 
import numpy as np 
import random as rnd 

#visualization package

import seaborn as sns
import matplotlib.pyplot as plt 
#%matplotlib inline

#ML package

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

#load data

train_df=pd.read_csv('train.csv')
test_df=pd.read_csv('test.csv')
combine=[train_df,test_df]#list type only support len(),not size()

print(train_df.columns.values)
train_df.head()

train_df['Embarked'].isnull().any()#查看Embarked是否存在nan

train_df.info()

train_df.describe()#查看数值类型变量统计信息

train_df.describe(include=['O'])#查看种类类型变量统计信息

#pivot data
train_df[['Pclass','Survived']]#如果需要取一列以上，需要传入list，也就是使用【】将需要取的列名包起来传入

train_df[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)

train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#visualize
g=sns.FacetGrid(train_df,col='Survived')
g.map(plt.hist,'Age',bins=20)


#grid=sns.FacetGrid(train_df,col='Survived',row='Pclass',size=2.2,aspect=1.6)
grid=sns.FacetGrid(train_df,col='Pclass',hue='Survived')
grid.map(plt.hist,'Age',alpha=0.5,bins=20)
grid.add_legend()


sns.pointplot(x="Sex", y="Survived", hue="Pclass", data=train_df)

grid=sns.FacetGrid(train_df,row='Embarked',size=2.2,aspect=1.6)#其中size定义的是宽，aspect定义的是长宽比，也就是说自动帮计算出相应的高
grid.map(sns.pointplot,'Pclass','Survived','Sex',palette='deep')#其中Pclass的位置表示x轴，Survived位置表示y轴，Sex表示填色部分
grid.add_legend()


grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)#ci参数表示设置竖线表示标准差
grid.add_legend()


#wrangle data

print("Before",train_df.shape,test_df.shape,combine[0].shape,combine[1].shape)
train_df=train_df.drop(['Ticket','Cabin'],axis=1)#看来在这里pandas的DateFrame的drop函数中axis=1表示列
test_df=test_df.drop(['Ticket','Cabin'],axis=1)#值得注意的是在DataFrame的选择中似乎都是使用列表来选择的
combine=[train_df,test_df]
print("After:",train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)



train_df['Name']
'Rice, Mrs. William (Margaret Norton)'.extract('([A-Za-z]+)\.',expand=False)#这样做不行，意味着str本身没有extract函数
for dataset in combine:
      print(type(dataset.Name.str))#out:<class 'pandas.core.strings.StringMethods'>
      dataset['Title']=dataset.Name.str.extract('([A-Za-z]+)\.',expand=False)#是在pandas的dataframe中才有这个extract函数
train_df['Title']#上面对combine进行循环中dataset进行处理，最后结果是train_df发生变化，因此可以认为是进行了一种类似指针的操作
pd.crosstab(train_df['Title'],train_df['Sex'])

for dataset in combine:#replace(x,y)将x换为y
	dataset['Title']=dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
 	dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
train_df[['Title','Survived']].groupby(['Title'],as_index=False).mean()#如果需要取出某一列或者某几列，则需要传入list

title_mapping={'Mr':1,'Miss':2,'Mrs':3,'MAster':4,'Rare':5}#将之前的title从文本类型转换为ordinal类型
for dataset in combine:
	dataset['Title']=dataset['Title'].map(title_mapping)#传入dict，使用map函数
	dataset['Title']=dataset['Title'].fillna(0)#0值填充NAN
train_df.head()

train_df=train_df.drop(['Name','PassengerId'],axis=1)
test_df=test_df.drop(['Name'],axis=1)
combine=[train_df,test_df]
train_df.shape
test_df.shape

sex_map={'female':1,'male':0}
for dataset in combine:
	dataset['Sex']=dataset['Sex'].map(sex_map).astype(int)
train_df.head()

#在进行guess填充值之前，可以先根据分组信息可视化一下
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

guess_ages=np.zeros((2,3))
guess_ages

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5#the code above is used to caculate result
            
    for i in range(0, 2):#the code below is used to fill nan
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()

#make age band
train_df['AgeBand']=pd.cut(train_df['Age'],5)#pd.cut(dataset,5)其中5位置的参数表示分为几组，也就是pandas自动根据dataset数值范围计算组的范围
train_df[['AgeBand','Survived']].groupby(['AgeBand'],as_index=False).mean().sort_values(by='AgeBand',ascending=True)

for dataset in combine:#看来之前的cut
	dataset.loc[dataset['Age']<=16,'Age']=0#此处的DataFrame loc函数传入的第一个参数选出的是行，第二个参数是在选出行的基础上选出列的数据
	dataset.loc[(dataset['Age']>16)&(dataset['Age']<=32),'Age']=1
	dataset.loc[(dataset['Age']>32)&(dataset['Age']<=48),'Age']=2
	dataset.loc[(dataset['Age']>48)&(dataset['Age']<=64),'Age']=3
	dataset.loc[dataset['Age']>64,'Age']=4#DataFrame的loc方法格式为data.loc[]
train_df.head()

#drop AgeBand
train_df=train_df.drop(['AgeBand'],axis=1)
combine=[train_df,test_df]
train_df.head()

for dataset in combine:#这里可以看出为什么需要将train和test组合进combine中，因为如果需要同时更新train和test中都有的变量时，直接对combine操作即可
	dataset['FamilySize']=dataset['SibSp']+dataset['Parch']+1
train_df[['FamilySize','Survived']].groupby(['FamilySize'],as_index=False).mean().sort_values(by='Survived',ascending=False)

for dataset in combine:
	dataset['IsAlone']=0 
	dataset.loc[dataset['FamilySize']==1,'IsAlone']=1
train_df[['IsAlone','Survived']].groupby(['IsAlone'],as_index=False).mean()


#如下这种使用for x in combine的方法进行drop最后不能传入指针指向的combine和train_df
#原因未知，但是如果是添加字段可以使用for x in combine，最后对x的操作会影响到combine和train_df
#for dataset in combine:
#      dataset=dataset.drop(['Parch','SibSp','FamilySize'],axis=1)
#      print(dataset.head())
#      print('-'*40)
#train_df=combine[0]
#train_df.head()

#所以删除还是要从train_df和test_df来删

train_df=train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df=test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine=[train_df,test_df]
train_df.head()

for dataset in combine:
	dataset['Age*Pclass']=dataset['Age']*dataset['Pclass']
train_df.loc[:,['Age*Pclass','Age','Pclass']].head(10)

freq_port=train_df.Embarked.dropna().mode()[0]#mode（）函数返回众数
for dataset in combine:
	dataset['Embarked']=dataset['Embarked'].fillna(freq_port)
train_df[['Embarked','Survived']].groupby(['Embarked',as_index=False]).mean().sort_values(by='Survived',ascending=False)

for dataset in combine:
	dataset['Embarked']=dataset['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)
train_df.head()

test_df['Fare'].fillna(test_df['Fare'].dropna().median(),inplace=True)
test_df.head()


train_df['FareBand']=pd.qcut(train_df['Fare'],4)
train_df[['FareBand','Survived']].groupby(['FareBand'],as_index=False).mean().sort_values(by='Survived',ascending=True)
#切分出FareBand的目的是之后根据Band的范围将Fare转换为数值类型来表示种类
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
train_df.head(10)


X_train=train_df.drop(['Survived','PassengerId'],axis=1)
Y_train=train_df['Survived']
X_test=test_df.drop('PassengerId',axis=1).copy()
X_train.shape,Y_train.shape,X_test.shape


#trainning model and make some predict

#logistic
logreg=LogisticRegression()
logreg.fit(X_train,Y_train)
Y_pred=logreg.predict(X_test)
acc_log=round(logreg.score(X_train,Y_train)*100,2)
acc_log

coeff_df=pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns=['Feature']
coeff_df['Correlation']=pd.Series(logreg.coef_[0])
coeff_df.sort_values(by='Correlation',ascending=False)


#SVM
svc=SVC()
svc.fit(X_train,Y_train)
Y_pred=svc.predict(X_test)
acc_svc=round(svc.score(X_train,Y_train)*100,2)#因为计算score需要有标注的lable，但test_df中没有答案，因此还是用train_df来计算score
acc_svc

#KNN
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,Y_train)
Y_pred=knn.predict(X_test)
acc_knn=round(knn.score(X_train,Y_train)*100,2)
acc_knn

#Gaussian NB
gaussian=GaussianNB()
gaussian.fit(X_train,Y_train)
Y_pred=gaussian.predict(X_test)
acc_gaussian=round(gaussian.score(X_train,Y_train)*100,2)
acc_gaussian

#Perceptron
perceptron=Perceptron()
perceptron.fit(X_train,Y_train)
Y_pred=perceptron.predict(X_test)
acc_perceptron=round(perceptron.score(X_train,Y_train)*100,2)
acc_perceptron

# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc

# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree

# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest

#evaluate models
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
# submission.to_csv('../output/submission.csv', index=False)