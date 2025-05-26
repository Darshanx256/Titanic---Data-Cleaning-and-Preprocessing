import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

tc = pd.read_csv("Titanic-Dataset.csv")     #importing dataset
print(tc.info())                            #loading info, most columns seems to be comleted(891)
print(tc.isnull().sum())                    #summing only true vales to make the data readable

#making boxplot to see which method will be the best
#age has 177 nulls, so doing for it first

sns.boxplot(x=tc['Age'])
plt.title('age boxplot')
plt.show()

#after checking the boxplot output, i did some research and asked chatGPT, learning
#that my plot is left skwed as i see the box and line within inclined to left and
#lots of dots (outliers) on the right, which means "extreme values" old passengers in
#this case, as there are mostly young passengers.

#using median would be appropriate for age, because it wont give crazy values due to few outliers.
#as per i know, median checks the most common age and uses it to fill gaps.
#i learnt that Pandas fillna fils null/missing values and the new value generated
#can be filed back into my tc database

tc['Age'].fillna(tc['Age'].median(), inplace=True)  #inplace=True replaces data onspot rather than providing a new copy

#Cabin got 687/891 nulls! and the values are too diverse
#can't do any assumption in this case, can fill unknown ones with 'U'

tc['Cabin'] = tc['Cabin'].fillna('U')

#this data dosent seem to be any useful as there is like 75% missing values
#I can make it clean by only getting the first word (Deck) and U means Unknown
#can totally make new column for it!

tc['Deck'] = tc['Cabin'].str[0]         #extracting first letter using str indexing

#lets make a graphical plot now! U means Unknown

sns.countplot(x='Deck', data=tc, order=sorted(tc['Deck'].unique()))
plt.title("Cabin Deck chart, U is unknown")
plt.show()

#while the 75% data is unknown for cabin, i can know that most of the known vales are from C deck,
#B deck is the second and D deck is the 3rd, least amount in T

#next missing value is embarked, lets check how many values are missing

print(tc['Embarked'].value_counts())

#i learnt that this column means the port from which people got into titanic
#with only 2 values missing, i can safely use 'S' for missing values as the database has most S
#not doing it directly, just using mode!

tc['Embarked'].fillna(tc['Embarked'].mode()[0], inplace=True)

#with this done, lets check if all values are filled

print(tc.isnull().sum())                    #ya! 0 nulls now, going to next step

#Convert categorical features into numerical using encoding.
#name can be kept, sex, cabin, embarked, deck columns needs to be converted!

tc['Sex'] = tc['Sex'].map({'male': 0, 'female':1})

#learnt that Map takes a dictionary and asigns key values in databse to dictionaty values
#this one is challenging.
#asked chatGPT, i can use One-Hot encoding and break into diffrernt columns taking 0/1 for T/F

tc = pd.get_dummies(tc, columns=['Embarked', 'Deck'], drop_first=True)

#can drop cabin, as Ive extracted Deck already and Cabin is useless

tc.drop('Cabin', axis=1 , inplace=True)     #axis=1 To operate on column

#next step is standardization/normaliztion
#learn from google that i can use Z-score method that ive learnt, along with Min-Max for normalization

num_cols = ['Age','Fare','SibSp','Parch']

for col in num_cols:
    tc[col] = (tc[col] - tc[col].mean()) / tc[col].std()  #Z-Score

for col in num_cols:
    tc[col] = (tc[col] - tc[col].min()) / (tc[col].max() - tc[col].min())

print(tc[num_cols].describe())

#Visualize boxplot and remove Outliers

for col in num_cols:
    sns.boxplot(x=tc[col])
    plt.title('Box plot of {col}')
    plt.show()

#Now using simple Quartile and Interquarite range method to remove Outliers.

for col in num_cols:
    Q1 = tc[col].quantile(0.25)     #25%
    Q3 = tc[col].quantile(0.75)     #75%
    IQR = Q3 - Q1                   #mid 50%

    lower = Q1 - 1.5 *IQR
    upper = Q3 + 1.5 *IQR

    tc = tc[(tc[col] >= lower) & (tc[col] <= upper)]

#saving processed copy now

tc.to_csv("Cleaned-Titanic.csv", index = False, float_format="%.4f")   #outputting with 4 digit precision








