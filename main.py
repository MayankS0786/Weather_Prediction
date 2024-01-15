from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import pandas as pd


data = pd.read_csv("seattle-weather.csv")

#print(data.head())
#print(data.info())

print(data.describe())

countrain = len(data[data.weather == 'rain'])

countsun = len(data[data.weather == 'sun'])

countdrizzle = len(data[data.weather == 'drizzle'])

countsnow = len(data[data.weather == 'snow'])

countfog = len(data[data.weather == 'fog'])


print('percent of rain:{:2f}%'.format((countrain/(len(data.weather))*100)))

print('percent of sun:{:2f}%'.format((countsun/(len(data.weather))*100)))

print('percent of drizzle:{:2f}%'.format((countdrizzle/(len(data.weather))*100)))

print('percent of snow:{:2f}%'.format((countsnow/(len(data.weather))*100)))

print('percent of fog:{:2f}%'.format((countfog/(len(data.weather))*100)))


print(data[['precipitation', 'temp_max', 'temp_min', 'wind']].describe())


print(data.isna().sum())

data.drop('date',axis=1,inplace=True)

#Identifying outliers
# Specify the numeric columns for outlier detection
numeric_columns = ['precipitation', 'temp_max', 'temp_min', 'wind']  # Add the names of your numeric columns

# Calculate the IQR for each numeric column
Q1 = data[numeric_columns].quantile(0.25)
Q3 = data[numeric_columns].quantile(0.75)
IQR = Q3 - Q1

# Identify outliers using the IQR method
outliers = ((data[numeric_columns] < (Q1 - 1.5 * IQR)) | (data[numeric_columns] > (Q3 + 1.5 * IQR))).any(axis=1)


print("Count of outliers:")
print(outliers.sum())


print("Rows with outliers:")
print(data[outliers])

# Remove outliers from the dataset
data_no_outliers = data[~outliers]


print("Dataset without outliers:")
print(data_no_outliers)


import numpy as np
data.precipitation = np.sqrt(data.precipitation)
data.wind=np.sqrt(data.wind)

data.head()

lc = LabelEncoder()
data['weather'] = lc.fit_transform(data['weather'])



x = ((data.loc[:,data.columns!='weather']).astype(int)).values[:,0:]
y = data['weather'].values

print(data.weather.unique())

#spliting
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.1, random_state=2)

#KNN
knn=KNeighborsClassifier()
knn.fit(x_train, y_train)
print('KNN accuracy:{:.2f}%'.format(knn.score(x_test,y_test)*100))

#xgboost classifier
xgb = XGBClassifier()
xgb.fit(x_train, y_train)
print('XGB accuracy:{:.2f}%'.format(xgb.score(x_test, y_test) * 100))

# Decision Tree classifier
from sklearn.tree import DecisionTreeClassifier

dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(x_train, y_train)
accuracy = dt_classifier.score(x_test, y_test)
print('Decision Tree accuracy: {:.2f}%'.format(accuracy * 100))

import pickle
file = 'model.pkl'
pickle.dump(xgb, open(file, 'wb'))


input = [[1.140175, 8.9, 2.8, 2.469818]]

result = xgb.predict(input)

print('the weather is:')
if(result == 0):
  print('Drizzle')
elif (result == 1):
  print('fogg')
elif (result == 2):
  print('rain')
elif (result == 3):
  print('snow')
else:
  print('sun')


