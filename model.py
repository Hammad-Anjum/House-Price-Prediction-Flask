import pandas as pd
import numpy as np
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error , r2_score , mean_absolute_error
from feature_engine.outliers import Winsorizer

data = fetch_california_housing(as_frame=True)
df = data.frame
Xcols = data.feature_names

print(df)
print('Features : ' , Xcols)
print('Target :' , data.target) 

print('Shape of dataframe :' , df.shape)

print(df.info())

print(df.describe())

print(df.isna().sum())


for col in df.columns:
    sns.histplot(x = col , data = df , kde = True , bins = 20)
    plt.title('Histogram of {}'.format(col))
    plt.show()
    plt.figure()


for col in df.columns:
    sns.boxplot(x = col , data = df)
    plt.title('Box Plot of {}'.format(col))
    plt.show()
    plt.figure()


sns.heatmap(df.corr() , annot = True, cmap = 'viridis')
plt.show()

winsorizer =  Winsorizer(capping_method= 'iqr' , tail = 'both' , fold = 3)

df[Xcols] = winsorizer.fit_transform(df[Xcols])

for col in df.columns:
    if col != 'MedHouseVal':
        sns.histplot(x = col , data = df , kde = True , bins = 20)
        plt.title('Histogram of {} after winsorizing'.format(col))
        plt.show()
        plt.figure()


X = df.drop('MedHouseVal' , axis = 1)
y = df['MedHouseVal']

X_train , X_test , y_train , y_test = train_test_split(X , y , shuffle=True , random_state=42 , test_size= 0.2)

model = RandomForestRegressor()

model.fit(X_train , y_train)

y_pred = model.predict(X_test)

print("R square : " , r2_score(y_test , y_pred))

print('Mean squared error : ' , mean_squared_error(y_test , y_pred))

print('Mean absolute error : ' , mean_absolute_error(y_test , y_pred))


joblib.dump(model , 'model.pkl' , compress = 3)
