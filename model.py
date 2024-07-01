import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('housing.csv')

print(df)

print(df.info())

print(df.isna().sum())

'''
for col in df.columns:
    sns.histplot(x = col , data = df , kde = True)
    plt.title('Histogram of {}'.format(col))
    plt.show()
    plt.figure()


for col in df.columns:
    sns.violinplot(x = col , data = df)
    plt.title('Voilin Plot of {}'.format(col))
    plt.show()
    plt.figure()

for i, col in enumerate(df.columns):
    for j, col2 in enumerate(df.columns):
        if j > i and col != col2:
            sns.scatterplot(x=col, y=col2, data=df, hue="MEDV")
            plt.title('Scatter Plot: {} vs {}'.format(col , col2))
            plt.xlabel(col)
            plt.ylabel(col2)
            plt.show()
            plt.figure()


sns.heatmap(df.corr() , annot = True, cmap = 'viridis')
plt.show()
'''
X = df.drop('MEDV' , axis = 1)
y = df['MEDV']

X_train , X_test , y_train , y_test = train_test_split(X , y , shuffle=True , random_state=42 , test_size= 0.2)

model = RandomForestRegressor()

model.fit(X_train , y_train)

y_pred = model.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

sns.lmplot(x = 'Actual',y = 'Predicted' , data = df)

plt.title('Regression plot with actual vs predicted values')
plt.show()



