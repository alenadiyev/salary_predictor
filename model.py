import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import matplotlib.pyplot as plt
import seaborn as sn
# %matplotlib inline
import pickle

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv('salary_train.csv')

df1 = df[df['job'] == "economist"]
df2 = df[df['job'] == "data scientist"]
df3 = df[df['job'] == "robotics engineer"]
df4 = df[df['job'] == "developer"]
df5 = df[df['job'] == "junior developer"]
df6 = df[df['job'] == "senior developer"]

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, 6):
    ax = fig.add_subplot(2, 3, i)
    if i == 1:
        ax.scatter(df1['algebra'], df1['salary'], color='red')
        plt.title('algebra', fontsize=14)
    elif i == 2:
        plt.scatter(df1['programming'], df1['salary'], color='blue')
        plt.title('programming', fontsize=14)
    elif i == 3:
        plt.scatter(df1['data science'], df1['salary'], color='green')
        plt.title('data science', fontsize=14)
    elif i == 4:
        plt.scatter(df1['robotics'], df1['salary'], color='green')
        plt.title('robotics', fontsize=14)
    else:
        plt.scatter(df1['economics'], df1['salary'], color='green')
        plt.title('economics', fontsize=14)
    
corrMatrix = df1.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, 6):
    ax = fig.add_subplot(2, 3, i)
    if i == 1:
        ax.scatter(df2['algebra'], df2['salary'], color='red')
        plt.title('algebra', fontsize=14)
    elif i == 2:
        plt.scatter(df2['programming'], df2['salary'], color='blue')
        plt.title('programming', fontsize=14)
    elif i == 3:
        plt.scatter(df2['data science'], df2['salary'], color='green')
        plt.title('data science', fontsize=14)
    elif i == 4:
        plt.scatter(df2['robotics'], df2['salary'], color='green')
        plt.title('robotics', fontsize=14)
    else:
        plt.scatter(df2['economics'], df2['salary'], color='green')
        plt.title('economics', fontsize=14)

corrMatrix = df2.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, 6):
    ax = fig.add_subplot(2, 3, i)
    if i == 1:
        ax.scatter(df3['algebra'], df3['salary'], color='red')
        plt.title('algebra', fontsize=14)
    elif i == 2:
        plt.scatter(df3['programming'], df3['salary'], color='blue')
        plt.title('programming', fontsize=14)
    elif i == 3:
        plt.scatter(df3['data science'], df3['salary'], color='green')
        plt.title('data science', fontsize=14)
    elif i == 4:
        plt.scatter(df3['robotics'], df3['salary'], color='green')
        plt.title('robotics', fontsize=14)
    else:
        plt.scatter(df3['economics'], df3['salary'], color='green')
        plt.title('economics', fontsize=14)

corrMatrix = df3.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, 6):
    ax = fig.add_subplot(2, 3, i)
    if i == 1:
        ax.scatter(df4['algebra'], df4['salary'], color='red')
        plt.title('algebra', fontsize=14)
    elif i == 2:
        plt.scatter(df4['programming'], df4['salary'], color='blue')
        plt.title('programming', fontsize=14)
    elif i == 3:
        plt.scatter(df4['data science'], df4['salary'], color='green')
        plt.title('data science', fontsize=14)
    elif i == 4:
        plt.scatter(df4['robotics'], df4['salary'], color='green')
        plt.title('robotics', fontsize=14)
    else:
        plt.scatter(df4['economics'], df4['salary'], color='green')
        plt.title('economics', fontsize=14)

corrMatrix = df4.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, 6):
    ax = fig.add_subplot(2, 3, i)
    if i == 1:
        ax.scatter(df5['algebra'], df5['salary'], color='red')
        plt.title('algebra', fontsize=14)
    elif i == 2:
        plt.scatter(df5['programming'], df5['salary'], color='blue')
        plt.title('programming', fontsize=14)
    elif i == 3:
        plt.scatter(df5['data science'], df5['salary'], color='green')
        plt.title('data science', fontsize=14)
    elif i == 4:
        plt.scatter(df5['robotics'], df5['salary'], color='green')
        plt.title('robotics', fontsize=14)
    else:
        plt.scatter(df5['economics'], df5['salary'], color='green')
        plt.title('economics', fontsize=14)

corrMatrix = df5.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, 6):
    ax = fig.add_subplot(2, 3, i)
    if i == 1:
        ax.scatter(df6['algebra'], df6['salary'], color='red')
        plt.title('algebra', fontsize=14)
    elif i == 2:
        plt.scatter(df6['programming'], df6['salary'], color='blue')
        plt.title('programming', fontsize=14)
    elif i == 3:
        plt.scatter(df6['data science'], df6['salary'], color='green')
        plt.title('data science', fontsize=14)
    elif i == 4:
        plt.scatter(df6['robotics'], df6['salary'], color='green')
        plt.title('robotics', fontsize=14)
    else:
        plt.scatter(df6['economics'], df6['salary'], color='green')
        plt.title('economics', fontsize=14)

corrMatrix = df6.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

X_train_1 = df1[["algebra","programming", "data science", "robotics", "economics"]]
Y_train_1 = df1['salary']

X_train_2 = df2[["algebra","programming", "data science", "robotics", "economics"]]
Y_train_2 = df2['salary']
 
X_train_3 = df3[["algebra","programming", "data science", "robotics", "economics"]]
Y_train_3 = df3['salary']

X_train_4 = df4[["algebra","programming", "data science", "robotics", "economics"]]
Y_train_4 = df4['salary']

X_train_5 = df5[["algebra","programming", "data science", "robotics", "economics"]]
Y_train_5 = df5['salary']

X_train_6 = df6[["algebra","programming", "data science", "robotics", "economics"]]
Y_train_6 = df6['salary']


model_1 = linear_model.LinearRegression()
# X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(X_train_1, Y_train_1, test_size=0.2, random_state=42)
model_1.fit(X_train_1, Y_train_1)
pickle.dump(model_1, open('model_econ.pkl', 'wb'))
y_pred_1 = model_1.predict(X_train_1)
print("Error for economists:", np.sqrt(mean_squared_error(Y_train_1, y_pred_1)))

model_2 = linear_model.LinearRegression()
# X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_train_2, Y_train_2, test_size=0.2, random_state=42)
model_2.fit(X_train_2, Y_train_2)
pickle.dump(model_2, open('model_ds.pkl', 'wb'))
y_pred_2 = model_2.predict(X_train_2)
print("Error for data scientist:", np.sqrt(mean_squared_error(Y_train_2, y_pred_2)))

model_3 = linear_model.LinearRegression()
# X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X_train_3, Y_train_3, test_size=0.2, random_state=42)
model_3.fit(X_train_3, Y_train_3)
pickle.dump(model_3, open('model_rob.pkl', 'wb'))
y_pred_3 = model_3.predict(X_train_3)
print("Error for robotics engineers:", np.sqrt(mean_squared_error(Y_train_3, y_pred_3)))

model_4 = linear_model.LinearRegression()
# X_train_4, X_test_4, y_train_4, y_test_4 = train_test_split(X_train_4, Y_train_4, test_size=0.2, random_state=42)
model_4.fit(X_train_4, Y_train_4)
pickle.dump(model_4, open('model_dev.pkl', 'wb'))
y_pred_4 = model_4.predict(X_train_4)
print("Error for developers:", np.sqrt(mean_squared_error(Y_train_4, y_pred_4)))

model_5 = linear_model.LinearRegression()
# X_train_5, X_test_5, y_train_5, y_test_5 = train_test_split(X_train_5, Y_train_5, test_size=0.2, random_state=42)
model_5.fit(X_train_5, Y_train_5)
pickle.dump(model_5, open('model_jundev.pkl', 'wb'))
y_pred_5 = model_5.predict(X_train_5)
print("Error for junior developers:", np.sqrt(mean_squared_error(Y_train_5, y_pred_5)))

model_6 = linear_model.LinearRegression()
# X_train_6, X_test_6, y_train_6, y_test_6 = train_test_split(X_train_6, Y_train_6, test_size=0.2, random_state=42)
model_6.fit(X_train_6, Y_train_6)
pickle.dump(model_6, open('model_sendev.pkl', 'wb'))
y_pred_6 = model_6.predict(X_train_6)
print("Error for senior developers:", np.sqrt(mean_squared_error(Y_train_6, y_pred_6)))

df_predict = pd.read_csv('salary_predict.csv')
jobs = df_predict['job'].tolist()
grades = df_predict[["algebra","programming", "data science", "robotics", "economics"]].to_numpy()
predictions = []

for i in range(len(jobs)):
    if jobs[i] == "economist":
        X_test = grades[i,[0,1,2,4]]
        y_pred = model_1.predict(X_test.reshape(-1, 1).T)
        predictions.append(y_pred[0])
    elif jobs[i] == 'data scientist':
        X_test = grades[i,[0,1,2]]
        y_pred = model_2.predict(X_test.reshape(-1, 1).T)
        predictions.append(y_pred[0])
    elif jobs[i] == 'robotics engineer':
        X_test = grades[i,[0,1,2,3]]
        y_pred = model_3.predict(X_test.reshape(-1, 1).T)
        predictions.append(y_pred[0])
    elif jobs[i] == 'developer':
        X_test = grades[i,[0,1]]
        y_pred = model_4.predict(X_test.reshape(-1, 1).T)
        predictions.append(y_pred[0])
    elif jobs[i] == 'junior developer':
        X_test = grades[i,[0,1]]
        y_pred = model_5.predict(X_test.reshape(-1, 1).T)
        predictions.append(y_pred[0])
    elif jobs[i] == 'senior developer':
        X_test = grades[i,[0,1,2,3]]
        y_pred = model_6.predict(X_test.reshape(-1, 1).T)
        predictions.append(y_pred[0])

print(len(predictions))
for i in range(len(predictions)):
    if predictions[i]>1000000:
        predictions[i] = 1000000

indeces = np.arange(9000, 10000, dtype=int)
data = np.column_stack((indeces, predictions))
df_final_final = pd.DataFrame(data, columns=['Id','salary'])
df_final_final["Id"] = df_final_final["Id"].astype(int)
df_final_final["salary"] = df_final_final["salary"].astype(int)
df_final_final

df_final_final.to_csv("salary_final_prediction.csv", index=False, header=True)
