import pandas as pd 
import numpy as np 
import csv
import math
from matplotlib import pyplot as plt
plt.style.use("seaborn")


testdata = pd.read_csv('./Data/test.csv', header = None, encoding = 'big5')
ans=pd.read_csv('./Data/ans.csv', header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
test_data = test_data.to_numpy()

for i in range(len(test_data)):
    for j in range(len(test_data[i])):
        if test_data[i][j]=='NR':
            test_data[i][j]=0


test_x = np.empty([240, 18*9], dtype = float)
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)

for i in range(len(test_x)):
    mean_x=np.mean(test_x[i])
    std_x=np.std(test_x[i])
    for j in range(len(test_x[i])):
        test_x[i][j]=(test_x[i][j]-mean_x)/std_x

x_2=np.zeros([np.shape(test_x)[0],np.shape(test_x)[1]])
x_2+=test_x**2
x_train=np.concatenate((test_x,x_2), axis=1).astype(float)
x_set = np.concatenate((np.ones([np.shape(x_train)[0], 1]), x_train), axis=1).astype(float)


w = np.load('weight.npy')
ans_y = np.dot(x_set, w)


with open('submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        print(row)
        csv_writer.writerow(row)


plt.plot(ans, 'o')
plt.plot(ans_y, '-', label='Logistic Model')
plt.title('Regression-Model2') # title
plt.show()


