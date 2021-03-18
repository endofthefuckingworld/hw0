import pandas as pd 
import numpy as np 
import math
from matplotlib import pyplot as plt
import csv
plt.style.use("seaborn")

df = pd.read_csv('C:/Users/Terry/machine_learning hw1/Data/train.csv')
data = df.iloc[:, 3:].to_numpy()   #選取資料，去除排頭，轉換成numpy array

for i in range(len(data)):
    for j in range(len(data[i])):
        if data[i][j]=='NR':
            data[i][j]=0


month_data = {}                     #dictionary
for month in range(12):
    sample = np.empty([18, 480])    #row*column
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]    #18*24=18*24
    month_data[month] = sample

print(np.shape(month_data[1]))

x = np.empty([12 * 471, 18 * 9], dtype = float)   #以y為row,x(解釋變數18測項*9hr)為column
y = np.empty([12 * 471, 1], dtype = float)


for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day==19 and hour>14:  #每個月損失9hr
                continue
            x[month*471+(day*24)+hour,:]=month_data[month][:,day*24+hour:day*24+hour+9].reshape(1,18*9)
            y[month*471+(day*24)+hour,0]=month_data[month][9,(day*24)+hour+9]      #從第10hr到471hr


"""
 feature scaling:將所有解釋變數的範圍變為一樣，loss的圖像就會趨於圓形，尋找最佳解不會再繞路
                 將每一個樣本的所有解釋變數(x1,x2....xn)標準化(不用樣本標準差)，並不是x1個別標準化
"""

for i in range(len(x)):
    mean_x=np.mean(x[i])
    std_x=np.std(x[i])
    for j in range(len(x[i])):
        x[i][j]=(x[i][j]-mean_x)/std_x




x_2=np.zeros([np.shape(x)[0],np.shape(x)[1]])
x_2+=x**2
x_train=np.concatenate((x,x_2), axis=1).astype(float)
x_set = np.concatenate((np.ones([np.shape(x_train)[0], 1]), x_train), axis=1).astype(float)

parameter=2*18*9+1  #總共18測項*9hr+1截距
parameter_set=np.zeros([parameter,1])   #儲存所有參數值的array，先全部給0
learning_rate=20
adagrad=np.zeros([parameter,1])
eps=0.000000001
Lambda=0.1
iter_time=2000
regularization=np.empty([parameter, 1],dtype = float)




for i in range(iter_time):
    for k in range(parameter):
        regularization[k][0]=Lambda*2*parameter_set[k][0]
        
    gradient = 2 * np.dot(x_set.transpose(), np.dot(x_set, parameter_set) - y)+regularization 
        
    if(i==iter_time-1):
        """
        plt.plot(y, 'o')
        plt.plot(np.dot(x_set, parameter_set), '-')
        plt.show()
        """        
    adagrad += gradient ** 2
    parameter_set = parameter_set - learning_rate * gradient / np.sqrt(adagrad + eps)
    np.save('weight.npy', parameter_set)
print(parameter_set)

    

