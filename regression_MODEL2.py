import pandas as pd 
import numpy as np 
import math
from matplotlib import pyplot as plt
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

"""
模型的誤差主要來自bias,variance

當你今天的模型連training data都適配不好，underfitting(雖然variance小，但你model但你model set根本不涵蓋最佳)這時拿更多資料是沒有用的
你必須加入更多解釋變數
但是當你training 適配的好但testing卻不如預期，代表bias小，variance大
解決之道:更多data ,regularization使圖形更平滑，variance變小
Split Training Data Into "train_set" and "validation_set" 由於如果只有training,testing set的話，假如
你的目標是loss<0.5，很容易因為選了誤差最小的testing，且覺得自己達成目標，但實際上的資料是完全未知，或許該模型只是在該testing
表現較好

n-
"""
Nfolds_cross_validation_x={}
Nfolds_cross_validation_y={}
Nfolds_cross_validation_x[0]=x[:math.floor(0.3333*len(x)),:]
Nfolds_cross_validation_y[0]=y[:math.floor(0.3333*len(y)),:]
Nfolds_cross_validation_x[1]=x[math.floor(0.3333*len(x)):math.floor(0.6667*len(x)),:]
Nfolds_cross_validation_y[1]=y[math.floor(0.3333*len(y)):math.floor(0.6667*len(y)),:]
Nfolds_cross_validation_x[2]=x[math.floor(0.6667*len(x)):,:]
Nfolds_cross_validation_y[2]=y[math.floor(0.6667*len(y)):,:]


#Model 2(全部參數一次項and全部參數二次項)
parameter=2*18*9+1  #總共18測項*9hr+1截距
parameter_set=np.zeros([parameter,1])   #儲存所有參數值的array，先全部給0
learning_rate=20
adagrad=np.zeros([parameter,1])
eps=0.000000001
Lambda=0.1
iter_time=2000
regularization=np.empty([parameter, 1],dtype = float)

for u in range(3):
    Nfolds=[[0,1,2],[1,2,0],[0,2,1]]
    xt_1=np.concatenate((Nfolds_cross_validation_x[Nfolds[u][0]],Nfolds_cross_validation_x[Nfolds[u][1]]), axis=0).astype(float)
    xt_2=np.zeros([np.shape(xt_1)[0],np.shape(xt_1)[1]])
    xt_2+=xt_1**2
    x_train=np.concatenate((xt_1,xt_2), axis=1).astype(float)
    
    y_train=np.concatenate((Nfolds_cross_validation_y[Nfolds[u][0]],Nfolds_cross_validation_y[Nfolds[u][1]]), axis=0).astype(float)
    xv_1=Nfolds_cross_validation_x[Nfolds[u][2]]
    xv_2=np.zeros([np.shape(xv_1)[0],np.shape(xv_1)[1]])
    xv_2+=xv_1**2
    x_valid=np.concatenate((xv_1,xv_2), axis=1).astype(float)
    y_valid=Nfolds_cross_validation_y[Nfolds[u][2]]
    x_set = np.concatenate((np.ones([np.shape(x_train)[0], 1]), x_train), axis=1).astype(float) #在此利用原x矩陣+截距項(全部設1)
    x_valid_set = np.concatenate((np.ones([np.shape(x_valid)[0], 1]), x_valid), axis=1).astype(float)
    for i in range(iter_time):
        for k in range(parameter):
            regularization[k][0]=2*Lambda*parameter_set[k][0]
        gradient = 2 * np.dot(x_set.transpose(), np.dot(x_set, parameter_set) - y_train)+regularization 
        
        if(i==iter_time-1):
            """
            plt.plot(y, 'o')
            plt.plot(np.dot(x_set, parameter_set), '-')
            plt.show()
            """
            
            
        adagrad += gradient ** 2
        parameter_set = parameter_set - learning_rate * gradient / np.sqrt(adagrad + eps)
        np.save('weight.npy', parameter_set)

    loss = np.sqrt(np.sum(np.power(np.dot(x_valid_set, parameter_set) - y_valid, 2))/np.shape(x_valid_set)[0])#rmse
    print(str(loss))
