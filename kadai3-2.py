# -*- coding: utf-8 -*-

# 行列演算ライブラリ
import numpy as np
# 描画ライブラリ
import matplotlib.pyplot as plt

# ファイルを開く
train1 = open("data/Train1.txt")
train2 = open("data/Train2.txt")
test1 = open("data/Test1.txt")
test2 = open("data/Test2.txt")

# 空データ作成
train1_data = []
train2_data = []
test1_data = []
test2_data = []

# データの整形
for row in train1.readlines():
    # 改行文字などを除いてデータ追加
    train1_data.append(map(float, row.strip().split()))
for row in train2.readlines():
    train2_data.append(map(float, row.strip().split()))
for row in test1.readlines():
    test1_data.append(map(float, row.strip().split()))
for row in test2.readlines():
    test2_data.append(map(float, row.strip().split()))

# numpy形式に変換
train1_data = np.array(train1_data)
train2_data = np.array(train2_data)
test1_data = np.array(test1_data)
test2_data = np.array(test2_data)

# ones1= np.ones([len(train1_data),1])
# ones2= -np.ones([len(train2_data),1])
# ones =np.r_[ones1,ones2]
#train1_dataとtrain2_dataをくっつける
X = np.r_[train1_data,train2_data]
# X = np.c_[ones,X]

right=np.zeros(10)
right1=np.zeros(10)
right2=np.zeros(10)
wrong=np.zeros(10)
for i in range(20):
    Y = np.array([[0,0]])
    for j in range(100):
        if j < 50:
            temp = np.array([[np.linalg.norm(X[j]-test1_data[i]),1]])
        else:
            temp = np.array([[np.linalg.norm(X[j]-test1_data[i]),-1]])
        Y=np.r_[Y,temp]
    Y = Y[1:101]
    order = Y[:,0].argsort()
    Y = np.take(Y,order,0)
    result = 0
    avr = 0
    print Y[0:10]
    for k in range(10):
        # print Y[k][1]
        result+=Y[k][1]
        avr -= Y[k][0]*Y[k][1]
        if result == 0:
            if avr>0:
                right1[k] += 1
                print "right!"
            else:
                wrong[k] += 1
                print "wrong!"
        elif result > 0:
            right1[k]+=1
            print "right!"
        else:
            wrong[k]+=1
            print "wrong!"


for i in range(20):
    Y = np.array([[0,0]])
    for j in range(100):
        if j < 50:
            temp = np.array([[np.linalg.norm(X[j]-test2_data[i]),1]])
        else:
            temp = np.array([[np.linalg.norm(X[j]-test2_data[i]),-1]])
        Y=np.r_[Y,temp]
    Y = Y[1:101]
    order = Y[:,0].argsort()
    Y = np.take(Y,order,0)
    result = 0
    avr = 0
    print Y[0:10]
    for k in range(10):
        result+=Y[k][1]
        avr -= Y[k][0]*Y[k][1]
        print result
        if result == 0:
            if avr<0:
                right2[k] += 1
                print "right!"
            else:
                wrong[k] += 1
                print "wrong!"
        elif result < 0:
            right2[k]+=1
            print "right!"
        else:
            wrong[k]+=1
            print "wrong!"
print right1,right2,wrong

right = right1 + right2


right =right*2.5
right1 =right1*5
right2 =right2*5

plt.plot(right,"b")
plt.plot(right1,"r")
plt.plot(right2,"g")

plt.title("kadai3-2")
plt.legend(["result","test1","test2"])

plt.xticks([0,1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9,10])
plt.ylim(70,100)

# 表示
plt.show()
