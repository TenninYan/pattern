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
wrong=np.zeros(10)
for i in range(100):
    Y = np.array([[0,0]])
    for j in range(100):
        if j < 50:
            temp = np.array([[np.linalg.norm(X[j]-X[i]),1]])
        else:
            temp = np.array([[np.linalg.norm(X[j]-X[i]),-1]])
        Y=np.r_[Y,temp]
    Y = Y[1:101]
    order = Y[:,0].argsort()
    Y = np.take(Y,order,0)

    if i<50:
    # if i == 1:
        # print Y[0:10]
        result = 0
        avr = 0
        for k in range(10):
            result += Y[k+1][1]
            avr -= Y[k+1][0]*Y[k+1][1]
            # print result
            # print avr
            if result == 0:
                if avr>0:
                    right[k] += 1
                else:
                    wrong[k] += 1
            elif result > 0:
                right[k] += 1
            else:
                wrong[k] += 1


    else:
        result = 0
        avr = 0
        for k in range(10):
            result += Y[k+1][1]
            avr -= Y[k+1][0]*Y[k+1][1]
            # print result
            # print avr
            if result == 0:
                if avr<0:
                    right[k] += 1
                else:
                    wrong[k] += 1
            elif result < 0:
                right[k] += 1
            else:
                wrong[k] += 1




# train1_dataを青色の点で表示
plt.plot(right,"b")

# train2_dataを赤色の点で表示
# plt.plot(train2_data[:,0],train2_data[:,1],"yo",label="train2")


plt.title("kadai3-1")

# 凡例を表示
# plt.legend(["line1","train1","train2","test1-positive","test1-negative","test2-positive","test2-negative"],loc="upper left",numpoints=1,prop={"size":12})
plt.legend(["result"])

plt.xticks([0,1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9,10])
plt.ylim(81,88)

plt.xlabel("k")
plt.ylabel("accuracy (%)")

# 表示
plt.show()
