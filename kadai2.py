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

# ここから重みの計算
#train1_dataとtrain2_dataをくっつける
X = np.r_[train1_data,train2_data]

ones= np.ones([len(X),1])

#Xの先頭列に1を入れる
X = np.c_[ones,X]

#train1の教師信号を１,train2の教師信号を-1とする
T1 = np.ones([len(train1_data),1])
T2 = -np.ones([len(train2_data),1])

T = np.r_[T1,T2]

W = np.array([[0.5],[0.5],[0.5]])

for i in range(1000):
    dt = np.dot(X,W) - T
    dj = np.dot(X.T,dt)
    W -= 0.001*dj
    if i%10 == 0:
        if np.linalg.norm(dj)<0.000001:
            print i
            print W
            break

invX = np.dot(np.linalg.inv(np.dot(X.T,X)),X.T)
W2 = np.dot(invX,T)

x = np.linspace(-3.5,4.5,100)
y1 = -(W[1]/W[2])*x-W[0]/W[2]
plt.plot(x,y1,"c")

y2 = -(W2[1]/W2[2])*x-W2[0]/W2[2]
plt.plot(x,y2,"m")
print W2

# train1_dataを青色の点で表示
plt.plot(train1_data[:,0],train1_data[:,1],"go",label="train1")

# train2_dataを赤色の点で表示
plt.plot(train2_data[:,0],train2_data[:,1],"yo",label="train2")


plt.plot(-5,0,"b^",label="test1-positive")
plt.plot(-5,0,"bv",label="test1-negative")
plt.plot(-5,0,"r^",label="test2-positive")
plt.plot(-5,0,"rv",label="test2-negative")

# test1_dataを点で表示
# for i in len(test1_data):
for i in range(20):
    if W[0]+W[1]*test1_data[i,0]+W[2]*test1_data[i,1] > 0:
        # plt.plot(test1_data[i,0],test1_data[i,1],"b^",label="test1-positive")
        plt.plot(test1_data[i,0],test1_data[i,1],"b^")
    else:
        # plt.plot(test1_data[i,0],test1_data[i,1],"bv",label="test1-negative")
        plt.plot(test1_data[i,0],test1_data[i,1],"bv")

# plt.plot(test1_data[:,0],test1_data[:,1],"b^",label="test1-positive")


for i in range(20):
# for i in len(test2_data):
    if W[0]+W[1]*test2_data[i,0]+W[2]*test2_data[i,1] < 0:
        # test1_dataを緑の点で表示
        # plt.plot(test2_data[i,0],test2_data[i,1],"r^",label="test2-positive")
        plt.plot(test2_data[i,0],test2_data[i,1],"r^")
    else:
        # plt.plot(test2_data[i,0],test2_data[i,1],"rv",label="test2-negative")
        plt.plot(test2_data[i,0],test2_data[i,1],"rv")

# test1_dataを緑の点で表示
# plt.plot(test2_data[:,0],test2_data[:,1],"r^",label="test2-positive")

#タイトルの表示
plt.title("kadai2")

# 凡例を表示
plt.legend(["line1","line2","train1","train2","test1-positive","test1-negative","test2-positive","test2-negative"],loc="upper left",numpoints=1,prop={"size":12})

plt.xlim(-4,5)

# 表示
plt.show()
