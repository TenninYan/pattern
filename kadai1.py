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

X = []
# ここから重みの計算
#train1_dataとtrain2_dataをくっつける
X = np.r_(train1_data,train2_data)

ones= np.ones([len(X),1])

#Xの先頭列に1を入れる
X = np.c_(ones,X)

#train1の教師信号を１,train2の教師信号を-1とする
T1 = np.ones([len(train1_data),1])
T2 = -np.ones([len(train2_data),1])

T = np.r_[T1,T2]

W = np.array([[0.5],[0.5],[0.5]])

for i in range(100):
    dt = np.dot(X,W) - T
    dj = np.dot(X.T,dt)
    W -= dj
    print W




# # train1_dataを青色の点で表示
# plt.plot(train1_data[:,0],train1_data[:,1],"bo",label="train1")
#
# # train2_dataを赤色の点で表示
# plt.plot(train2_data[:,0],train2_data[:,1],"ro",label="train2")
#
# # test1_dataを緑の点で表示
# plt.plot(test1_data[:,0],test1_data[:,1],"g^",label="test1-positive")
#
# # test1_dataを緑の点で表示
# plt.plot(test2_data[:,0],test2_data[:,1],"y^",label="test2-positive")
#
# #タイトルの表示
# plt.title("kadai1")
#
# # 凡例を表示
# plt.legend()
#
# # 表示
# plt.show()
