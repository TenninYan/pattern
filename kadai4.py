# -*- coding: utf-8 -*-

# 行列演算ライブラリ
import numpy as np
# 描画ライブラリ
import matplotlib.pyplot as plt

# ファイルを開く
omega1 = open("data/omega1.txt")
omega2 = open("data/omega2.txt")
omega3 = open("data/omega3.txt")

# 空データ作成
omega1_data = []
omega2_data = []
omega3_data = []

# データの整形
for row in omega1.readlines():
    # 改行文字などを除いてデータ追加
    omega1_data.append(map(float, row.strip().split()))
for row in omega2.readlines():
    omega2_data.append(map(float, row.strip().split()))
for row in omega3.readlines():
    omega3_data.append(map(float, row.strip().split()))

# numpy形式に変換
omega1_data = np.array(omega1_data)
omega2_data = np.array(omega2_data)
omega3_data = np.array(omega3_data)
omega_data = np.r_[omega1_data,omega2_data,omega3_data]
omega_data.resize((3,10,3))

u1 =  np.mean(omega1_data,axis=0)
u2 =  np.mean(omega2_data,axis=0)
u3 =  np.mean(omega3_data,axis=0)
u = np.r_[u1,u2,u3]
u.resize((3,3))

test = np.array([[1,2,1],[5,3,2],[0,0,0],[1,0,0]])

length = np.zeros((3,4))
g = np.zeros((3,4))
for k in range(3):
    for i in range(3):
        x1 = omega_data[k].T[0]-u[k][0]
        x2 = omega_data[k].T[1]-u[k][1]
        x3 = omega_data[k].T[2]-u[k][2]
        x = np.r_[x1,x2,x3]
        x.resize(3,10)
    
        # S = np.zeros((3,3))
        # for a in range(3):
        #     for b in range(3):
        #         S[a][b]=np.dot(x[a],x[b].T)/10
        S = np.cov(omega_data[k],rowvar=0)
        S_det = np.linalg.det(S)
        S_inv = np.linalg.inv(S)
        for j in range(4):
            # length[j] = np.dot((test[j]-u[0]),S,(test[j]-u[0]).T)
            delta = test[j]-u[k]
            # length[k][j] = np.sqrt(np.dot(temp2,temp.T))
            length[k][j] = np.sqrt(np.dot(np.dot(delta,S_inv),delta.T))
            g[k][j] = -0.5*length[k][j]-0.5*np.log(S_det)-3/2*np.log(np.pi)+np.log(0.3)

# # ここから重みの計算
# #train1_dataとtrain2_dataをくっつける
# X = np.r_[train1_data,train2_data]
#
# ones= np.ones([len(X),1])
#
# #Xの先頭列に1を入れる
# X = np.c_[ones,X]
#
# #train1の教師信号を１,train2の教師信号を-1とする
# T1 = np.ones([len(train1_data),1])
# T2 = -np.ones([len(train2_data),1])
#
# T = np.r_[T1,T2]
#
# W = np.array([[0.5],[0.5],[0.5]])
#
# for i in range(1000):
#     dt = np.dot(X,W) - T
#     dj = np.dot(X.T,dt)
#     W -= 0.001*dj
#     if i%10 == 0:
#         if np.linalg.norm(dj)<0.000001:
#             print i
#             print W
#             break
#
# x = np.linspace(-3.5,4.5,100)
# y1 = -(W[1]/W[2])*x-W[0]/W[2]
# plt.plot(x,y1,"c")
#
# # train1_dataを青色の点で表示
# plt.plot(train1_data[:,0],train1_data[:,1],"go",label="train1")
#
# # train2_dataを赤色の点で表示
# plt.plot(train2_data[:,0],train2_data[:,1],"yo",label="train2")
#
#
# plt.plot(-5,0,"b^",label="test1-positive")
# plt.plot(-5,0,"bv",label="test1-negative")
# plt.plot(-5,0,"r^",label="test2-positive")
# plt.plot(-5,0,"rv",label="test2-negative")
#
# # test1_dataを点で表示
# # for i in len(test1_data):
# for i in range(20):
#     if W[0]+W[1]*test1_data[i,0]+W[2]*test1_data[i,1] > 0:
#         # plt.plot(test1_data[i,0],test1_data[i,1],"b^",label="test1-positive")
#         plt.plot(test1_data[i,0],test1_data[i,1],"b^")
#     else:
#         # plt.plot(test1_data[i,0],test1_data[i,1],"bv",label="test1-negative")
#         plt.plot(test1_data[i,0],test1_data[i,1],"bv")
#
# # plt.plot(test1_data[:,0],test1_data[:,1],"b^",label="test1-positive")
#
#
# for i in range(20):
# # for i in len(test2_data):
#     if W[0]+W[1]*test2_data[i,0]+W[2]*test2_data[i,1] < 0:
#         # test1_dataを緑の点で表示
#         # plt.plot(test2_data[i,0],test2_data[i,1],"r^",label="test2-positive")
#         plt.plot(test2_data[i,0],test2_data[i,1],"r^")
#     else:
#         # plt.plot(test2_data[i,0],test2_data[i,1],"rv",label="test2-negative")
#         plt.plot(test2_data[i,0],test2_data[i,1],"rv")
#
# # test1_dataを緑の点で表示
# # plt.plot(test2_data[:,0],test2_data[:,1],"r^",label="test2-positive")
#
# #タイトルの表示
# plt.title("kadai1")
#
# # 凡例を表示
# plt.legend(["line1","train1","train2","test1-positive","test1-negative","test2-positive","test2-negative"],loc="upper left",numpoints=1,prop={"size":12})
#
# plt.xlim(-4,5)
#
# # 表示
# plt.show()
