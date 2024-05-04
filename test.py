import json

# Tweet_num=[]
#
# with open("Events.txt", "r", encoding="utf-8") as E:
#     for line in E:
#         data_E = json.loads(line)
#         Tweet_num.append(len(data_E['tweetList']))
#     print(Tweet_num)
from sklearn.model_selection import KFold
x=[1,2,3,4,5,6,7,8,9,10]
y=[11,12,13,14,15,16,17,18,19,20]
# print(y)
# KF = KFold(n_splits=5)
# print(KF)
# for train_index, test_index in KF.split(x):
#     print("TRAIN:", train_index, "TEST:", test_index)
# #
# for i in range (10,20):
#     print(i)

# x=[[1,2,3]]
# y =x[0]
# print(y)
# import csv
# f = open('get_vectors.csv','w',encoding='utf-8')
# csv_writer = csv.writer(f)
# csv_writer.writerow(x)
# csv_writer.writerow(y)

# with open('Get_Vectors.csv', 'r') as g:
#     for line in g:
#         print(line)
str='abcDEFabcDEFabc'
print(str)
str=str.split('DEF')  #根据'DEF'来分割字符串，返回分割结果的list
print(str)
