import json
import matplotlib.pyplot as plt
import numpy as np

Label=[]

def GetLabel():
    tweetList_n = []
    with open("Events.txt", "r",encoding = "utf-8") as f:
        for line in f:
            data = json.loads(line)
            # print(data)
            # print(data['describe'])
            # print(data['keyWords'])
            #print(len(data['tweetList']))
            tweetList_n.append(len(data['tweetList']))
    #print(tweetList_n)

    List_count=[]
    for i in tweetList_n:
        List_count.append(tweetList_n.count(i))
    # print(List_count)
    # print(len(List_count))

    for j in List_count:
        if j==1:
            Label.append(1)
        elif j!=1:
            Label.append(0)
    print(Label)
    print(len(Label))
    # number1=0
    # number_not1=0
    # for j in List_count:
    #     if j==1:
    #         number1+=1
    #     elif j!=1:
    #         number_not1+=1
    # # print(y)
    # print(number1)
    # print(number_not1)

if __name__=='__main__':
    GetLabel()
    Y = Label[9:20]
    print(Y)
