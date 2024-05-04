# with open("Tweets.txt", "r") as f:
#     for line in f.readlines():
#         line = line.strip('\n')  #去掉列表中每一个元素的换行符
#         print(line)
import json
import matplotlib.pyplot as plt
import numpy as np

def openE():
    tweetList_n = []
    with open("Events.txt", "r",encoding = "utf-8") as f:
        for line in f:
            data = json.loads(line)
            #print(data)
            #print(data['describe'])
            # print(data['keyWords'])
            #print(len(data['tweetList']))
            tweetList_n.append(len(data['tweetList']))
    print(tweetList_n)
    n=0
    for i in tweetList_n:
        if i>=1000:
            print(n,i)
            n+=1
        else:
            n+=1

    # count=[]
    # x=[]
    # y=[]
    # for i in tweetList_n:
    #     if i not in count:
    #         count.append(i)
    #
    # for i in count:
    #     #print(f"{i}: {tweetList_n.count(i)}")
    #     if i < 1000:
    #         x.append(i)
    #         print(i)
    #         y.append(tweetList_n.count(i))
    #         print(y)
    #     else:
    #         continue

    # plt.bar(x,y)
    # plt.show()

def openT():
    with open("Tweets.txt", "r",encoding = "utf-8") as f:
        NER=[]
        for line in f:
            data = json.loads(line)
           #print(data)
            #print(data['id'])
            #print(data['user'])
            #print(type(data['user']))
            #print(data['user']['id'])
            #print(len(data['words']))
            count=0
            while count<len(data['words']):
                if data['words'][count]['ner'] not in NER:
                    NER.append(data['words'][count]['ner'])
                count+=1
        print(NER)

def watch_tweets():
    line_count = 1
    Tweet_List=[]
    Tweets=[]
    with open("Events.txt", "r", encoding="utf-8") as E:
        for line in E:
            if line_count==4:  #选择第i个簇
                data_E = json.loads(line)
                tweet_count=0
                while tweet_count < len(data_E['tweetList']):
                    Tweet_List.append(data_E['tweetList'][tweet_count][0])
                    print(data_E['describe'])
                    tweet_count += 1
                break
            else:
                line_count+=1
    with open("Tweets.txt", "r", encoding="utf-8") as T:
        for line in T:
            data = json.loads(line)
            # print((data))
            if data['id'] in Tweet_List:
                Tweets.append(data['text'])
    print(Tweets)
    print(len(Tweets))

def watch_text():
    line_count=1
    text=[]
    with open("Events.txt", "r", encoding="utf-8") as E:
        for line in E:
            if line_count == 4:
                data_E = json.loads(line)
                text.append(data_E['describe'])
                break
            else:
                line_count += 1
    print(text)
if __name__=='__main__':
    #watch_tweets()
   watch_tweets()