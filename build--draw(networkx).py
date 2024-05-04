import json
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
Tweet_List=[] #某一聚类中的Tweet文本id
user_list=[]  #某一聚类中的用户id
Oridinary_words_List=[] #某一聚类中的所有普通单词
Entity_words_List=[] #某一聚类中的所有实体单词
count=0       #染色计数变量
colors = []   #颜色向量
line_count=1
Label=[]      #存储每一张图的标签

with open("Events.txt", "r", encoding="utf-8") as E:
    for line in E:
        if line_count==4:  #选择第i个簇画图
            data_E = json.loads(line)
            tweet_count=0
            while tweet_count < len(data_E['tweetList']):
                G.add_node(data_E['tweetList'][tweet_count][0],label='tweet_text')  #添加文本节点
                Tweet_List.append(data_E['tweetList'][tweet_count][0])
                tweet_count += 1
            break
        else:
            line_count+=1

with open("Tweets.txt", "r",encoding = "utf-8") as T:
    for line in T:
        data = json.loads(line)
        #print((data))
        if data['id'] in Tweet_List:
            if data['user']['id'] not in user_list:
                user_list.append(data['user']['id'])
                G.add_node(data['user']['id'], label='user')    #添加用户节点
                G.add_edge(data['id'], data['user']['id'])      #连接：推文文本−发送该推文的用户
            else:
                continue
            words_count=0
            while words_count<len(data['words']):
                if data['words'][words_count]['ner']=='LOC' or data['words'][words_count]['ner']=='ORG' or data['words'][words_count]['ner']=='PER':
                    Entity_words_List.append(data['words'][words_count]['word'])
                    G.add_node(data['words'][words_count]['word'], label='Entity_word')   #添加实体单词节点
                    G.add_edge(data['id'], data['words'][words_count]['word'])  # 连接：推文文本−推文文本−该推文中的实体单词
                elif data['words'][words_count]['ner']=='O':
                    Oridinary_words_List.append(data['words'][words_count]['word'])
                    G.add_node(data['words'][words_count]['word'], label='Ordinary_word') #添加普通单词节点
                    G.add_edge(data['id'], data['words'][words_count]['word'])  # 连接：推文文本−推文文本−该推文中的普通单词
                else:
                    continue
                words_count+=1
        else:
            continue

while count<G.number_of_nodes():                                        #为不同种类节点上色
    if list(G.nodes(data=True))[count][1]['label']=='tweet_text':       # 红色：文本节点 蓝色：用户节点 绿色：实体单词 黄色：普通单词
        colors.append('red')
        count +=1
    elif list(G.nodes(data=True))[count][1]['label']=='user':
        colors.append('blue')
        count +=1
    elif list(G.nodes(data=True))[count][1]['label']=='Entity_word':
        colors.append('green')
        count +=1
    elif list(G.nodes(data=True))[count][1]['label']=='Ordinary_word':
        colors.append('yellow')
        count +=1
    else:
        continue

# print(len(Tweet_List))
# print(len(user_list))
# print(Entity_words_List)
# print(Oridinary_words_List)

nx.draw(G, with_labels=False, node_color=colors,node_size=10)  ##画图
plt.show()

#######################################################获取每一张图的标签##################################
tweetList_n = [] #存储每一类的文本数
with open("Events.txt", "r",encoding = "utf-8") as f:
    for line in f:
        data = json.loads(line)
        # print(data)
        # print(data['describe'])
        # print(data['keyWords'])
        #print(len(data['tweetList']))
        tweetList_n.append(len(data['tweetList']))
#print(tweetList_n)

List_count=[] #存储每一类的文本数出现的次数
for i in tweetList_n:
    List_count.append(tweetList_n.count(i))
# print(List_count)
# print(len(List_count))

for j in List_count:
    if j==1:
        Label.append(1)
    elif j!=1:
        Label.append(0)
# print(Label)
# print(len(Label))

#####################################################################################################

