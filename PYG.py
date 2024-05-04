from torch_geometric.data import HeteroData
import json
from torch_geometric.utils.convert import to_networkx
import networkx as nx
import matplotlib.pyplot as plt

Graph = HeteroData() # 实例化一个空对象

Tweet_List=[] #某一聚类中的Tweet文本id
user_list=[]  #某一聚类中的用户id
Oridinary_words_List=[] #某一聚类中的所有普通单词
Entity_words_List=[] #某一聚类中的所有实体单词
User_sents_Tweet_text_num=0     #推文文本−发送该推文的用户边数量
Entity_word_affiliated_with_Tweet_text=0    #推文文本−该推文中的实体单词边数量
Oridinary_word_affiliated_with_Tweet_text=0 #推文文本−该推文中的普通单词边数量
line_count=1

with open("Events.txt", "r", encoding="utf-8") as E:
    for line in E:
        if line_count==1:  #选择第i个簇画图
            data_E = json.loads(line)
            tweet_count=0
            while tweet_count < len(data_E['tweetList']):
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
                User_sents_Tweet_text_num+=1      #推文文本−发送该推文的用户
            else:
                continue
            words_count=0
            while words_count<len(data['words']):
                if data['words'][words_count]['ner']=='LOC' or data['words'][words_count]['ner']=='ORG' or data['words'][words_count]['ner']=='PER':
                    Entity_words_List.append(data['words'][words_count]['word'])
                    Entity_word_affiliated_with_Tweet_text+=1                   # 推文文本−该推文中的实体单词
                elif data['words'][words_count]['ner']=='O':
                    Oridinary_words_List.append(data['words'][words_count]['word'])
                    Oridinary_word_affiliated_with_Tweet_text+=1                # 推文文本−该推文中的普通单词
                else:
                    continue
                words_count+=1
        else:
            continue

Graph['Tweet_text'].x = [len(Tweet_List), 1]                #推文节点特征张量
Graph['User'].x = [len(user_list), 1]                       #用户节点特征张量
Graph['Oridinary_word'].x = [len(Oridinary_words_List), 1]  #普通单词节点特征张量
Graph['Entity_word'].x = [len(Entity_words_List), 1]        #实体单词节点特征张量

Graph['User', 'sents', 'Tweet_text'].edge_index = [2, User_sents_Tweet_text_num]                                     #边索引
Graph['Entity_word', 'affiliated_with', 'Tweet_text'].edge_index = [2, Entity_word_affiliated_with_Tweet_text]
Graph['Oridinary_word', 'affiliated_with', 'Tweet_text'].edge_index = [2, Oridinary_word_affiliated_with_Tweet_text]

Graph['User', 'sents', 'Tweet_text'].edge_attr = [User_sents_Tweet_text_num,1]                                     #边索引
Graph['Entity_word', 'affiliated_with', 'Tweet_text'].edge_attr = [Entity_word_affiliated_with_Tweet_text,1]
Graph['Oridinary_word', 'affiliated_with', 'Tweet_text'].edge_attr = [Oridinary_word_affiliated_with_Tweet_text,1]

Graph_networkx = to_networkx(Graph)
nx.draw(Graph_networkx, with_labels=False, node_color='red',node_size=10)
plt.show()
