import json
import networkx as nx
import matplotlib.pyplot as plt
from graph2vec import Get_graph_embedding
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import csv

def Get_graph(ID):
    G = nx.Graph()
    Tweet_List=[] #某一聚类中的Tweet文本id
    user_list=[]  #某一聚类中的用户id
    Oridinary_words_List=[] #某一聚类中的所有普通单词
    Entity_words_List=[] #某一聚类中的所有实体单词
    count=0       #染色计数变量
    colors = []   #颜色向量
    line_count=1

    with open("Events.txt", "r", encoding="utf-8") as E:
        for line in E:
            if line_count==ID:  #选择第i个簇画图
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
                        G.add_node(words_count, label='Entity_word')   #添加实体单词节点
                        G.add_edge(data['id'], words_count)  # 连接：推文文本−该推文中的实体单词
                    elif data['words'][words_count]['ner']=='O':
                        Oridinary_words_List.append(data['words'][words_count]['word'])
                        G.add_node(words_count+len(data['words']), label='Ordinary_word') #添加普通单词节点
                        G.add_edge(data['id'], words_count+len(data['words']))  # 连接：推文文本−该推文中的普通单词
                    else:
                        continue
                    words_count+=1
            else:
                continue
    return G
    # while count<G.number_of_nodes():                                        #为不同种类节点上色
    #     if list(G.nodes(data=True))[count][1]['label']=='tweet_text':       # 红色：文本节点 蓝色：用户节点 绿色：实体单词 黄色：普通单词
    #         colors.append('red')
    #         count +=1
    #     elif list(G.nodes(data=True))[count][1]['label']=='user':
    #         colors.append('blue')
    #         count +=1
    #     elif list(G.nodes(data=True))[count][1]['label']=='Entity_word':
    #         colors.append('green')
    #         count +=1
    #     elif list(G.nodes(data=True))[count][1]['label']=='Ordinary_word':
    #         colors.append('yellow')
    #         count +=1
    #     else:
    #         continue
    # nx.draw(G, with_labels=False, node_color=colors,node_size=10)
    # plt.show()

def Get_label():
    Label=[]  #存储所有的标签
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
    return Label


def Graph_to_vec(graph):
    class args:                        #graph2vec所需参数
        def __init__(self):
            self.wl_iterations = 2
            self.output_path = "./features/ret.csv"
            self.dimensions = 128
            self.workers =4
            self.epochs=10
            self.min_count=5
            self.learning_rate = 0.025
            self.down_sampling = 0.0001

    arg=args()
    Vector=Get_graph_embedding(arg,graph,'test')
    vec = Vector[0]
    return vec

if __name__=='__main__':
    Label=Get_label()
    X=[]
    for i in range (3,501):
        graph=Get_graph(i)
        vec=Graph_to_vec(graph)
        X.append(vec)
    Y=Label[2:500]

    # with open('Get_Vectors.txt', 'w') as f:
    #     f.write(str(X) + '\n')
    #     f.write(str(Y))
    f = open('Get_vectors.csv', 'w', encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(X)
    csv_writer.writerow(Y)

    #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state = 1)
    scores = cross_val_score(clf, X, Y, cv=5)
    print(scores)
    print("准确率:",scores.mean())
    print("方差:", scores.var())
    print("标准差:", scores.std())
   # print("准确率：",clf.score(X_test, y_test))




