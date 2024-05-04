import networkx as nx
import node2vec
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

graph = nx.fast_gnp_random_graph(n=100,p=6.5)
# Precompute probabilities and generate walks - **ON MINDOwS ONLY WORKS WITH workers=1**
# nx.draw(graph, with_labels=False, node_color='red',node_size=100)  ##画图
# plt.show()
mynode2vec = node2vec.Node2Vec(graph,dimensions=64,walk_length=30,num_walks=200,workers=4) # Use temp_folder for big graph
#Embed nodes
model = mynode2vec.fit(window=10,min_count=1, batch_words=4)# Any keywords acceptable by gensim.word2Vec can be pas:
#Look for most similar nodes
model.wv.most_similar('2') # output node names are always strings
#Save embeddings for later use
model.wv.save_word2vec_format('model')
# Save model for later use
model.save('model')

print(type(mynode2vec))
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(model)