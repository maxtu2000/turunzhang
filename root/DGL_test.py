import dgl
import networkx as nx
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch

nx_g = nx.DiGraph()
# Add 3 nodes and two features for them
nx_g.add_nodes_from([0, 1, 2], feat1=np.zeros((3, 1)), feat2=np.ones((3, 1)))
# Add 2 edges (1, 2) and (2, 1) with two features, one being edge IDs
nx_g.add_edge(1, 2, weight=np.ones((1, 1)), eid=np.array([1]))
nx_g.add_edge(2, 1, weight=np.ones((1, 1)), eid=np.array([0]))

# print(nx_g.nodes)
# print(type(nx_g.nodes))
nlist = sorted(nx_g.nodes())
# nx.draw(nx_g, with_labels=False, node_color='red',node_size=100)
# plt.show()
g = dgl.from_networkx(nx_g)




g1 = dgl.graph(([0, 1], [1, 0]))
g1.ndata['h'] = torch.tensor([1., 2.])
g2 = dgl.graph(([0, 1], [1, 2]))
g2.ndata['h'] = torch.tensor([1., 2., 3.])

dgl.readout_nodes(g1, 'h')
# tensor([3.])  # 1 + 2

bg = dgl.batch([g1, g2])
dgl.readout_nodes(bg, 'h')
# tensor([3., 6.])  # [1 + 2, 1 + 2 + 3]
print(g1.ndata)