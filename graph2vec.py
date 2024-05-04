"""Graph2Vec module."""

import os
import json
import glob
import hashlib
import time

import pandas as pd
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed
# from param_parser import parameter_parser
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

class WeisfeilerLehmanMachine:
    """
    Weisfeiler Lehman feature extractor class.
    """
    def __init__(self, graph, features, iterations):
        """
        Initialization method which also executes feature extraction.
        :param graph: The Nx graph object.
        :param features: Feature hash table.
        :param iterations: Number of WL iterations.
        """
        self.iterations = iterations
        self.graph = graph
        self.features = features
        self.nodes = self.graph.nodes()
        self.extracted_features = [str(v) for k, v in features.items()]
        self.do_recursions()

    def do_a_recursion(self):
        """
        The method does a single WL recursion.
        :return new_features: The hash table with extracted WL features.
        """
        new_features = {}
        for node in self.nodes:
            # 获取当前节点的邻居
            nebs = self.graph.neighbors(node)
            # 得到邻居节点所对应的节点度分布列表
            degs = [self.features[neb] for neb in nebs]
            # 当前节点的度分布列表+其邻居节点排序后的度分布列表
            features = [str(self.features[node])]+sorted([str(deg) for deg in degs])
            features = "_".join(features)
            # 采用md5加密当前节点特征
            hash_object = hashlib.md5(features.encode())
            # 得到加密和的序列
            hashing = hash_object.hexdigest()
            # 把加密后的序列作为当前节点新的特征
            new_features[node] = hashing
        self.extracted_features = self.extracted_features + list(new_features.values())

        return new_features

    def do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for _ in range(self.iterations):
            self.features = self.do_a_recursion()

def path2name(path):
    base = os.path.basename(path)
    return os.path.splitext(base)[0]

def dataset_reader(path):
    """
    Function to read the graph and features from a json file.
    :param path: The path to the graph json.
    :return graph: The graph object.
    :return features: Features hash table.
    :return name: Name of the graph.
    """
    name = path2name(path)
    data = json.load(open(path))
    graph = nx.from_edgelist(data["edges"])

    if "features" in data.keys():
        features = data["features"]
        features = {int(k): v for k, v in features.items()}
    else:
        features = nx.degree(graph)
        features = {int(k): v for k, v in features}
       
    return graph, features, name

def feature_extractor(graph, name, rounds):
    """
    Function to extract WL features from a graph.
    :param path: The path to the graph json.
    :param rounds: Number of WL iterations.
    :return doc: Document collection object.
    """
    # 采用原算法中无标签情况给图节点设置特征
    features = nx.degree(graph)
    features = {int(k): v for k, v in features}
    machine = WeisfeilerLehmanMachine(graph, features, rounds)
    doc = TaggedDocument(words=machine.extracted_features, tags=["g_" + name])
    return doc

def return_graph_embedding(output_path, model, name, dimensions):
    """
    Function to save the embedding.
    :param output_path: Path to the embedding csv.
    :param model: The embedding model object.
    :param files: The list of files.
    :param dimensions: The embedding dimension parameter.
    """
    out = []
    out.append(list(model.docvecs["g_"+name]))
    
    return out

def Get_graph_embedding(args, graph, name):
    """
    Main function to read the graph list, extract features.
    Learn the embedding and save it.
    :param args: Object with the arguments.
    """
    document_collection = feature_extractor(graph, name, args.wl_iterations)
    document_collections = [document_collection]
    model = Doc2Vec(document_collections,
                    vector_size=args.dimensions,
                    window=0,
                    min_count=args.min_count,
                    dm=0,
                    sample=args.down_sampling,
                    workers=args.workers,
                    epochs=args.epochs,
                    alpha=args.learning_rate)

    graph_embedding = return_graph_embedding(args.output_path, model, name, args.dimensions)

    return graph_embedding

