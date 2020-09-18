'''
Reference implementation of node2vec. 
Author: Aditya Grover
For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import sys
import os
import re
from pathlib import Path

import networkx as nx
from node2vec import Graph
from gensim.models import Word2Vec


def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='../graph/graph-v4.1.edgelist', help='Input graph path')

    parser.add_argument('--output', nargs='?', default='../graph/coauthor-v4.1.emb', help='Embeddings path')

    parser.add_argument('--output-model', nargs='?', default='../graph/coauthor-v4.1.model', help='Embeddings model path')

    parser.add_argument('--dimensions', type=int, default=128, help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80, help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10, help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10, help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=1, type=int, help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1, help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1, help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=True)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()


def read_graph():
    '''
    Reads the input network in networkx.
    '''
    if args.weighted:
        G = nx.read_edgelist(args.input, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        G = G.to_undirected()

    return G


def learn_embeddings(walks):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [['a' + str(x) for x in walk] for walk in walks]
    # walks = [map(str, walk) for walk in walks]
    print(walks[0])
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers,
                     iter=args.iter)
    # model.save_word2vec_format(args.output)
    model.wv.save_word2vec_format(args.output)
    model.save(args.output_model)

    return


def main(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    for fellow_type in ['acm', 'ieee']:
        print('******* {} *******'.format(fellow_type))
        graph_data_dir = os.path.join('..', '..', '..', 'data', 'graph')
        edge_list_files_dir_path = os.path.join(graph_data_dir, 'edgelist', fellow_type)
        pattern = re.compile('graph-(\d+)\.edgelist')

        edge_file_names = os.listdir(edge_list_files_dir_path)
        edge_file_names.sort()
        for edge_file_name in edge_file_names:
            matcher = pattern.match(edge_file_name)
            if not matcher:
                continue

            current_year = matcher.group(1)
            print('Processing year: {}'.format(current_year))

            model_dir = os.path.join(graph_data_dir, 'model', fellow_type)
            emb_dir = os.path.join(graph_data_dir, 'emb', fellow_type)

            Path(model_dir).mkdir(parents=True, exist_ok=True)
            Path(emb_dir).mkdir(parents=True, exist_ok=True)

            args.input = os.path.join(edge_list_files_dir_path, edge_file_name)
            args.output = os.path.join(emb_dir, 'coauthor-{}.emb'.format(current_year))
            args.output_model = os.path.join(model_dir, 'coauthor-{}.model'.format(current_year))

            print('Reading Graph...')
            sys.stdout.flush()
            nx_G = read_graph()
            G = Graph(nx_G, args.directed, args.p, args.q)
            print('Preprocessing of transition probabilities...')
            G.preprocess_transition_probs(current_year)
            print('Starting simulate walks')
            sys.stdout.flush()
            walks = G.simulate_walks(args.num_walks, args.walk_length)
            learn_embeddings(walks)


if __name__ == "__main__":
    args = parse_args()
    main(args)
