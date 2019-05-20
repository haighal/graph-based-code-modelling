import json
import argparse
import networkx as nx
from networkx.algorithms.traversal.depth_first_search import dfs_tree
import os
import random

'''
**** MSR Dataset Format ****

ContextGraph
    Edges
        Child - each of these is an list of [src, dst] node IDs
        LastLexicalUse
        LastUse
        LastWrite
        NextToken
    NodeLabels - dict from node ID (as a string) to name (either the node type or the name of the variable): "We label syntax nodes with the name of the nonterminal from the program’s grammar, whereas
                 syntax tokens are labeled with the string that they represent"
    NodeTypes - the variable type: we don't have that
SlotDummyNode - int corresponding to the index of the <SLOT> token in the vocabulary
SymbolCandidates -  list of dicts containing {"IsCorrect": true, "SymbolDummyNode": 1, "SymbolName": "parameter"}
filename - code file that the AST came from
slotTokenIdx - Index of token in the source file that is described by this problem instance

**** 150k Python Dataset Format ****

python100k_train.txt contains the names and corresponding GitHub repos
python100k_train.json contains the ASTs of the parsed files.  Each line is a json file.
    The orders are the same (the ith AST came from the ith file); ASTs are by file

Each graph is a list of (0-indexed) nodes, each represented as an object with several name/value pairs:
    (Required) type: string containing type of current AST node
    (Optional) value: string containing value (if any) of the current AST node
    (Optional) children: array of integers denoting indices of children (if any) of the current AST node. Indices are 0-based starting from the first node in the JSON file

**** Code2Seq Dataset Format ****

Each row in the text file is an example.
Each example is a space-delimited list of fields, where:
    1) First field is the target label (i.e. the method name), internally delimited by the "|" character (for example: compare|ignore|case)
    2) Each of the following field are contexts, where each context is a path between terminal nodes in an AST.  Represented by three components separated by commas (","). 
       None of these components can include spaces nor commas.
        - start AST terminal (delimited by |), list of non-terminal AST nodes on the path (delimited by |), end AST terminal
        - e.g. my|key,StringExression|MethodCall|Name,get|value
'''

'''
Generates a NetworkX Graph from an AST in the format provided from the Python150k Dataset
'''
def gen_networkx_graph(ast):
    edge_list = []
    terminals = []
    G = nx.DiGraph()
    for nid, node in enumerate(ast):
        node_type = node['type']
        children = None if 'children' not in node else node['children']
        value = None if 'value' not in node else node['value']

        if children is None:
            terminals.append(nid)
            ## Label terminals with their value
            ## Do we need to check for special characters like `decorator_list`?
            node_value = value if (value is not None) else node_type
            G.add_node(nid, label=node_value, type=node_type)
        else:
            ## label syntax nodes with the name of the nonterminal from the program’s grammar
            G.add_node(nid, label=node_type, type=node_type)
            for child in children:
                edge_list += [(nid, child)]
    G.add_edges_from(edge_list)
    return G

## Adapted from Dylan's code + simplified a bit
def get_snippet(G, start_node, dmin, dmax):
    snippet = None
    node_id = start_node
    while snippet is None and node_id < G.number_of_nodes():
        hub_ego = dfs_tree(G, node_id)
        if hub_ego.number_of_nodes() > dmin and hub_ego.number_of_nodes()  < dmax:
            snippet = hub_ego
        node_id += 1
    return snippet, node_id

def generate_snippets(G, num_snippets = 10, dmin = 10, dmax=64):
    snippets = []
    start_node = 0

    num_generated = 0
    while num_generated < num_snippets:
        snippet, next_start_node = get_snippet(G, start_node, dmin, dmax)
        if snippet is None or next_start_node == G.number_of_nodes():
            break
        else:
            subgraph = G.subgraph(snippet).copy() ## Make a copy of the snippet
            
            ## Get all the possible variables that you could mask, only continue if there's actually a variable there
            nodes = list(subgraph.nodes(data=True))
            variable_names = list(filter(contains_name, nodes))
            
            if len(variable_names) > 0:
                snippets.append(subgraph) 
                num_generated += 1

            start_node = next_start_node
    # print(f'Generated {len(snippets)} snippets')
    return snippets

def contains_name(node):
    _, data = node
    return 'Name' in data['type']

## Choose node to mask based on all the ones containing names
def choose_var_to_mask(G):
    nodes = list(G.nodes(data=True))
    possible_slots = list(filter(contains_name, nodes))
    return random.choice(possible_slots)[0]

'''
ContextGraph
    Edges
        Child - each of these is an list of [src, dst] node IDs
        LastLexicalUse
        LastUse
        LastWrite
        NextToken
    NodeLabels - dict from node ID (as a string) to name (either the node type or the name of the variable): "We label syntax nodes with the name of the nonterminal from the program’s grammar, whereas
                 syntax tokens are labeled with the string that they represent"
    NodeTypes - the variable type: we don't have that
SlotDummyNode - int corresponding to the index of the <SLOT> token in the vocabulary
SymbolCandidates -  list of dicts containing {"IsCorrect": true, "SymbolDummyNode": 1, "SymbolName": "parameter"}
filename - code file that the AST came from
slotTokenIdx - Index of token in the source file that is described by this problem instance (not needed)
'''
def make_context_graph(G, filename):

    ## Skeleton ContextGraph schema, gets filled in during the later steps of the method
    context_graph = {
        'Edges' : {
            'Child' : [],
            'NextToken' : []
        },
        'NodeLabels' : {},
        'NodeTypes' : {}, ## We also don't have this, ignore for now/leaving blank unless further notice
    }

    ## Relabel nodes to start at index 0
    varMapping = {i:idx for idx, i in enumerate(G.nodes())}
    G_relabeled = nx.relabel_nodes(G, varMapping)
    
    ## Add child edges from the raw AST
    for node in G_relabeled.nodes(data=True):
        nid, data = node
        
        context_graph['NodeLabels'][str(nid)] = data['label']
        for child in G_relabeled.neighbors(nid):
            context_graph['Edges']['Child'].append([nid, child])

    ## Add NextToken nodes between consecutive terminals/Syntax Tokens: test if a node is a leaf,
    ## and chain all the leaves together by increasing node ID
    def is_leaf(nid):
        return G_relabeled.out_degree(nid) == 0

    terminals = list(filter(is_leaf, list(G_relabeled.nodes())))
    for idx, nid in enumerate(terminals[:-1]):
        context_graph['Edges']['NextToken'].append([terminals[idx-1], nid])

    ## Chooses a random terminal to be the mask, then masks it
    slot_node_id = choose_var_to_mask(G_relabeled)
    correct_var_name = context_graph['NodeLabels'][str(slot_node_id)]
    context_graph['NodeLabels'][str(slot_node_id)] = '<SLOT>'

    output = {
        'ContextGraph' : context_graph,
        'SlotDummyNode' : slot_node_id,
        'filename' : filename,
        'SymbolCandidates' : [{"IsCorrect": True, "SymbolName": correct_var_name}], ## This part is dumb but necessary for their schema
        'slotTokenIdx' : ''
    }

    return output

def create_varnaming_samples(ast, filename, num_snippets, dmin, dmax): 
    # print(f'Generating up to {num_snippets} Snippets for AST from file: {filename}')
    G = gen_networkx_graph(ast)
    snippets = generate_snippets(G, num_snippets, dmin, dmax)
    output_graphs = []
    for snippet in snippets:
        output_graphs += [make_context_graph(snippet, filename)]
    return output_graphs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graphs", dest="graphs",
                        help="path to file containing json asts", required=True)
    parser.add_argument("-f", "--filenames", dest="filenames",
                        help="path to file containing the list of source files per ast", metavar="FILE", required=True)
    parser.add_argument("-s", "--save_folder", dest="save_folder",
                        help="path to save file", metavar="FILE", required=True)
    parser.add_argument("--seed", dest='seed', type=int, default=0,
                        help="seed for random generator (for reproducibility across runs)")
    parser.add_argument("--num_snippets", dest='num_snippets', type=int, default=10,
                        help="number of snippets to generate per python file")
    parser.add_argument("--dmin", dest='dmin', type=int, default=10,
                        help="minimum sixe for AST subgraph to be considered a snippet")
    parser.add_argument("--dmax", dest='dmax', type=int, default=64,
                        help="minimum sixe for AST subgraph to be considered a snippet")
    parser.add_argument("--snippets_per_file", dest='file_size', type=int, default=5000)
    args = parser.parse_args()
    
    print(args)

    random.seed(args.seed)
    graph_file = open(args.graphs, 'r')
    filenames = open(args.filenames, 'r')

    
    num_files = 0
    current_file = []

    ## Need to avoid hardcoding
    outfile_base_name = f'{args.save_folder}/{os.path.basename(args.graphs)[-5]}'
    os.makedirs(outfile_base_name)

    num_iters = 50000 if '50k' in args.graphs else 100000
        
    for i in range(num_iters):
        if (i + 1) % 1000 == 0:
            print(f'Processed {i+1} ASTs')
        ast = json.loads(graph_file.readline())
        filename = filenames.readline()

        snippets = create_varnaming_samples(ast, filename, args.num_snippets, args.dmin, args.dmax)
        current_file.extend(snippets)
        if len(current_file) >= args.file_size:
            outfile = f'{outfile_base_name}/graphs{num_files}.json'
            with open(outfile, 'w+') as f:
                print(f'Saving {len(current_file)} snippets to {outfile}')
                json.dump(current_file, f)
                current_file = []
                num_files += 1

    filenames.close()
    graph_file.close()