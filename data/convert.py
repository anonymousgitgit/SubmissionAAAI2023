import glob
from tree_sitter import Language, Parser
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import matplotlib.pyplot as plt

from data.convert_c import convert_into_graph

paths = glob.glob('OJ_raw_train_test_val/*/*/*.c')

parser = Parser()
parser.set_language(Language('.tree-sitter/bin/c.so', 'c'))
count = 0
for path in paths:
    with open(path) as f:
        lines = f.readlines()
    text = ''.join(lines)
    try:
        root_json, refactor_json, stmts, G = convert_into_graph(parser.parse(text.encode()), text)
    except: 
        print(path)
        count += 1
print(count)
# print(refactor_json)
# print('----------------------------------------------------------------------------------------')
# for stmt in stmts:
#     print(stmt['node_type'], stmt)
#     print('----------------------------------------------------------------------------------------')



# edges = G.edges()

# colors = []

# for (u,v,attrib_dict) in list(G.edges.data()):
#     colors.append(attrib_dict['color'])
# pos = graphviz_layout(G, prog='dot')
# nx.draw(G, pos=pos, edgelist=list(edges), edge_color=colors, with_labels = True)
# plt.show()

