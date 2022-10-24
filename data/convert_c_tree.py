from tree_sitter import Language, Parser
from pathlib import Path
# from ast_util import ASTUtil
from copy import deepcopy
import networkx as nx
import matplotlib.pyplot as plt
from collections import namedtuple
from networkx.drawing.nx_agraph import graphviz_layout


# Parse source code to get AST
# Remove delimiter nodes
# Convert AST into a json-based tree

def convert_into_json_tree(tree, text):
    root = tree.root_node

    ignore_types = ["\n", '[', ']', '{', '}', '(', ')', ';', ',', '', ':']
    num_nodes = 0
    root_type = str(root.type)
    queue = [root]

    root_json = {
        "node_type": root_type,
        "node_token": "", # usually root does not contain token
        "children": []
    }

    queue_json = [root_json]
    while queue:
        
        current_node = queue.pop(0)
        current_node_json = queue_json.pop(0)
        num_nodes += 1
        children = current_node.children
        # print(children)
        
        # if current_node.type == 'for_statement':
        #     if children[1].type == '(' and children[2].type == 'declaration': # bugs of tree-sitter
        #         children = children[:3] + [namedtuple('child', ['type', 'children', 'start_byte', 'end_byte'])(';', [], 0, 0)] + children[3:]
        #     if children[1].type == '(' and children[2].type == ';':
        #         _child = namedtuple('child', ['type', 'children', 'start_byte', 'end_byte'])('missing_assignment_expression', [], 0, 0)
        #         children = children[:2] + [_child] + children[2:]
        #     if children[3].type == ';' and children[4].type == ';':
        #         _child = namedtuple('child', ['type', 'children', 'start_byte', 'end_byte'])('missing_binary_expression', [], 0, 0)
        #         children = children[:4] + [_child] + children[4:]      
        #     if children[5].type == ';' and children[6].type == ')':
        #         _child = namedtuple('child', ['type', 'children', 'start_byte', 'end_byte'])('missing_update_expression', [], 0, 0)
        #         children = children[:6] + [_child] + children[6:]            
            # print(type(children), children)
        for child in children:
            child_type = str(child.type)
            # print(current_node.type, child_type)
            if child_type.strip() not in ignore_types:
                queue.append(child)

                child_token = ""
                has_child = len(child.children) > 0

                if not has_child:
                    child_token = text[child.start_byte:child.end_byte]
                # print(child_type)
                child_json = {
                    "node_type": child_type,
                    "node_token": child_token,
                    "children": []
                }

                current_node_json['children'].append(child_json)
                queue_json.append(child_json)
    return root_json

# refactor tree
# removing redundant nodes, e.g., expression_stmt, parentheses_exp, move binary_token to parent's node
# #
def remove_redundant_nodes(root_json):
    program_json = root_json
    refactor_json = {}
    queue = [program_json]
    queue_json = [refactor_json]

    while queue:
        current_node = queue.pop(0)
        current_node_json = queue_json.pop(0)
        

        children = current_node['children']
        current_type = current_node['node_type']
        current_token = current_node['node_token']

        # if current_type == 'for_statement' and len(children) != 5:
        #     raise ValueError("Error: for statement ")

        # if current_type in ['if_statement', 'while_statement', 'for_statement', 'do_statement', 'break_statement', 'continue_statement', 'switch_statement', 'struct_specifier']:
        #     skip_indices = [0]
        #     current_token = current_type.split('_')[0]
        # elif current_type in ['binary_expression', 'assignment_expression', 'init_declarator']:
        #     skip_indices = [1]
        #     current_token = children[1]['node_token']
        # elif current_type == 'unary_expression':
        #     skip_indices = [0]
        #     current_token = children[0]['node_token']
        # else: skip_indices = []
        
        current_node_json['node_type'] = current_type
        current_node_json['node_token'] = current_token
        current_node_json['children'] = []
        
        for i, child in enumerate(children):
            # if i in skip_indices: continue
            
            if child['node_type'] == 'compound_statement':
                child_children = child['children']
                # remove block node
                for child_child in child_children:
                    if len(child_child['children']) == 1 and child_child['node_type'] in ['expression_statement', 'parenthesized_expression']:
                        _child = child_child['children'][0]
                    else:
                        _child = child_child
                    current_node_json['children'].append(_child)
                    queue.append(_child)
                    queue_json.append(_child)
                continue

            if len(child['children']) == 1 and child['node_type'] in ['expression_statement', 'parenthesized_expression']:
                _child = child['children'][0]
                while len(_child['children']) == 1 and _child['node_type'] in ['expression_statement', 'parenthesized_expression']:
                    _child = _child['children'][0]
            else:
                _child = child
            if _child['node_type'] == 'function_definition':
                # _child['node_token'] = child['children'][1]['children'][0]['node_token']
                for subchild in child['children']:
                    if subchild['node_type'] == 'function_declarator':
                        _child['node_token'] = subchild['children'][0]['node_token']
                        break
            
            current_node_json['children'].append(_child)
            queue.append(_child)
            queue_json.append(_child)
    return refactor_json

# # split into the statement-level AST and individual statements
# # index the nodes

def dfs_split_stmt(refactor_json):
    atomic_stmt_types = ['function_definition', 'declaration', 'cast_expression', 'conditional_expression', 'assignment_expression', 'call_expression', 'binary_expression', 'unary_expression', 'comma_expression', 'update_expression', 'return_statement', 'break_statement', 'continue_statement', 'identifier', 'subscript_expression', 'field_expression']
    _node_index = 0
    def fn(root, parent):
        nonlocal _node_index
        current_type = root['node_type']
        root['is_stmt'] = False
        if 'missing_' in current_type:
            root['node_index'] = -1
            return list()
        children = root['children']
        if 'node_index' not in root:
            root['node_index'] = _node_index
            _node_index += 1
           
        
        if root['node_index'] == 0 and root['node_type'] == 'translation_unit':
            function_header = {
                'node_type': root['node_type'],
                'node_token': root['node_token'],
                'children': [],
                'node_index': root['node_index']
            }
            root['is_stmt'] = True
            stmt = [function_header]
        else: 
            stmt = []
        if current_type == 'function_definition':
            stmt = [{
                    'node_type': 'function_definition',
                    'node_token': root['node_token'],
                    'children': [],
                    'node_index': root['node_index']
                }]
            root['is_stmt'] = True
            for child in children:
                stmt.extend(fn(child, root))
        elif current_type in atomic_stmt_types:
            if current_type in ['identifier', 'subscript_expression', 'field_expression']: # a variable is used as an expression
                if parent['node_type'] in ['if_statement', 'while_statement', 'for_statement', 'do_statement', 'switch_statement', 'function_definition']: # only accept in control statement
                    stmt = [deepcopy(root)]
                    root['is_stmt'] = True
                    root['children'] = []
            else:
                stmt = [deepcopy(root)]
                root['is_stmt'] = True
                root['children'] = []
        else:
            for child in children:
                stmt.extend(fn(child, root))
        return stmt
    stmts = fn(refactor_json, None)
    return stmts

def build_tree(root):
    G = nx.DiGraph()
    queue = [root]
    G.add_node(root['node_index'])
    while queue:
        current_node = queue.pop(0)
        node_index = current_node['node_index']
        children = current_node['children']
        # print('------------------------------')
        # print(current_node)
        for child in children:
            # print(child)
            child_index = child['node_index']
            if child_index != -1:
                G.add_node(child_index)
                G.add_edge(child_index, node_index)
                queue.append(child)
    return G

def get_node_info_from_index(root, stmt_ids):
    ast_nodes = {}
    queue = [root]
    while queue:
        current_node = queue.pop(0)
        node_index = current_node['node_index']
        if node_index != -1 and node_index not in stmt_ids:
            ast_nodes[node_index] = {
                'node_token': current_node['node_token'],
                'node_type': current_node['node_type']
            }
        queue.extend(current_node['children'])
    return ast_nodes

def get_sequence_from_stmt(root_stmt):
    seq = []
    seq.append((root_stmt['node_type'], root_stmt['node_token']))
    for child in root_stmt['children']:
        seq.extend(get_sequence_from_stmt(child))
    return seq

def convert_into_tree(tree, text, tree_lstm = False):
    root_json = convert_into_json_tree(tree, text)
    refactor_json = remove_redundant_nodes(root_json)
    
    stmts = dfs_split_stmt(refactor_json)
    stmt_indices = list(map(lambda x: x['node_index'], stmts))
    stmt_seqs = list(map(get_sequence_from_stmt, stmts))
   

    # stmt_ids = list(map(lambda x: x['node_index'], stmts))
    ast_nodes = get_node_info_from_index(refactor_json, stmt_indices)
    # print(refactor_json)
    if tree_lstm:
        return refactor_json, ast_nodes, stmts, stmt_seqs, stmt_indices, build_tree(refactor_json)
    return refactor_json, ast_nodes, stmts, stmt_seqs, stmt_indices
# root_json = convert_into_json_tree(tree)
# refactor_json = remove_redundant_nodes(root_json)
# stmts = dfs_split_stmt(refactor_json)



