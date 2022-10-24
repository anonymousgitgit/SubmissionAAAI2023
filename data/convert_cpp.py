from tree_sitter import Language, Parser
from pathlib import Path
# from ast_util import ASTUtil
from copy import deepcopy
import networkx as nx
import matplotlib.pyplot as plt
from collections import namedtuple
from networkx.drawing.nx_agraph import graphviz_layout
from collections import defaultdict

# Parse source code to get AST
# Remove delimiter nodes
# Convert AST into a json-based tree

def convert_into_json_tree(tree, text):
    root = tree.root_node

    ignore_types = ["\n", '[', ']', '{', '}', '(', ')', ';', ',', '', ':', 'comment']
    num_nodes = 0
    root_type = str(root.type)
    queue = [root]

    root_json = {
        "node_type": root_type,
        "node_token": "", # usually root does not contain token
        "children": [],
    }

    queue_json = [root_json]
    while queue:
        
        current_node = queue.pop(0)
        current_node_json = queue_json.pop(0)
        num_nodes += 1
        children = current_node.children
        # print(children)
        
        if current_node.type == 'for_statement':
            if children[1].type == '(' and children[2].type == 'declaration': # bugs of tree-sitter
                children = children[:3] + [namedtuple('child', ['type', 'children', 'start_byte', 'end_byte'])(';', [], 0, 0)] + children[3:]
            if children[1].type == '(' and children[2].type == ';':
                _child = namedtuple('child', ['type', 'children', 'start_byte', 'end_byte'])('missing_assignment_expression', [], 0, 0)
                children = children[:2] + [_child] + children[2:]
            if children[3].type == ';' and children[4].type == ';':
                _child = namedtuple('child', ['type', 'children', 'start_byte', 'end_byte'])('missing_binary_expression', [], 0, 0)
                children = children[:4] + [_child] + children[4:]      
            if children[5].type == ';' and children[6].type == ')':
                _child = namedtuple('child', ['type', 'children', 'start_byte', 'end_byte'])('missing_update_expression', [], 0, 0)
                children = children[:6] + [_child] + children[6:]            
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
                    "children": [],
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

        if current_type == 'for_statement' and len(children) != 5:
            # print(current_node)
            raise ValueError("Error: for statement ")


        if current_type in ['if_statement', 'while_statement', 'for_statement', 'do_statement', 'break_statement', 'continue_statement', 'switch_statement', 'struct_specifier', 'template_declaration']:
            skip_indices = [0]
            current_token = current_type.split('_')[0]

        elif current_type in ['binary_expression', 'assignment_expression', 'init_declarator']: 
            skip_indices = [1]
            current_token = children[1]['node_token']
        elif current_type == 'unary_expression': 
            skip_indices = [0]
            current_token = children[0]['node_token']
        else: skip_indices = []
        
        current_node_json['node_type'] = current_type
        current_node_json['node_token'] = current_token
        current_node_json['children'] = []
        
        for i, child in enumerate(children):
            if i in skip_indices: continue
            
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

            if len(child['children']) == 1 and child['node_type'] in ['expression_statement', 'parenthesized_expression', 'condition_clause']:
                _child = child['children'][0]
                while len(_child['children']) == 1 and _child['node_type'] in ['expression_statement', 'parenthesized_expression', 'condition_clause']:
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
    atomic_stmt_types = ['preproc_function_def', 'type_definition', 'goto_statement', 'pointer_expression', 'function_definition', 'declaration', 'cast_expression', 'conditional_expression', 'assignment_expression', 'call_expression', 'binary_expression', 'unary_expression', 'comma_expression', 'update_expression', 'return_statement', 'break_statement', 'continue_statement', 'identifier', 'subscript_expression', 'field_expression']
    _node_index = 0
    def fn(root, parent):
        nonlocal _node_index
        current_type = root['node_type']
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
                'node_index': root['node_index'],
            }
            stmt = [function_header]
        else: 
            stmt = []
        if current_type == 'template_declaration':
            try:
                _token = children[1]['children'][0]['node_token']
            except: _token = ''
            dt = {
                    'node_type': 'template_declaration',
                    'node_token': root['node_token'],
                    'children': [
                        children[0],
                        {
                            'node_type': 'function_definition',
                            'node_token': _token,
                            'children': []
                        }
                    ],
                    'node_index': root['node_index'],
            }
            stmt = [dt]
            root['children'] = children
            for child in children:
                stmt.extend(fn(child, root))
        if current_type == 'function_definition':
            dt = {
                    'node_type': 'function_definition',
                    'node_token': root['node_token'],
                    'children': [],
                    'node_index': root['node_index'],
            }
            for subchild in children:
                if subchild['node_type'] == 'function_declarator':
                    dt['children'].append(subchild.copy())
                    break
                elif subchild['node_type'] == 'storage_class_specifier':
                    dt['children'].append(subchild.copy())
                elif subchild['node_type'] == 'type_identifier':
                    dt['node_token'] = subchild['node_token']

            stmt = [dt]
            for child in children:
                stmt.extend(fn(child, root))
        
        elif current_type in atomic_stmt_types:
            if current_type in ['identifier', 'subscript_expression', 'field_expression']: # a variable is used as an expression
                if parent['node_type'] in ['if_statement', 'while_statement', 'for_statement', 'do_statement', 'switch_statement', 'function_definition']: # only accept in control statement
                    stmt = [deepcopy(root)]
                    root['children'] = []
                else:
                    for child in children:
                        stmt.extend(fn(child, root))
            else:
                stmt = [deepcopy(root)]
                root['children'] = []
        else:
            for child in children:
                stmt.extend(fn(child, root))
        return stmt
    stmts = fn(refactor_json, None)
    return stmts

# build graph

def get_key_from_metadata(parent_index, child_index, etype, stmt_indices):
    edge_index = ['ast_edge', 'control_flow_edge', 'next_stmt_edge', 'data_flow_edge'].index(etype)
    is_parent_stmt = int(parent_index in stmt_indices)
    is_child_stmt = int(child_index in stmt_indices)
    return f'{is_parent_stmt}{is_child_stmt}{edge_index}'



def bfs_build_graph(root, stmts, stmt_indices, has_cdg = True):
    stmt_types =  ['preproc_function_def', 'type_definition', 'declaration', 'pointer_expression', 'goto_statement', 'assignment_expression', 'call_expression', 'binary_expression', 'unary_expression', 'comma_expression', 'update_expression', 'return_statement', 'if_statement', 'while_statement', 'for_statement', 'do_statement', 'switch_statement']
    G = nx.MultiDiGraph()

    # add all the nodes
    # add AST-type edges
    G.add_node(root['node_index'], node_type = root['node_type'])
    queue = [root]
    
    pre_sibling_index = 0
    for child in root['children']:
        if child['node_type'] in ['declaration', 'function_definition']:
            child_index = child['node_index']
            if has_cdg:
                key = get_key_from_metadata(pre_sibling_index, child_index, 'control_flow_edge', stmt_indices)
                G.add_edge(pre_sibling_index, child_index, key = key, color = 'r', edge_type = 'control_flow_edge')
            pre_sibling_index = child_index

    while queue:
        current_node = queue.pop(0)
        node_index = current_node['node_index']
        children = current_node['children']
        # print(current_node)
        for child in children:
            child_index = child['node_index']
            if child_index != -1:
                # G.add_node(child_index, node_type = child['node_type'])
                key = get_key_from_metadata(node_index, child_index, 'ast_edge', stmt_indices)
                G.add_edge(node_index, child_index, key = key, color = 'g', edge_type = 'ast_edge')

            queue.append(child)
        # continue
        # connect consecutive statements
        if not has_cdg: continue
        if current_node['node_type'] == 'function_definition':
            pre_sibling_index = node_index
            for child in children:
                if child['node_type'] in stmt_types:
                    child_index = child['node_index']
                    key = get_key_from_metadata(pre_sibling_index, child_index, 'control_flow_edge', stmt_indices)
                    G.add_edge(pre_sibling_index, child_index, key = key, color = 'r', edge_type = 'control_flow_edge')
                    pre_sibling_index = child_index

        if current_node['node_type'] == 'do_statement':
            for i, child in enumerate(children):
                if child['node_type'] == 'while': break
                key = get_key_from_metadata(node_index, child['node_index'], 'control_flow_edge', stmt_indices)
                G.add_edge(node_index, child['node_index'], key = key, color = 'r', edge_type = 'control_flow_edge')
            if i:
                key = get_key_from_metadata(children[i - 1]['node_index'], children[i + 1]['node_index'], 'control_flow_edge', stmt_indices)
                G.add_edge(children[i - 1]['node_index'], children[i + 1]['node_index'], key = key, color = 'r', edge_type = 'control_flow_edge')
            else:
                key = get_key_from_metadata(node_index, children[i + 1]['node_index'], 'control_flow_edge', stmt_indices)
                G.add_edge(node_index, children[i + 1]['node_index'], key = key, color = 'r', edge_type = 'control_flow_edge')
            key = get_key_from_metadata(children[i + 1]['node_index'], node_index, 'control_flow_edge', stmt_indices)
            G.add_edge(children[i + 1]['node_index'], node_index, key = key, color = 'r', edge_type = 'control_flow_edge')
        elif current_node['node_type'] == 'if_statement':
            key = get_key_from_metadata(node_index, children[0]['node_index'], 'control_flow_edge', stmt_indices)
            G.add_edge(node_index, children[0]['node_index'], key = key, color = 'r', edge_type = 'control_flow_edge')
            pre_src_index = children[0]['node_index']
            for child in children[1:]:
                if child['node_type'] == 'else':
                    key = get_key_from_metadata(node_index, child['node_index'], 'control_flow_edge', stmt_indices)
                    G.add_edge(node_index, child['node_index'], key = key, color = 'r', edge_type = 'control_flow_edge')
                    pre_src_index = child['node_index']
                    continue
                key = get_key_from_metadata(pre_src_index, child['node_index'], 'control_flow_edge', stmt_indices)
                G.add_edge(pre_src_index, child['node_index'], key = key, color = 'r', edge_type = 'control_flow_edge')
        elif current_node['node_type'] == 'while_statement':
            key = get_key_from_metadata(node_index, children[0]['node_index'], 'control_flow_edge', stmt_indices)
            G.add_edge(node_index, children[0]['node_index'], key = key, color = 'r', edge_type = 'control_flow_edge')
            for child in children[1:]:
                key = get_key_from_metadata(children[0]['node_index'], child['node_index'], 'control_flow_edge', stmt_indices)
                G.add_edge(children[0]['node_index'], child['node_index'], key = key, color = 'r', edge_type = 'control_flow_edge')
            key = get_key_from_metadata(children[-1]['node_index'], node_index, 'control_flow_edge', stmt_indices)
            G.add_edge(children[-1]['node_index'], node_index, key = key, color = 'r', edge_type = 'control_flow_edge')
        elif current_node['node_type'] == 'for_statement':
            for i in range(0, 2):
                if children[i]['node_index'] != -1:
                    key = get_key_from_metadata(node_index, children[i]['node_index'], 'control_flow_edge', stmt_indices)
                    G.add_edge(node_index, children[i]['node_index'], key = key, color = 'r', edge_type = 'control_flow_edge')
            condition_child_index = children[1]['node_index'] if children[1]['node_index'] != -1 else node_index
            for child in children[3:]:
                key = get_key_from_metadata(condition_child_index, child['node_index'], 'control_flow_edge', stmt_indices)
                G.add_edge(condition_child_index, child['node_index'], key = key, color = 'r', edge_type = 'control_flow_edge')
            if children[2]['node_index'] != -1: # update expression
                key = get_key_from_metadata(children[-1]['node_index'], children[2]['node_index'], 'control_flow_edge', stmt_indices)
                G.add_edge(children[-1]['node_index'], children[2]['node_index'], key = key, color = 'r', edge_type = 'control_flow_edge')
                key = get_key_from_metadata(children[2]['node_index'], node_index, 'control_flow_edge', stmt_indices)
                G.add_edge(children[2]['node_index'], node_index, key = key, color = 'r', edge_type = 'control_flow_edge')
            elif children[-1]['node_index'] != -1: # except empty block
                key = get_key_from_metadata(children[-1]['node_index'], node_index, 'control_flow_edge', stmt_indices)
                G.add_edge(children[-1]['node_index'], node_index, key = key, color = 'r', edge_type = 'control_flow_edge')
        elif current_node['node_type'] == 'switch_statement':
            key = get_key_from_metadata(node_index, children[0]['node_index'], 'control_flow_edge', stmt_indices)
            G.add_edge(node_index, children[0]['node_index'], key = key, color = 'r', edge_type = 'control_flow_edge')
            pre_stmt_case = None
            for child in children[1:]:
                key = get_key_from_metadata(children[0]['node_index'], child['node_index'], 'control_flow_edge', stmt_indices)
                G.add_edge(children[0]['node_index'], child['node_index'], key = key, color = 'r', edge_type = 'control_flow_edge')
                if pre_stmt_case is not None:
                    key = get_key_from_metadata(pre_stmt_case, child['node_index'], 'control_flow_edge', stmt_indices)
                    G.add_edge(pre_stmt_case, child['node_index'], key = key, color = 'r', edge_type = 'control_flow_edge')
                for subchild in child['children']:
                    if subchild['node_type'] in (stmt_types + ['break_statement']):
                        key = get_key_from_metadata(child['node_index'], subchild['node_index'], 'control_flow_edge', stmt_indices)
                        G.add_edge(child['node_index'], subchild['node_index'], key = key, color = 'r', edge_type = 'control_flow_edge')
                        pre_stmt_case = subchild['node_index']
        elif current_node['node_type'] == 'compound_statement':
            for child in children:
                key = get_key_from_metadata(node_index, child['node_index'], 'control_flow_edge', stmt_indices)
                G.add_edge(node_index, child['node_index'], key = key, color = 'r', edge_type = 'control_flow_edge')

    # add next-statement-type edges
    # for stmt, next_stmt in zip(stmts[:-1], stmts[1:]):
    #     key = get_key_from_metadata(stmt['node_index'], next_stmt['node_index'], 'next_stmt_edge', stmt_indices)
    #     G.add_edge(stmt['node_index'], next_stmt['node_index'], key = key, color = 'b', edge_type = 'next_stmt_edge')

    return G

def add_edges_brkcnt(root, loop_stmt_path, G, stmt_indices):
    children = root['children']
    flag = len(loop_stmt_path) > 0
    is_switch_stmt = False
    if flag:
        loop_stmt, _ = loop_stmt_path[0]
        is_switch_stmt = loop_stmt_path[-1][0]['node_type'] == 'switch_statement'
   
    brkcnt_node_idx = None
    is_root_loop = root['node_type'] in ['while_statement', 'for_statement', 'do_statement']
    is_in_do_while = False
    for i, child in enumerate(children):
        if flag and child['node_type'] in ['break_statement', 'continue_statement']:
            if child['node_type'] == 'break_statement' and not is_switch_stmt:
                if loop_stmt['node_type'] == 'while_statement':
                    key = get_key_from_metadata(child['node_index'], loop_stmt['node_index'], 'control_flow_edge', stmt_indices)
                    G.add_edge(child['node_index'], loop_stmt['node_index'], key = key, color = 'r', edge_type = 'control_flow_edge')
                elif loop_stmt['node_type'] == 'for_statement':
                    if loop_stmt['children'][2]['node_index'] != -1:
                        idx = loop_stmt['children'][2]['node_index']
                    else:
                        idx = loop_stmt['node_index']
                    key = get_key_from_metadata(child['node_index'], idx, 'control_flow_edge', stmt_indices)
                    G.add_edge(child['node_index'], idx, key = key, color = 'r', edge_type = 'control_flow_edge')
                elif loop_stmt['node_type'] == 'do_statement':
                    key = get_key_from_metadata(child['node_index'], loop_stmt['children'][-1]['node_index'], 'control_flow_edge', stmt_indices)
                    G.add_edge(child['node_index'], loop_stmt['children'][-1]['node_index'], key = key, color = 'r', edge_type = 'control_flow_edge')
            for parent, local_index in loop_stmt_path:
                for _child in parent['children'][local_index + 1:]:
                    if _child['node_type'] in ['else', 'while']: break
                    key = get_key_from_metadata(child['node_index'], _child['node_index'], 'control_flow_edge', stmt_indices)
                    G.add_edge(child['node_index'], _child['node_index'], key = key, color = 'r', edge_type = 'control_flow_edge')
        elif not flag and child['node_type'] in ['break_statement', 'continue_statement'] and is_root_loop:
            if brkcnt_node_idx is None: brkcnt_node_idx = child['node_index']
            if child['node_type'] == 'break_statement' and not is_switch_stmt:
                if root['node_type'] == 'do_statement':
                    key = get_key_from_metadata(child['node_index'], children[-1]['node_index'], 'control_flow_edge', stmt_indices)
                    G.add_edge(child['node_index'], children[-1]['node_index'], key = key, color = 'r', edge_type = 'control_flow_edge')
                elif root['node_type'] == 'for_statement':
                    if children[2]['node_index'] != -1:
                        idx = children[2]['node_index']
                    else:
                        idx = root['node_index']
                    key = get_key_from_metadata(child['node_index'], idx, 'control_flow_edge', stmt_indices)
                    G.add_edge(child['node_index'], idx, key = key, color = 'r', edge_type = 'control_flow_edge')
                else:
                    key = get_key_from_metadata(child['node_index'], root['node_index'], 'control_flow_edge', stmt_indices)
                    G.add_edge(child['node_index'], root['node_index'], key = key, color = 'r', edge_type = 'control_flow_edge')
        elif brkcnt_node_idx is not None:
            is_in_do_while = is_in_do_while or child['node_type'] == 'while'
            if not is_in_do_while:
                key = get_key_from_metadata(brkcnt_node_idx, child['node_index'], 'control_flow_edge', stmt_indices)
                G.add_edge(brkcnt_node_idx, child['node_index'], key = key, color = 'r', edge_type = 'control_flow_edge')

        if is_root_loop:
            new_path = [(root, i)]
        elif flag:
            new_path = loop_stmt_path + [(root, i)]
        else:
            new_path = loop_stmt_path
        add_edges_brkcnt(child, new_path, G, stmt_indices)

def add_data_flow_edges(root, G, stmt_indices, stmt_id2names, stmt_id2stmts, scope, *, brk_scope = None, con_stmt_id = [-1]):
    node_index = root['node_index']
    is_stmt = node_index in stmt_indices
    children = root['children']
    
    if is_stmt:
        for env_name, env_stmt_ids in scope.items():
            names = stmt_id2names[node_index]
            if root['node_type'] == 'assignment_expression':
                names = names[1:]
            elif root['node_type'] == 'declaration':
                names = []
            for name in names:
                if env_name == name:
                    for env_stmt_id in set(env_stmt_ids):                        
                        if brk_scope is None:
                            key = get_key_from_metadata(env_stmt_id, node_index, 'data_flow_edge', stmt_indices)
                            G.add_edge(env_stmt_id, node_index, key = key, color = 'y', edge_type = 'data_flow_edge')
                        elif name in brk_scope and env_stmt_id in brk_scope[name] and (env_stmt_id > node_index or node_index in con_stmt_id): continue
                        elif env_stmt_id == node_index: continue
                        else:
                            key = get_key_from_metadata(env_stmt_id, node_index, 'data_flow_edge', stmt_indices)
                            G.add_edge(env_stmt_id, node_index, key = key, color = 'y', edge_type = 'data_flow_edge')


        current_scope = scope
        if root['node_type'] == 'assignment_expression' and len(stmt_id2names[node_index]):
            # for name in stmt_id2names[node_index]:
            #     current_scope[name] = [node_index]
            name = stmt_id2names[node_index][0]
            current_scope[name] = [node_index]
        elif root['node_type'] == 'update_expression':
            for name in stmt_id2names[node_index]:
                current_scope[name] = [node_index]
        elif root['node_type'] == 'function_definition':
            for name in stmt_id2names[node_index]:
                current_scope[name] = [node_index]
        elif root['node_type'] == 'declaration':
            _children = stmt_id2stmts[node_index]['children']
            for _child in _children:
                if _child['node_type'] == 'identifier':
                    current_scope[_child['node_token']] = [node_index]
                elif _child['node_type'] == 'init_declarator':
                    names = get_identifiers_from_stmt(_child)
                    for env_name, env_stmt_ids in scope.items():
                        for name in names[1:]:
                            if env_name == name:
                                for env_stmt_id in set(env_stmt_ids):
                                    key = get_key_from_metadata(env_stmt_id, node_index, 'data_flow_edge', stmt_indices)
                                    G.add_edge(env_stmt_id, node_index, key = key, color = 'y', edge_type = 'data_flow_edge')
                    if len(names):
                        name = names[0]
                        current_scope[name] = [node_index]
            
    else:
        current_scope = scope
    inner_scope = current_scope.copy()
    if root['node_type'] == 'for_statement':
        children = children[:2] + children[3:] + children[2:0:-1]
    elif root['node_type'] == 'while_statement':
        children = children + children[0:1]

    if root['node_type'] == 'if_statement':
        last_inner_scope = {}
        first_local_vars = set()
        else_local_vars = set()
        flag = True
        origin_inner_scope = inner_scope.copy()
        for i, child in enumerate(children):
            if i == 0:
                if child['node_type'] == 'update_expression':
                    inner_scope = add_data_flow_edges(child, G, stmt_indices, stmt_id2names, stmt_id2stmts, inner_scope, brk_scope = brk_scope, con_stmt_id = con_stmt_id)
                else:
                    flag = False
                    if child['node_index'] in stmt_id2stmts:
                        _children = stmt_id2stmts[child['node_index']]['children']
                        for _child in _children:
                            if _child['node_type'] == 'assignment_expression':
                                name = get_identifiers_from_stmt(_child)[0]
                                inner_scope[name] = [child['node_index']]
                                flag = True
                                inner_scope = add_data_flow_edges(child, G, stmt_indices, stmt_id2names, stmt_id2stmts, inner_scope, brk_scope = brk_scope, con_stmt_id = con_stmt_id)
                            elif _child['node_type'] == 'update_expression':
                                flag = True
                                names = get_identifiers_from_stmt(_child)
                                for name in names:
                                    inner_scope[name] = [child['node_index']]
                                inner_scope = add_data_flow_edges(child, G, stmt_indices, stmt_id2names, stmt_id2stmts, inner_scope, brk_scope = brk_scope, con_stmt_id = con_stmt_id)
                    if not flag:
                        inner_scope = add_data_flow_edges(child, G, stmt_indices, stmt_id2names, stmt_id2stmts, inner_scope, brk_scope = brk_scope, con_stmt_id = con_stmt_id)

                first_inner_scope = inner_scope.copy()
            else:
                inner_scope = add_data_flow_edges(child, G, stmt_indices, stmt_id2names, stmt_id2stmts, inner_scope, brk_scope = brk_scope, con_stmt_id = con_stmt_id)
            if child['node_type'] == 'else':
                last_inner_scope = inner_scope
                inner_scope = first_inner_scope.copy()
                flag = False

            if child['node_type'] == 'declaration':
                if flag:
                    first_local_vars.update(stmt_id2names[child['node_index']])
                else:
                    else_local_vars.update(stmt_id2names[child['node_index']])
        if flag:
            last_inner_scope = inner_scope
            inner_scope = first_inner_scope.copy()
    elif root['node_type'] == 'for_statement':
        local_vars = set()
        first_local_vars = set()
        flag = True
        brk_scope = defaultdict(list)
        cnt_scope = defaultdict(list)
        add_brkcnt_data_flow(root, G, stmt_indices, stmt_id2names, stmt_id2stmts, brk_scope, cnt_scope, [])
        brk_scope = dict(brk_scope)
        cnt_scope = dict(cnt_scope)
        for i, child in enumerate(children):
            inner_scope = add_data_flow_edges(child, G, stmt_indices, stmt_id2names, stmt_id2stmts, inner_scope, brk_scope = brk_scope, con_stmt_id = con_stmt_id)
            if i == 1: #after conditional expression
                first_inner_scope = inner_scope.copy()
                flag = False
            if child['node_type'] == 'declaration':
                local_vars.update(stmt_id2names[child['node_index']])
                if flag:
                    first_local_vars.update(stmt_id2names[child['node_index']])
        for i, child in enumerate(children[1:]):
            inner_scope = add_data_flow_edges(child, G, stmt_indices, stmt_id2names, stmt_id2stmts, inner_scope, brk_scope = brk_scope, con_stmt_id = con_stmt_id)
        
        
    elif root['node_type'] == 'while_statement':
        local_vars = set()

        brk_scope = defaultdict(list)
        cnt_scope = defaultdict(list)
        add_brkcnt_data_flow(root, G, stmt_indices, stmt_id2names, stmt_id2stmts, brk_scope, cnt_scope, [])
        brk_scope = dict(brk_scope)
        cnt_scope = dict(cnt_scope)
        for i, child in enumerate(children):
            if i == 0:
                if child['node_type'] == 'update_expression':
                    inner_scope = add_data_flow_edges(child, G, stmt_indices, stmt_id2names, stmt_id2stmts, inner_scope, brk_scope = brk_scope, con_stmt_id = con_stmt_id)
                else:
                    flag = False
                    if child['node_index'] in stmt_indices:
                        _children = stmt_id2stmts[child['node_index']]['children']
                        for _child in _children:
                            if _child['node_type'] == 'assignment_expression':
                                name = get_identifiers_from_stmt(_child)[0]
                                inner_scope[name] = [child['node_index']]
                                flag = True
                            elif _child['node_type'] == 'update_expression':
                                flag = True
                                names = get_identifiers_from_stmt(_child)
                                for name in names:
                                    inner_scope[name] = [child['node_index']]
                    if not flag:
                        inner_scope = add_data_flow_edges(child, G, stmt_indices, stmt_id2names, stmt_id2stmts, inner_scope, brk_scope = brk_scope, con_stmt_id = con_stmt_id)

                first_inner_scope = inner_scope.copy()
            else:
                inner_scope = add_data_flow_edges(child, G, stmt_indices, stmt_id2names, stmt_id2stmts, inner_scope, brk_scope = brk_scope, con_stmt_id = con_stmt_id)
            if child['node_type'] == 'declaration':
                local_vars.update(stmt_id2names[child['node_index']])
        for i, child in enumerate(children[1:]):
            inner_scope = add_data_flow_edges(child, G, stmt_indices, stmt_id2names, stmt_id2stmts, inner_scope, brk_scope = brk_scope, con_stmt_id = con_stmt_id)

       
    elif root['node_type'] == 'do_statement':
        local_vars = set()
        brk_scope = defaultdict(list)
        cnt_scope = defaultdict(list)
        add_brkcnt_data_flow(root, G, stmt_indices, stmt_id2names, stmt_id2stmts, brk_scope, cnt_scope, [])
        brk_scope = dict(brk_scope)
        cnt_scope = dict(cnt_scope)
        con_stmt_id = [children[-1]['node_index']]
        for i, child in enumerate(children):
            inner_scope = add_data_flow_edges(child, G, stmt_indices, stmt_id2names, stmt_id2stmts, inner_scope, brk_scope = brk_scope, con_stmt_id = con_stmt_id)
            if child['node_type'] == 'declaration':
                local_vars.update(stmt_id2names[child['node_index']])
        
        for i, child in enumerate(children):
            if child['node_type'] == 'while': break
            inner_scope = add_data_flow_edges(child, G, stmt_indices, stmt_id2names, stmt_id2stmts, inner_scope, brk_scope = brk_scope, con_stmt_id = con_stmt_id)
        
    elif root['node_type'] == 'case_statement':
        local_vars = set()
        is_brk_case = False
        is_default_case = children[0]['node_type'] == 'default'
        for child in children:
            inner_scope = add_data_flow_edges(child, G, stmt_indices, stmt_id2names, stmt_id2stmts, inner_scope, brk_scope = brk_scope, con_stmt_id = con_stmt_id)
            if child['node_type'] == 'declaration':
                local_vars.update(stmt_id2names[child['node_index']])
            elif child['node_type'] == 'break_statement':
                is_brk_case = True
                break
    elif root['node_type'] == 'switch_statement':
        all_inner_scopes = []
        last_case = False
        is_default_case = False
        default_case_index = None
        for i, child in enumerate(children):
            inner_scope = add_data_flow_edges(child, G, stmt_indices, stmt_id2names, stmt_id2stmts, inner_scope, brk_scope = brk_scope, con_stmt_id = con_stmt_id)
            if i == 0:
                first_inner_scope = inner_scope.copy()
                for k, v in first_inner_scope.items():
                    current_scope[k] = v
            
            if child['node_type'] == 'case_statement':
                flag = inner_scope.pop('default_case')
                if inner_scope.pop('brk_case'):
                    all_inner_scopes.append(inner_scope)
                    inner_scope = current_scope.copy()
                    last_case = i == len(children) - 1
                is_default_case = is_default_case or flag
                if is_default_case and default_case_index is None:
                    default_case_index = len(all_inner_scopes) - 1
        else:
            if not last_case:
                all_inner_scopes.append(inner_scope)


    else:
        for i, child in enumerate(children):
            inner_scope = add_data_flow_edges(child, G, stmt_indices, stmt_id2names, stmt_id2stmts, inner_scope, brk_scope = brk_scope, con_stmt_id = con_stmt_id)


    if root['node_type'] == 'if_statement':
        for k, v in first_inner_scope.items():
            if k in current_scope:
                current_scope[k] = v
        for k, v in last_inner_scope.items():
            if k in first_local_vars: continue
            if k not in current_scope:
                current_scope[k] = v
            else:
                tmp = set(current_scope[k]) | set(v)
                current_scope[k] = list(tmp)
        for k, v in inner_scope.items():
            if k in else_local_vars: continue
            if k not in current_scope:
                current_scope[k] = v
            else:
                tmp = set(current_scope[k]) | set(v)
                current_scope[k] = list(tmp)
            # current_scope[k].extend(v)
    elif root['node_type'] == 'for_statement':
        for k, v in first_inner_scope.items():
            if k in current_scope and k not in first_local_vars:
                current_scope[k] = v
        for k, v in inner_scope.items():
            if k in local_vars: continue
            if k not in current_scope:
                current_scope[k] = v
            else:
                tmp = set(current_scope[k]) | set(v)
                current_scope[k] = list(tmp)
        for k, v in brk_scope.items():
            if k in current_scope:
                tmp = set(current_scope[k]) | set(v)
                current_scope[k] = list(tmp)
        for k, v in cnt_scope.items():
            if k in current_scope:
                tmp = set(current_scope[k]) | set(v)
                current_scope[k] = list(tmp)
    elif root['node_type'] == 'while_statement':
        for k, v in first_inner_scope.items():
            if k in current_scope:
                current_scope[k] = v
        for k, v in inner_scope.items():
            if k in local_vars: continue
            if k not in current_scope:
                current_scope[k] = v
            else:
                tmp = set(current_scope[k]) | set(v)
                current_scope[k] = list(tmp)
        for k, v in brk_scope.items():
            if k in current_scope:
                tmp = set(current_scope[k]) | set(v)
                current_scope[k] = list(tmp)
        for k, v in cnt_scope.items():
            if k in current_scope:
                tmp = set(current_scope[k]) | set(v)
                current_scope[k] = list(tmp)
    elif root['node_type'] == 'do_statement':
        for k, v in inner_scope.items():
            if k in local_vars: continue
            current_scope[k] = v
            # if k not in current_scope:
            #     current_scope[k] = v
            # else:
            #     current_scope[k].extend(v)
        for k, v in brk_scope.items():
            if k in current_scope:
                tmp = set(current_scope[k]) | set(v)
                current_scope[k] = list(tmp)
        for k, v in cnt_scope.items():
            if k in current_scope:
                tmp = set(current_scope[k]) | set(v)
                current_scope[k] = list(tmp)
    elif root['node_type'] == 'case_statement':
        for k, v in inner_scope.items():
            if k in local_vars: continue
            current_scope[k] = v
            # if k not in current_scope:
            #     current_scope[k] = v
            # else:
            #     current_scope[k].extend(v)
        current_scope['brk_case'] = is_brk_case
        current_scope['default_case'] = is_default_case
        
    elif root['node_type'] == 'switch_statement':
        if is_default_case:
            for k, v in all_inner_scopes[default_case_index].items():   
                try:
                    current_scope.pop(k)
                except: pass
        for _inner_scope in all_inner_scopes:
            for k, v in _inner_scope.items():
                if k not in current_scope:
                    current_scope[k] = v
                else:
                    tmp = set(current_scope[k]) | set(v)
                    current_scope[k] = list(tmp)
        
    return current_scope

def add_brkcnt_data_flow(root, G, stmt_indices, stmt_id2names, stmt_id2stmts, brK_scope, cnt_scope, loop_path):
    node_index = root['node_index']
    is_stmt = node_index in stmt_indices
    children = root['children']

    is_root_loop = root['node_type'] in ['while_statement', 'for_statement', 'do_statement']
    flag = len(loop_path) > 0
    is_switch_statement = loop_path[-1][0]['node_type'] == 'switch_statement' if flag else False
    # scope = scope.copy()
    for i, child in enumerate(children):
        if child['node_type'] == 'break_statement' and not is_switch_statement:
            local_vars = []
            for stmt, idx in loop_path + [(root, i - 1)]:
                if stmt['node_type'] == 'if_statement':
                    else_idx = None
                    for _idx, _child in enumerate(stmt['children']):
                        if _child['node_type'] == 'else':
                            else_idx = _idx
                            break
                    if else_idx is None or else_idx > idx:
                        s_idx = 0      
                    else:
                        s_idx = else_idx
                else:
                    s_idx = 0
                for _child in stmt['children'][s_idx:idx + 1]:
                    _child_idx = _child['node_index']
                    if _child_idx in stmt_indices:
                        # if _child['node_type'] in ['declaration', 'update_expression']:
                        #     for name in stmt_id2names[_child_idx]:
                        #         scope[name] = [_child_idx]
                        if _child['node_type'] == 'assignment_expression' and len(stmt_id2names[_child_idx]):
                            # for name in stmt_id2names[node_index]:
                            #     current_scope[name] = [node_index]
                            name = stmt_id2names[_child_idx][0]
                            if _child_idx not in brK_scope[name]:
                                brK_scope[name].append(_child_idx)
                        elif _child['node_type'] == 'update_expression':
                            for name in stmt_id2names[_child_idx]:
                                if _child_idx not in brK_scope[name]:
                                    brK_scope[name].append(_child_idx)
            #             else:
            #                 local_vars.extend(stmt_id2names[_child_idx])
            # for var in local_vars:
            #     if var in scope: scope.pop(var)
        elif child['node_type'] == 'continue_statement':
            _scope = {}
            for stmt, idx in loop_path + [(root, i - 1)]:
                if stmt['node_type'] == 'if_statement':
                    else_idx = None
                    for _child in stmt['children']:
                        if _child['node_type'] == 'else':
                            else_idx = _child['node_index']
                            break
                    if else_idx is None or else_idx > stmt['children'][idx]['node_index']:
                        s_idx = 0
                    else:
                        s_idx = else_idx
                else:
                    s_idx = 0
                for _child in stmt['children'][s_idx:idx + 1]:
                    _child_idx = _child['node_index']
                    if _child_idx in stmt_indices:
                        if _child['node_type'] == 'assignment_expression' and stmt_id2names[_child_idx]:
                            # for name in stmt_id2names[node_index]:
                            #     current_scope[name] = [node_index]
                            name = stmt_id2names[_child_idx][0]
                            _scope[name] = [_child_idx]
                        elif _child['node_type'] == 'update_expression':
                            for name in stmt_id2names[_child_idx]:
                                _scope[name] = [_child_idx]
                        
            exit_outer_loop = False
            for stmt, idx in loop_path + [(root, i - 1)]:
                s_index = 0
                if stmt['node_type'] in ['while_statement', 'for_statement', 'do_statement']:
                    children = stmt['children']
                    if stmt['node_type'] == 'for_statement':
                        s_index = 1
                        con_exp, up_exp = children[1:3]
                        children[1], children[2] = up_exp, con_exp
                    elif stmt['node_type'] == 'do_statement':
                        idx += 1
                        children = [children[-1]] + children
                else:
                    children = stmt['children']
                for _child in stmt['children'][s_index:idx + 1]:
                    if _child['node_type'] in ['assignment_expression', 'update_expression', 'declaration']: 
                        exit_outer_loop = True
                        break
                    _child_idx = _child['node_index']
                    if _child_idx in stmt_indices:
                        for name in stmt_id2names[_child_idx]:
                            if name in _scope:
                                _idx = _scope[name][0]
                                if _idx != _child_idx or (len(root['children']) > 1 and root['children'][i - 1]['node_type'] == _child['node_type'] == 'call_expression'):
                                    key = get_key_from_metadata(_idx, _child_idx, 'data_flow_edge', stmt_indices)
                                    G.add_edge(_idx, _child_idx, key = key, color = 'y', edge_type = 'data_flow_edge')
                if exit_outer_loop: break
            cnt_scope.update(_scope)
        
        if is_root_loop:
            new_loop_path = [(root, i)]
        elif flag:
            new_loop_path = loop_path + [(root, i)]
        else:
            new_loop_path = loop_path
        add_brkcnt_data_flow(child, G, stmt_indices, stmt_id2names, stmt_id2stmts, brK_scope, cnt_scope, new_loop_path)





def get_identifiers_from_stmt(stmt_json):
    queue = [stmt_json]
    identifiers = []
    while queue:
        node = queue.pop(0)
        if node['node_type'] in ['identifier', 'function_definition'] and node['node_token'] not in identifiers:
            if node['node_token'] not in {'cin', 'cout', 'en'}:
                identifiers.append(node['node_token'])
        queue.extend(node['children'])
    return identifiers

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

def convert_into_cpp_graph(tree, text, has_data_flow = True, has_cdg = True):
    root_json = convert_into_json_tree(tree, text)
    refactor_json = remove_redundant_nodes(root_json)
    # print(refactor_json)
    # print('-------------------------------------------')
    
    _stmts = dfs_split_stmt(refactor_json)
    # print(refactor_json)
    # print('-------------------------------------------')
    # print(text)
    existing_ids = set()
    stmts = []
    for _stmt in _stmts:
        _idx = _stmt['node_index']
        if _idx not in existing_ids:
            stmts.append(_stmt)
            existing_ids.update({_idx})
    stmt_indices = list(map(lambda x: x['node_index'], stmts))
    stmt_id2names = {stmt['node_index']: get_identifiers_from_stmt(stmt) for stmt in stmts}
    stmt_id2stmts = {stmt['node_index']: stmt for stmt in stmts}

    # print(refactor_json)
    G = bfs_build_graph(refactor_json, stmts, stmt_indices, has_cdg = has_cdg)
    if has_cdg:
        add_edges_brkcnt(refactor_json, [], G, stmt_indices)
    if has_data_flow:
        add_data_flow_edges(refactor_json, G, stmt_indices, stmt_id2names, stmt_id2stmts, {})
    # stmt_ids = list(map(lambda x: x['node_index'], stmts))
    ast_nodes = get_node_info_from_index(refactor_json, stmt_indices)
    # print(refactor_json)
    return refactor_json, ast_nodes, stmts, stmt_indices, G
# root_json = convert_into_json_tree(tree)
# refactor_json = remove_redundant_nodes(root_json)
# stmts = dfs_split_stmt(refactor_json)



if __name__ == "__main__":
    parser = Parser()
    # parser.set_language(Language('language/cpp.so', 'cpp'))
    parser.set_language(Language('language/c.so', 'c'))

    import glob
    # text = "\tpublic int getPushesLowerbound() {\n\t\treturn pushesLowerbound;\n\t}\n"
    text = """

#include <bits/stdc++.h>

using namespace std;

typedef long long ll;
typedef unsigned long long ull;
typedef long double ld;
typedef pair<ll, ll> P;

#define EACH(i,a) for (auto& i : a)
#define FOR(i,a,b) for (ll i=(a);i<(b);i++)
#define RFOR(i,a,b) for (ll i=(b)-1;i>=(a);i--)
#define REP(i,n) for (ll i=0;i<(n);i++)
#define RREP(i,n) for (ll i=(n)-1;i>=0;i--)
#define debug(x) cout<<#x<<": "<<x<<endl
#define pb push_back
#define ALL(a) (a).begin(),(a).end()

const ll linf = 1e18;
const int inf = 1e9;
const double eps = 1e-12;
const double pi = acos(-1);

template<typename T>
istream& operator>>(istream& is, vector<T>& vec) {
	EACH(x,vec) is >> x;
	return is;
}
template<typename T>
ostream& operator<<(ostream& os, vector<T>& vec) {
	REP(i,vec.size()) {
		if (i) os << " ";
		os << vec[i];
	}
	return os;
}
template<typename T>
ostream& operator<<(ostream& os, vector< vector<T> >& vec) {
	REP(i,vec.size()) {
		if (i) os << endl;
		os << vec[i];
	}
	return os;
}

int main() {
	std::ios::sync_with_stdio(false);
	std::cin.tie(0);
	FOR(i, 1, 10) FOR(j, 1, 10) {
		cout << i << "x" << j << "=" << i*j << endl;
	}
}
    """

    # with open('code.txt') as f:
    #     text = f.read().strip()

    tree = parser.parse(text.encode())
    def read_file(path):
        with open(path) as f:
            lines = f.readlines()
        return ''.join(lines)
    from tqdm import tqdm


    # train_paths = glob.glob('OJ_raw_train_test_val/test/*/*.c')
    # # # print(train_paths[-1])
    # black_lst = ['OJ_raw_train_test_val/train/10/696.c']
    # train_paths = list(filter(lambda x: x not in black_lst, train_paths))
    
    # for path in tqdm(train_paths):
    #     text = read_file(path)
    # #     # print(path)
    #     convert_into_c_graph(parser.parse(text.encode()), text)
    #     try:
    #         convert_into_c_graph(parser.parse(text.encode()), text)
    #     except:
    #         print(path)

    root_json, ast_nodes, stmts, stmt_indices, G = convert_into_cpp_graph(tree, text)
    print(G.number_of_nodes(), len(ast_nodes), len(stmts))
    # # print(G.edges(data = 'edge_type'))
    
    # print(root_json)

    # # print(ast_nodes)
    # # import dgl
    # # g = dgl.from_networkx(G)
    # # print(g.edges())
    print('----------------------------------------------------------------------------------------')
    print(root_json)  
    print('----------------------------------------------------------------------------------------')
    for stmt in stmts:
        print(stmt['node_type'], stmt, get_identifiers_from_stmt(stmt))
        print('----------------------------------------------------------------------------------------')



    edges = G.edges()
    print('edges:', len(edges), len(G.edges(keys = True)), len(G.edges.data()))
    print(G.number_of_nodes(), list(G.nodes()))

    colors = []
    print(G.edges(data = 'edge_type', keys = True))
    # print(G.edges.data())
    for (u,v,attrib_dict) in list(G.edges.data()):
        colors.append(attrib_dict['color'])
    pos = graphviz_layout(G, prog='dot')
    nx.draw(G, pos = pos, edgelist=list(edges), edge_color=colors, with_labels = True)
    plt.show()