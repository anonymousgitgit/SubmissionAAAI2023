### Implementation for a submission of AAAI 2023
***
All source code are written in Python. Besides Pytorch, we also use many other libraries such as DGL, scikit-learn, pandas, jsonlines
#### Code Hierarchy
Given a code snippet, to extract Code Hierarchy, we use the function *convert_into_\*_graph* in the file *convert_\*.py*. As for Cpp language, for example, we apply *convert_into_cpp_graph* 
```
convert_into_cpp_graph(tree, text, has_data_flow = True, has_cdg = True)
    * tree: root node of the parse tree produced by tree-sitter
    * text: source code
```
The function returns
```
    * refactor_json: parse tree is simplified (unused)
    * ast_nodes: original AST nodes cannot be represented as subtrees
    * stmts: all subtrees in the Subtree-level layer
    * stmt_indices: index of each subtree in the graph (G) of the AST-level layer 
    * G: graph in the AST-level layer
```
#### Experiments
1. Firstly, download models and datasets from the links https://drive.google.com/drive/folders/10IeuSO54CQWHmYfcMMBfoutus3R1AxdE?usp=sharing
2. Update configuration files in the configs so that the paths inside them are valid
3. Run the provided script
    * Classification: train[1000|1400|104]_[pretrain|scratch].sh
    * Clone detection: train_clone_astnn_pretrain.sh
    * Any-code completion: train_anycode_java.sh