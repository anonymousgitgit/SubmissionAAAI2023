import sys
from pathlib import Path

from data.convert_c import convert_into_c_graph
sys.path.append(str(Path(__file__).parent.parent.absolute()))
import glob
from tree_sitter import Language, Parser
from functools import lru_cache
from typing import List
from tqdm import tqdm
import yaml


def split_camelcase(camel_case_identifier: str) -> List[str]:
    """
    Split camelCase identifiers.
    """
    if not len(camel_case_identifier):
        return []

    # split into words based on adjacent cases being the same
    result = []
    current = str(camel_case_identifier[0])
    prev_upper = camel_case_identifier[0].isupper()
    prev_digit = camel_case_identifier[0].isdigit()
    prev_special = not camel_case_identifier[0].isalnum()
    for c in camel_case_identifier[1:]:
        upper = c.isupper()
        digit = c.isdigit()
        special = not c.isalnum()
        new_upper_word = upper# and not prev_upper
        new_digit_word = digit and not prev_digit
        new_special_word = special and not prev_special
        if new_digit_word or new_upper_word or new_special_word:
            result.append(current)
            current = c
        elif not upper and prev_upper and len(current) > 1:
            result.append(current[:-1])
            current = current[-1] + c
        elif not digit and prev_digit:
            result.append(current)
            current = c
        elif not special and prev_special:
            result.append(current)
            current = c
        else:
            current += c
        prev_digit = digit
        prev_upper = upper
        prev_special = special
    result.append(current)
    return result


@lru_cache(maxsize=5000)
def split_identifier_into_parts(identifier: str) -> List[str]:
    """
    Split a single identifier into parts on snake_case and camelCase
    """
    snake_case = identifier.split("_")

    identifier_parts = []
    for i in range(len(snake_case)):
        part = snake_case[i]
        if len(part) > 0:
            identifier_parts.extend(s.lower() for s in split_camelcase(part))
    if len(identifier_parts) == 0:
        return [identifier]
    identifier_parts = [x for x in identifier_parts if x]
    return identifier_parts

def get_sub_tokens(root):
    token_dict, type_dict = set(), set()
    queue = [root]
    while queue:
        node = queue.pop(0)
        token_dict.update(split_identifier_into_parts(node['node_token']))
        type_dict = type_dict | {node['node_type']}
        queue.extend(node['children'])
    return token_dict, type_dict

def update_all_configs(vocab_token_size, vocab_type_size):
    config_paths = glob.glob('configs/*.yml')
    for path in config_paths:
        content = yaml.load(open(path), Loader = yaml.FullLoader)
        content['vocab_token_size'] = vocab_token_size
        content['vocab_type_size'] = vocab_type_size
        yaml.dump(content, open(path, 'w'), default_flow_style = False)

