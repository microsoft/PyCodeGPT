# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Transform code to sketch"""
from redbaron import RedBaron, NameNode, NodeList, Node
from typing import List, Dict, Tuple, Union, Iterable

def traverse_node_fst(node_fst):
    if isinstance(node_fst, list):
        for this_node in node_fst:
            traverse_node_fst(this_node)
    elif isinstance(node_fst, dict):
        if node_fst.get("type") is not None:
            this_type = node_fst.get("type")
            if node_fst.get("name") is not None:
                if this_type == "def":
                    node_fst["name"] = "func"
                elif this_type == "class":
                    node_fst["name"] = "AnClass"
            if node_fst.get("value") is not None:
                if this_type == "raw_string":
                    node_fst["value"] = "rawstring"
                elif this_type == "int":
                    node_fst["value"] = "number"
                elif this_type == "interpolated_raw_string":
                    node_fst["value"] = "interrawstring"
                elif this_type == "complex":
                    node_fst["value"] = "complex" # 1j
                elif this_type == "string" and "\"\"\"" not in node_fst["value"] and "\'\'\'" not in node_fst["value"]:
                    node_fst["value"] = "string"
                elif this_type == "float_exponant":
                    node_fst["value"] = "floatexponant"
                elif this_type == "interpolated_string":
                    node_fst["value"] = "interstring"
                elif this_type == "float":
                    node_fst["value"] = "float"
                elif this_type == "binary_string":
                    node_fst["value"] = "binarystring"
                elif this_type == "unicode_string":
                    node_fst["value"] = "unicodestring"
                else:
                    pass
   
        for this_key in node_fst:
            if isinstance(node_fst[this_key], list) or isinstance(node_fst[this_key], dict):
                traverse_node_fst(node_fst[this_key])

    return node_fst

def transform_code_to_sketch(desp: str):
    red = RedBaron(desp)
    node_fst = red.fst()
    node_fst = traverse_node_fst(node_fst)
    code_schema = NodeList.from_fst(node_fst).dumps()
    return code_schema


def craft_merged_corpus(sketch_list:List=[] , text_list:List=[], linker="\n"):
    sketch_norm_list = []
    for this_sketch, this_text in zip(sketch_list, text_list):
        if this_text.count("import") >= 2 or "__name__" in this_text: # whether removing the highest overlap schema
            sketch_norm_list.append(this_text)
        else:
            sketch_norm_list.append(this_sketch+linker+this_text)
    return "\n\n\n".join(sketch_norm_list)