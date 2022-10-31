import re
import json
import os
import glob
from typing import List, Set, Dict, Tuple
from tqdm import tqdm
import random

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

def get_libraries_path_from_code_content(code_content: str) -> List[str]:
    """
    Get libraries from code content.
    :param code_content: code content
    :return: libraries
    """
    matched_items = re.findall(r"(?m)^(?:from[ ]+(\S+)[ ]+)?import[ ]+(\S+)(?:[ ]+as[ ]+\S+)?[ ]*$", code_content)
    relevanted_libraries = [a+"."+b if a != "" else b for a, b in matched_items]
    return relevanted_libraries

def get_libraries_name_from_path(relevanted_libraries: List[str]) -> Set[str]:
    """
    get libraries name from libraries path
    :param relevanted_libraries: libraries path
    :return: libraries name
    """
    return list(set(lib_path.split(".")[0] for lib_path in relevanted_libraries))

def judge_if_import_header_block(block_text: str) -> bool:
    """
    judge if the block is import header block
    :param block_text: block text
    :return: True or False
    """
    return True if len(re.findall(r"(?m)^(?:from[ ]+(\S+)[ ]+)?import[ ]+(\S+)(?:[ ]+as[ ]+\S+)?[ ]*$", block_text)) > 0 else False

def judge_if_comment_block(block_text: str) -> bool:
    """
    judge if the block is comment block
    :param block_text: block text
    :return: True or False
    """
    is_all_comment = True
    for line in block_text.split("\n"):
        if line.strip()[:1] != "#":
            is_all_comment = False
    return is_all_comment
        
def detect_api_names_from_code_block(
    code_block: str,
    concat_mode: str
    ) -> List[str]:
    """
    detect function name from code block
    :param code_block: code block
    :return: function name list
    """
    assert concat_mode in ["only function", "all"]
    api_function_names, api_property_names = re.findall(r"(\w+)\(", code_block), re.findall(r"\.(\w+)\[", code_block)
    return api_function_names + api_property_names if concat_mode == "all" else api_function_names

def normalizer_api_desp(code_api_content: str) -> str:
    """
    normalize api content
    :param code_api_content: api content
    :return: normalized api content
    """
    return re.sub(r"[\n\s]+", " ", code_api_content)

def get_first_sentence_from_api_desp(api_desp: str) -> str:
    """
    get first sentence from api desp
    """
    return api_desp.split(".")[0].strip() + "."


def get_api_prompter(
        code_block_text,
        block_api_names, 
        library_names,
        lib_name_api_name_api_paths, 
        api_path_api_signature_and_api_desp, 
        concat_mode,
        perturbation_probability
    ):
    """
    get api prompter
    :param block_api_names: block api names
    :param library_names: library names
    :param lib_name_api_name_api_paths: lib name and api name and api paths
    :param api_path_api_signature_and_api_desp: api path and api signature and api desp
    :param concat_mode: concat mode (concat_mode = "only function" or concat_mode = "all")
    :param perturbation_probability: the probability of perturbation
    """
    block_total_api_docs = 0 

    deduplication_block_api_names = list(set(block_api_names))
    deduplication_block_api_names.sort(key=block_api_names.index)

    block_total_api_funcs = len(deduplication_block_api_names)

    block_api_paths = ["" for _ in deduplication_block_api_names]

    for lib_name in library_names:
        if lib_name_api_name_api_paths.get(lib_name) is None:
            continue
        for api_name, api_paths_list in lib_name_api_name_api_paths[lib_name].items():
            if api_name in deduplication_block_api_names:
                api_name_index = deduplication_block_api_names.index(api_name)

                if block_api_paths[api_name_index] == "": 
                    block_total_api_docs += len(api_paths_list)
                    block_api_paths[api_name_index] = random.choice(api_paths_list)

                    # add perturbation
                    if random.random() < perturbation_probability:
                        random_api_path = random.choice(lib_name_api_name_api_paths[lib_name][random.sample(lib_name_api_name_api_paths[lib_name].keys(), 1)[0]])
                        block_api_paths.append(random_api_path)

    block_retrieved_api_funcs = len(block_api_paths)-block_api_paths.count("")
    block_prompt_content = "# [start]\n"
    """
    prompt format:
    # [start]
    # api_signature: api_desp
    # api_signature: api_desp
    # ...
    # [end]
    """

    random.shuffle(block_api_paths)
    for block_api_path in block_api_paths:
        if block_api_path == "":
            continue
        api_name = api_path_api_signature_and_api_desp[block_api_path]["api_name"]
        api_sign = api_path_api_signature_and_api_desp[block_api_path]["api_signature"]
        api_desp = normalizer_api_desp(api_path_api_signature_and_api_desp[block_api_path]["api_description"])
        first_sententce = get_first_sentence_from_api_desp(api_desp)

        if api_sign == "":
            continue
        block_prompt_content += f"# {api_name}{api_sign}: {first_sententce}\n"
    block_prompt_content += "# [end]\n"

    return block_prompt_content, block_total_api_funcs, block_total_api_docs, block_retrieved_api_funcs

def extract_main_comment_from_code(code_block):
    import re
    double_comment = re.findall(r'\"\"\"[\S\s.]+?\"\"\"', code_block)
    single_comment = re.findall(r"\'\'\'[\S\s.]+?\'\'\'", code_block)
    if len(double_comment) > 0:
        return double_comment[0]
    elif len(single_comment) > 0:
        return single_comment[0]
    else:
        return ""

def extract_all_comment_from_code(code_block):
    import re
    all_comment = re.findall(r'#[\S\s.]+?\n', code_block)
    if len(all_comment) > 0:
        return "@@@".join(all_comment)
    else:
        return ""

def get_api_prompter_retrieval(
        code_block_text,
        block_api_names, 
        library_names,
        lib_name_api_name_api_paths, 
        api_path_api_signature_and_api_desp, 
        concat_mode,
        perturbation_probability,
        pos_neg_ratio,
    ):
    """
    get api prompter
    :param block_api_names: block api names
    :param library_names: library names
    :param lib_name_api_name_api_paths: lib name and api name and api paths
    :param api_path_api_signature_and_api_desp: api path and api signature and api desp
    :param concat_mode: concat mode (concat_mode = "only function" or concat_mode = "all")
    :param perturbation_probability: the probability of perturbation
    :return: api prompt
    """
    block_total_api_docs = 0 

    deduplication_block_api_names = list(set(block_api_names))
    deduplication_block_api_names.sort(key=block_api_names.index)

    block_total_api_funcs = len(deduplication_block_api_names)

    block_api_paths = ["" for _ in deduplication_block_api_names]
    neg_block_api_paths = []

    for lib_name in library_names:
        if lib_name_api_name_api_paths.get(lib_name) is None:
            continue

        for api_name, api_paths_list in lib_name_api_name_api_paths[lib_name].items():
            if api_name in deduplication_block_api_names:
                api_name_index = deduplication_block_api_names.index(api_name)

                if block_api_paths[api_name_index] == "": 
                    block_total_api_docs += len(api_paths_list)
                    block_api_paths[api_name_index] = random.choice(api_paths_list)

                    for this_key in random.sample(lib_name_api_name_api_paths[lib_name].keys(), pos_neg_ratio):
                        neg_block_api_paths.append(random.choice(lib_name_api_name_api_paths[lib_name][this_key]))

    block_retrieved_api_funcs = len(block_api_paths)-block_api_paths.count("")

    block_api_code_pairs = {
        "code_block": code_block_text,
        "code_doc": extract_main_comment_from_code(code_block_text),
        "code_all_doc": extract_all_comment_from_code(code_block_text),
        "positive_APIs": [],
        "negative_APIs": []
    }
    """
    {
        "code_block": code_block,
        "code_doc": code_doc,
        "code_all_doc": code_all_doc,
        "positive_APIs":{
            api_path: [api_name, api_signature, api_desp],
            ...
        },
        "negative_APIs":{
            api_path: [api_name, api_signature, api_desp],
            ...
        }
    }
    """

    if block_api_code_pairs["code_doc"] == "":
        return block_api_code_pairs, block_total_api_funcs, block_total_api_docs, block_retrieved_api_funcs

    for block_api_path in block_api_paths:
        if block_api_path == "":
            continue
        if not block_api_path in block_api_code_pairs["positive_APIs"]:
            block_api_code_pairs["positive_APIs"].append(block_api_path)

    for block_api_path in neg_block_api_paths:
        if block_api_path == "":
            continue
        if not block_api_path in block_api_code_pairs["negative_APIs"]:
            block_api_code_pairs["negative_APIs"].append(block_api_path)

    return block_api_code_pairs, block_total_api_funcs, block_total_api_docs, block_retrieved_api_funcs

def get_dict_of_api_name_lib_api_paths(
    private_data_path: str, 
    private_libs: str, 
    build_in_libs: str,
    contain_build_in: str
    ):
    """
    get the dictionary of api name and lib api paths
    the dictionary format is :
    {
        "lib_name": {
            "api_name":[api_path, api_path, ...],
            ...
        },
        ...
    }
    return the dictionary
    """
    lib_name_api_name_api_paths = {}
    all_lib_list_names = private_libs.split(",") + build_in_libs.split(",") if contain_build_in == "True" else private_libs.split(",")

    logging.info(f"Start to get the dictionary of api name and lib api paths from the total length of libs: {len(all_lib_list_names)}")
    for this_lib_name in tqdm(all_lib_list_names):
        this_lib_path = os.path.join(private_data_path, this_lib_name, f"{this_lib_name}_apis_doc_details.jsonl")
        assert os.path.exists(this_lib_path), "lib path not exists"
        lib_api_reader = open(this_lib_path, "r")
        api_name_api_paths = {}
        for line in lib_api_reader:
            line_dict = eval(line)
            api_name_api_paths[line_dict["api_name"]] = [line_dict["api_path"]] if api_name_api_paths.get(line_dict["api_name"]) is None \
                else api_name_api_paths[line_dict["api_name"]] + [line_dict["api_path"]]
        lib_name_api_name_api_paths[this_lib_name] = api_name_api_paths

        # break # debug: avaoid to read all libs， if you want to read all libs, delete this line.
    logging.info(f"Have loaded the lib_name_api_name_api_paths dictionary, which has {len(lib_name_api_name_api_paths)} libs.")
    return lib_name_api_name_api_paths

def get_dict_of_api_path_api_signature_and_api_desp(
    private_data_path: str, 
    private_libs: str, 
    build_in_libs: str,
    contain_build_in: str
    ):
    """
    get the dictionary of api path -> api signature and api desp
    the dictionary format is :
    {
        "api_path": {
            "api_signature": api_signature,
            "api_desp": api_desp
        },
        ...
    }
    """
    api_path_api_signature_and_api_desp = {}
    all_lib_list_names = private_libs.split(",") + build_in_libs.split(",") if contain_build_in == "True" else private_libs.split(",")

    for this_lib_name in all_lib_list_names:
        this_lib_path = os.path.join(private_data_path, this_lib_name, f"{this_lib_name}_apis_doc_details.jsonl")
        assert os.path.exists(this_lib_path), "lib path not exists"
        lib_api_reader = open(this_lib_path, "r")
        for line in lib_api_reader:
            line_dict = eval(line)
            api_path_api_signature_and_api_desp[line_dict["api_path"]] = {
                "api_name": line_dict["api_name"],
                "api_signature": line_dict["api_signature"],
                "api_description": line_dict["api_description"]
            }

        # break # debug: avaoid to read all libs， if you want to read all libs, delete this line.
    # logging.info(f"Have loaded the api_path_api_signature_and_api_desp dictionary, which has {len(api_path_api_signature_and_api_desp)} apis.")
    return api_path_api_signature_and_api_desp

def get_our_defined_function_names(code_str):
    """
    get our defined function names from the code string
    :param: code_str: the code string
    :return: our defined function names list
    """
    import re
    our_defined_names = []
    for def_func_name in re.findall(r"def[\s]+[a-zA-Z_]+\(", code_str):
        our_defined_names.append(def_func_name.replace("def", "").replace("(", "").strip())
    return our_defined_names

def is_class_and_contain_multiple_functions(block_str: str) -> bool:
    """
    is the block is a class and contain multiple functions?
    :param block_str: the block string
    :return: True or False
    """
    import re
    if re.search(r"class[\s]+[a-zA-Z_]+[\(:]{1}", block_str) and re.search(r"def[\s]+[a-zA-Z_]+\(", block_str):
        return True
    return False

def split_code_block(code_block: str) -> list:
    """
    Split a code block into a list of lines according to the following rules:
    --- a function definition starts with 'def'
    """
    lines = code_block.split('\n\n')
    code_block_list = []
    sub_code_block = ""
    for line in lines:
        if "def" not in line:
            sub_code_block = sub_code_block + line if sub_code_block == "" else sub_code_block + "\n\n" + line
        else:
            if sub_code_block != "":
                code_block_list.append(sub_code_block)
                sub_code_block = ""
            sub_code_block = line

    if sub_code_block != "":
        code_block_list.append(sub_code_block)
    return code_block_list

def get_pre_space(sub_code_block: str) -> int:
    """
    Get the number of spaces before the first line of a code block
    """
    space_number = 0
    for char in sub_code_block:
        if char == " ":
            space_number += 1
        else:
            break
    return space_number

def re_encapsulate_sub_api_prompter(sub_api_prompter: str, space_number: int) -> str:
    """
    add the space number to the sub api prompter
    """
    new_sub_api_prompter_list = list()
    for idx, sub_prompter_line in enumerate(sub_api_prompter.split("\n")):
        if idx != len(sub_api_prompter.split("\n")) - 1: # not the last line
            new_sub_api_prompter_list.append(space_number*" " + sub_prompter_line)
        else:
            new_sub_api_prompter_list.append(sub_prompter_line)
    return "\n".join(new_sub_api_prompter_list)

def merge_all_subs_code_block(subs_code_block, this_block_api_prompter_list):
    """
    temporarily not used, this implementation seems to be wrong
    """
    new_this_block_api_prompter_list = list()
    for sub_code_block, sub_api_prompter in zip(subs_code_block, this_block_api_prompter_list):
        space_number = get_pre_space(sub_code_block)
        new_this_block_api_prompter_list.append(re_encapsulate_sub_api_prompter(sub_api_prompter, space_number))
        pass
    return "\n\n".join(new_this_block_api_prompter_list)

def pre_process_text_list(text_list: list) -> list:
    """
    some code block in the text_list may be a class and contain multiple def functions, then we need to split it.
    """
    new_text_list = list()
    for code_block in text_list:
        if not is_class_and_contain_multiple_functions(code_block):
            new_text_list.append(code_block)
        else:
            subs_code_block = split_code_block(code_block)
            new_text_list += subs_code_block

    return new_text_list

def craft_api_merged_corpus(
    norm_text: str,
    text_list: List[str], 
    linker :str,
    lib_name_api_name_api_paths: Dict[str, Dict[str, List[str]]],
    api_path_api_signature_and_api_desp: Dict[str, Dict[str, str]],
    concat_mode: str,
    perturbation_probability: float
    ) -> str:
    """
    make the api sketch merged corpus
    :param text_list: api text list
    :param linker: linker
    :param lib_name_api_name_api_paths: lib name and api name and api paths
    :param api_path_api_signature_and_api_desp: api path and api signature and api desp
    :param concat_mode: concat mode (concat_mode = "only function" or concat_mode = "all")
    """
    library_names = get_libraries_name_from_path(get_libraries_path_from_code_content(norm_text))
    our_defined_func_names = get_our_defined_function_names(norm_text)
    cleaned_library_names = [func_name for func_name in library_names if func_name not in our_defined_func_names]
    api_prompter_list = list()
    total_api_funcs, total_api_docs, total_retrieved_api_funcs = 0, 0, 0
    new_text_list = pre_process_text_list(text_list)
    for code_block in new_text_list:
        if judge_if_import_header_block(code_block) or judge_if_comment_block(code_block):
            api_prompter_list.append("")
        else:
            block_api_names = detect_api_names_from_code_block(code_block, concat_mode)
            block_prompt_content, block_total_api_funcs, block_total_api_docs, block_retrieved_api_funcs = get_api_prompter(
                code_block,
                block_api_names, 
                cleaned_library_names,
                lib_name_api_name_api_paths, 
                api_path_api_signature_and_api_desp, 
                concat_mode,
                perturbation_probability=perturbation_probability
            )
            api_prompter_list.append(block_prompt_content if block_prompt_content != "# [start]\n# [end]\n" else "")
            if block_prompt_content != "# [start]\n# [end]\n":
                total_api_funcs += block_total_api_funcs
                total_api_docs += block_total_api_docs
                total_retrieved_api_funcs += block_retrieved_api_funcs
    
    assert len(api_prompter_list) == len(new_text_list)
    file_code_content = ""
    pre_block_code = ""
    for api_prompt, block_code in zip(api_prompter_list, new_text_list):
        assert api_prompt != "# [start]\n# [end]\n" 
        block_code_space_number = get_pre_space(block_code) 
        api_prompt = re_encapsulate_sub_api_prompter(api_prompt, block_code_space_number) if api_prompt != "" else ""
        file_code_content += f"{api_prompt}{linker}{block_code}\n\n\n" if block_code_space_number == 0 else f"{api_prompt}{linker}{block_code}\n\n"
        pre_block_code = block_code
    file_code_content = file_code_content.strip()

    return file_code_content, total_api_funcs, total_api_docs, total_retrieved_api_funcs

def craft_code_context_and_doc_pairs(
    norm_text: str,
    text_list: List[str],
    linker :str,
    lib_name_api_name_api_paths: Dict[str, Dict[str, List[str]]],
    api_path_api_signature_and_api_desp: Dict[str, Dict[str, str]],
    concat_mode: str
    ):
    """
    make the api sketch merged corpus
    :param text_list: api text list
    :param linker: linker
    :param lib_name_api_name_api_paths: lib name and api name and api paths
    :param api_path_api_signature_and_api_desp: api path and api signature and api desp
    :param concat_mode: concat mode (concat_mode = "only function" or concat_mode = "all")
    :return: the format is : [
            {
                "code_block": code_block,
                "code_doc": code_doc,
                "code_all_doc": code_all_doc,
                "positive_APIs":{
                    api_name: api_desp,
                    ...
                },
                "negative_APIs":{
                    api_name: api_desp,
                    ...
                }
            },
            ...
    ]
    """
    api_code_pairs_list = list()

    library_names = get_libraries_name_from_path(get_libraries_path_from_code_content(norm_text))
    for code_block in text_list:
        if judge_if_import_header_block(code_block):
            pass
        else:
            block_api_names = detect_api_names_from_code_block(code_block, concat_mode)
            block_api_code_pairs, block_total_api_funcs, block_total_api_docs, block_retrieved_api_funcs = get_api_prompter_retrieval(
                code_block,
                block_api_names, 
                library_names,
                lib_name_api_name_api_paths, 
                api_path_api_signature_and_api_desp, 
                concat_mode,
                0,
                50
            )

            if block_api_code_pairs["code_doc"] == "" or block_total_api_funcs == 0 or block_retrieved_api_funcs == 0:
                continue
            api_code_pairs_list.append(block_api_code_pairs)
    
    return api_code_pairs_list
