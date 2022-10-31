import os
import json
import re
import sys
from tqdm import tqdm

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def normalize_text(text):
    """
    normalize text
    """
    text = re.sub(r"[\s]*[\n]$", "", re.sub(r"^[\n]+[\s]*", "", text.strip()))
    return text

def get_number_of_params(api_signature: str, library_name: str) -> str:
    """
    get number of parameters from api signature
    """
    prefix = ">" if "**kwargs" in api_signature else "="
    api_signature = re.sub(r"[\s]*[,]*[\s]*\*\*kwargs", "", re.sub(r"self[\s]*[,]*[\s]*", "", re.sub(r"[\s]*->(.)*", "", api_signature))).replace("(", "").replace(")", "").strip()
    param_nums = 0 if len(api_signature) == 0 else len(api_signature.split(","))
    return prefix + str(param_nums)

def get_key_words(library_path, library_name):
    """
    get key words from api's docstring
    """
    all_key_words_set = set()
    library_reader = open(library_path, "r")
    if library_name == "pandas" or library_name == "numpy" or library_name == "sklearn" or library_name == "matplotlib" or library_name == "scipy"\
        or library_name == "seaborn" or library_name == "nltk" or library_name == "pygame" or library_name == "gensim" or library_name == "spacy"\
        or library_name == "fairseq" or library_name == "datasets" or library_name == "mxnet" or library_name == "imageio" or library_name == "metpy":
        for line in library_reader:
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                this_kw_list = [kw.replace("\n","").replace("-","").strip() for kw in re.findall(r"[\s]*[A-Za-z]+[ ]*[A-Za-z]*[\n]*[\s]+[-]{2,}", api_doc, re.I)]
                all_key_words_set.update(this_kw_list)
    elif library_name == "torch" or library_name == "tensorflow" or library_name == "torchdata":
        for line in library_reader:
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                this_kw_list = [kw.replace("\n","").replace(":","").strip() for kw in re.findall(r"[\n]+[\s]*[a-zA-Z]+[\s]*[a-zA-Z]*:[\s]*[\n]+", api_doc, re.I)]
                all_key_words_set.update(this_kw_list)
    elif library_name == "torcharrow":
        for line in library_reader:
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                this_kw_list = [kw.replace("\n","").replace(":","").strip() for kw in re.findall(r"[\n]{2,}[\s]*[a-zA-Z]+[\s]*[a-zA-Z]*[\n]{1,}", api_doc, re.I)]
                all_key_words_set.update(this_kw_list)
    elif library_name == "selenium":
        for line in library_reader:
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                this_kw_list = [kw.replace("\n","").replace(":","").strip() for kw in re.findall(r"[\n]+[\s]*[:]*[a-zA-Z]+[\s]*[a-zA-Z]*[:]*[\s]*[\n]+", api_doc, re.I)]
                all_key_words_set.update(this_kw_list)
    elif library_name == "beautifulsoup":
        for line in library_reader:
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                this_kw_list = [kw.replace("\n","").replace(":","").strip() for kw in re.findall(r"[\n]+[\s]*[a-zA-Z]+[\s]*[a-zA-Z]*:[\s]*[\n]*", api_doc, re.I)]
                all_key_words_set.update(this_kw_list)
    elif library_name == "jieba":
        for line in library_reader:
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                this_kw_list = [kw.replace("\n","").replace(":","").strip() for kw in re.findall(r"[\n]{1,}[\s]*[a-zA-Z]+[\s]*[a-zA-Z]*:[\s]*", api_doc, re.I)]
                all_key_words_set.update(this_kw_list)
    elif library_name == "transformers":
        for line in library_reader:
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                this_kw_list = [kw.replace("\n","").replace(":","").strip() for kw in re.findall(r"[\n]{2,}[\s]*[a-zA-Z]+[\s]*[a-zA-Z]*:[\n]+", api_doc, re.I)]
                all_key_words_set.update(this_kw_list)
    elif library_name == "sqlalchemy" or library_name == "allennlp":
        for line in library_reader:
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                this_kw_list = [kw.replace("\n","").replace(":","").strip() for kw in re.findall(r"[\n]{2,}[\s]*[a-zA-Z]+[\s]*[a-zA-Z]*[:]+[\n]+", api_doc, re.I)]
                all_key_words_set.update(this_kw_list)
    elif library_name == "django":
        for line in library_reader:
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                this_kw_list = [kw.replace("\n","").replace(":","").strip() for kw in re.findall(r"[\n]{2,}[\s]*[a-zA-Z]+[\s]*[a-zA-Z]*[:]+[\n]+", api_doc, re.I)]
                all_key_words_set.update(this_kw_list)
    elif library_name == "flask":
        for line in library_reader:
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                this_kw_list = [kw.replace("\n","").replace(":","").strip() for kw in re.findall(r"[\n]{2,}[\s]*[a-zA-Z]+[\s]*[a-zA-Z]*[:]+[\n]+", api_doc, re.I)]
                all_key_words_set.update(this_kw_list)
    elif library_name == "PIL":
        for line in library_reader:
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                this_kw_list = [kw.replace("\n","").replace(":","").strip() for kw in re.findall(r"[\n]+[\s]*[a-zA-Z]+[\s]*[a-zA-Z]*[:]+[\n]*", api_doc, re.I)]
                all_key_words_set.update(this_kw_list)
    elif library_name == "scrapy":
        for line in library_reader:
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                this_kw_list = [kw.replace("\n","").replace(":","").strip() for kw in re.findall(r"[\n]+[\s]*[a-zA-Z]+[\s]*[a-zA-Z]*[:]+[\n]*", api_doc, re.I)]
                all_key_words_set.update(this_kw_list)
    elif library_name == "tokenizers":
        for line in library_reader:
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                this_kw_list = [kw.replace("\n","").replace(":","").strip() for kw in re.findall(r"[\n]+[\s]*[a-zA-Z]+[\s]*[a-zA-Z]*[:]+[\n]*", api_doc, re.I)]
                all_key_words_set.update(this_kw_list)
    elif library_name == "pytest" or library_name == "ansible" or library_name == "requests":
        for line in library_reader:
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                this_kw_list = [kw.replace("\n","").replace(":","").strip() for kw in re.findall(r"[\n]+[\s]*[a-zA-Z]+[\s]*[a-zA-Z]*:[\n]+", api_doc, re.I)]
                all_key_words_set.update(this_kw_list)
    elif library_name == "datetime" or library_name == "zlib" or library_name == "random" or library_name == "math" or library_name == "sys" or library_name == "glob"\
        or library_name == "os" or library_name == "urllib" or library_name == "uuid" or library_name == "pprint" or library_name == "time" or library_name == "re"\
        or library_name == "json" or library_name == "unittest" or library_name == "collections" or library_name == "subprocess" or library_name == "copy"\
        or library_name == "functools" or library_name == "six" or library_name == "itertools" or library_name == "threading" or library_name == "tempfile"\
        or library_name == "io" or library_name == "pickle" or library_name == "pathlib" or library_name == "socket" or library_name == "struct" or library_name == "hashlib"\
        or library_name == "traceback" or library_name == "csv":
        for line in library_reader:
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                this_kw_list = [kw.replace("\n","").replace(":","").strip() for kw in re.findall(r"[\n]{2}[\s]*[a-zA-Z]+[\s]*[a-zA-Z]*:", api_doc, re.I)]
                all_key_words_set.update(this_kw_list)
    else:
        raise NotImplementedError(f"{library_name} not implemented")
        
    return all_key_words_set

def write_to_file(details_writer, api_infos):
    """
    write api infos to file
    """
    details_writer.write(json.dumps({
        "api_path": api_infos[0],
        "api_name": api_infos[1],
        "api_doc": api_infos[2],
        "api_signature": api_infos[3],
        "api_description": api_infos[4],
        "api_parameters": api_infos[5],
        "api_parameters_number": api_infos[6],
        "api_returns": api_infos[7],
        "api_see_also": api_infos[8],
        "api_notes": api_infos[9],
        "api_examples": api_infos[10]
    })+"\n")

def judge_none(code_doc):
    tmp_doc = code_doc.strip()
    if tmp_doc == "" or tmp_doc == None or tmp_doc == "None" or tmp_doc == "null" or tmp_doc == "NULL" or tmp_doc == "Null":
        return True
    else:
        return False
        

def get_details(library_path, all_key_words, library_name, details_writer):
    """
    input: api's docstring
    output: api's description, parameters, returns, see_also, notes, examples
    """
    library_reader = open(library_path, "r")
    if library_name == "pandas" or library_name == "numpy" or library_name == "sklearn" or library_name == "scipy" or library_name == "seaborn"\
        or library_name == "nltk" or library_name == "pygame" or library_name == "gensim" or library_name == "spacy" or library_name == "fairseq"\
        or library_name == "datasets":
        for line in tqdm(library_reader):
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                api_description, api_parameters, api_returns, api_see_also, api_notes, api_examples = "", "", "", "", "", ""
                if judge_none(api_doc):
                    continue
                api_doc_formated = api_doc
                for kw_line in all_key_words:
                    pattern = r"[\s]*{}[\n]*[\s]+[-]".format(kw_line) + r"{2,}"
                    api_doc_formated = re.sub(pattern, f"######{kw_line}******", api_doc_formated, flags=re.I)
                api_doc_formated_list = api_doc_formated.split("######")
                if len(api_doc_formated_list) < 1:
                    continue
                for api_doc_formated_item in api_doc_formated_list:
                    if "******" not in api_doc_formated_item and len(api_doc_formated_item) > 0:
                        api_description = normalize_text(api_doc_formated_item)
                    elif "******" in api_doc_formated_item and ("Parameters" in api_doc_formated_item or "Other Parameters" in api_doc_formated_item\
                        or "Arguments" in api_doc_formated_item):
                        api_parameters += "\n"+normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Returns" in api_doc_formated_item:
                        api_returns = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "See Also".lower() in api_doc_formated_item.lower():
                        api_see_also = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Notes" in api_doc_formated_item:
                        api_notes = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and ("Examples" in api_doc_formated_item or "Example" in api_doc_formated_item):
                        api_examples += "\n"+normalize_text(api_doc_formated_item.split("******")[1])
                    else:
                        pass
                write_to_file(details_writer, [api_path, api_name, api_doc, api_signature, api_description, api_parameters, get_number_of_params(api_signature, library_name),\
                    api_returns, api_see_also, api_notes, api_examples])
        details_writer.close()
    elif library_name == "torch":
        for line in tqdm(library_reader):
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                api_description, api_parameters, api_returns, api_see_also, api_notes, api_examples = "", "", "", "", "", ""
                if judge_none(api_doc):
                    continue
                api_doc_formated = api_doc
                for kw_line in all_key_words:
                    pattern = r"[\n]+[\s]*{}:[\s]*[\n]+".format(kw_line)
                    api_doc_formated = re.sub(pattern, f"######{kw_line}******", api_doc_formated, flags=re.I)
                api_doc_formated_list = api_doc_formated.split("######")
                if len(api_doc_formated_list) < 1:
                    continue
                for api_doc_formated_item in api_doc_formated_list:
                    if "******" not in api_doc_formated_item and len(api_doc_formated_item) > 0:
                        api_description = normalize_text(api_doc_formated_item)
                    elif "******" in api_doc_formated_item and "Args" in api_doc_formated_item and "Keyword Args" not in api_doc_formated_item:
                        api_parameters += "\n"+normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Returns" in api_doc_formated_item:
                        api_returns = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "See Also" in api_doc_formated_item:
                        api_see_also = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Note" in api_doc_formated_item:
                        api_notes = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Examples" in api_doc_formated_item:
                        api_examples = normalize_text(api_doc_formated_item.split("******")[1])
                    else:
                        pass
                write_to_file(details_writer, [api_path, api_name, api_doc, api_signature, api_description, api_parameters, get_number_of_params(api_signature, library_name),\
                    api_returns, api_see_also, api_notes, api_examples])
        details_writer.close()
    elif library_name == "torchdata":
        for line in tqdm(library_reader):
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                api_description, api_parameters, api_returns, api_see_also, api_notes, api_examples = "", "", "", "", "", ""
                if judge_none(api_doc):
                    continue
                api_doc_formated = api_doc
                for kw_line in all_key_words:
                    pattern = r"[\n]+[\s]*{}:[\s]*[\n]+".format(kw_line)
                    api_doc_formated = re.sub(pattern, f"######{kw_line}******", api_doc_formated, flags=re.I)
                api_doc_formated_list = api_doc_formated.split("######")
                if len(api_doc_formated_list) < 1:
                    continue
                for api_doc_formated_item in api_doc_formated_list:
                    if "******" not in api_doc_formated_item and len(api_doc_formated_item) > 0:
                        api_description = normalize_text(api_doc_formated_item)
                    elif "******" in api_doc_formated_item and "Args" in api_doc_formated_item:
                        api_parameters += "\n"+normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Returns" in api_doc_formated_item:
                        api_returns = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "See Also" in api_doc_formated_item:
                        api_see_also = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Note" in api_doc_formated_item:
                        api_notes = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Examples" in api_doc_formated_item:
                        api_examples = normalize_text(api_doc_formated_item.split("******")[1])
                    else:
                        pass
                write_to_file(details_writer, [api_path, api_name, api_doc, api_signature, api_description, api_parameters, get_number_of_params(api_signature, library_name),\
                    api_returns, api_see_also, api_notes, api_examples])
        details_writer.close()
    elif library_name == "torcharrow":
        for line in tqdm(library_reader):
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                api_description, api_parameters, api_returns, api_see_also, api_notes, api_examples = "", "", "", "", "", ""
                if judge_none(api_doc):
                    continue
                api_doc_formated = api_doc
                for kw_line in all_key_words:
                    pattern = r"[\n]{2,}[\s]*" + r"{}".format(kw_line) + r"[\n]{1,}"
                    api_doc_formated = re.sub(pattern, f"######{kw_line}******", api_doc_formated, flags=re.I)
                api_doc_formated_list = api_doc_formated.split("######")
                if len(api_doc_formated_list) < 1:
                    continue
                for api_doc_formated_item in api_doc_formated_list:
                    if "******" not in api_doc_formated_item and len(api_doc_formated_item) > 0:
                        api_description = normalize_text(api_doc_formated_item)
                    elif "******" in api_doc_formated_item and "Arguments" in api_doc_formated_item:
                        api_parameters += "\n"+normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Returns" in api_doc_formated_item:
                        api_returns = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "See Also".lower() in api_doc_formated_item.lower():
                        api_see_also = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Note" in api_doc_formated_item:
                        api_notes = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Examples" in api_doc_formated_item:
                        api_examples = normalize_text(api_doc_formated_item.split("******")[1])
                    else:
                        pass
                write_to_file(details_writer, [api_path, api_name, api_doc, api_signature, api_description, api_parameters, get_number_of_params(api_signature, library_name),\
                    api_returns, api_see_also, api_notes, api_examples])
        details_writer.close()
    elif library_name == "tensorflow":
        for line in tqdm(library_reader):
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                api_description, api_parameters, api_returns, api_see_also, api_notes, api_examples = "", "", "", "", "", ""
                if judge_none(api_doc):
                    continue
                api_doc_formated = api_doc
                for kw_line in all_key_words:
                    pattern = r"[\n]+[\s]*{}:[\s]*[\n]+".format(kw_line)
                    api_doc_formated = re.sub(pattern, f"######{kw_line}******", api_doc_formated, flags=re.I)
                api_doc_formated = re.sub(r"[\n]+[\s]*>>>[\s]*", "######>>>******", api_doc_formated, flags=re.I)
                api_doc_formated_list = api_doc_formated.split("######")
                if len(api_doc_formated_list) < 1:
                    continue
                for api_doc_formated_item in api_doc_formated_list:
                    if "******" not in api_doc_formated_item and len(api_doc_formated_item) > 0:
                        api_description = normalize_text(api_doc_formated_item)
                    elif "******" in api_doc_formated_item and "Args" in api_doc_formated_item and "Keyword Args" not in api_doc_formated_item\
                        and "False    Args" not in api_doc_formated_item and "True    Args" not in api_doc_formated_item and "Call Args" not in api_doc_formated_item\
                        and "True  Args" not in api_doc_formated_item and "running  Args" not in api_doc_formated_item:
                        api_parameters += "\n"+normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Returns" in api_doc_formated_item and "True    Returns" not in api_doc_formated_item\
                        and "False  Returns" not in api_doc_formated_item:
                        api_returns += "\n"+normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "See Also".lower() in api_doc_formated_item.lower():
                        api_see_also = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Note" in api_doc_formated_item:
                        api_notes = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and ("Examples" in api_doc_formated_item or ">>>" in api_doc_formated_item):
                        api_examples += "\n"+normalize_text(api_doc_formated_item.split("******")[1])
                    else:
                        pass
                write_to_file(details_writer, [api_path, api_name, api_doc, api_signature, api_description, api_parameters, get_number_of_params(api_signature, library_name),\
                    api_returns, api_see_also, api_notes, api_examples])
        details_writer.close()
    elif library_name == "selenium":
        for line in tqdm(library_reader):
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                api_description, api_parameters, api_returns, api_see_also, api_notes, api_examples = "", "", "", "", "", ""
                if judge_none(api_doc):
                    continue
                api_doc_formated = api_doc
                for kw_line in all_key_words:
                    pattern = r"[\n]+[\s]*[:]*{}:[\s]*[\n]+".format(kw_line)
                    api_doc_formated = re.sub(pattern, f"######{kw_line}******", api_doc_formated, flags=re.I)
                api_doc_formated_list = api_doc_formated.split("######")
                if len(api_doc_formated_list) < 1:
                    continue
                for api_doc_formated_item in api_doc_formated_list:
                    if "******" not in api_doc_formated_item and len(api_doc_formated_item) > 0:
                        api_description = normalize_text(api_doc_formated_item)
                    elif "******" in api_doc_formated_item and "Args" in api_doc_formated_item:
                        api_parameters = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Returns" in api_doc_formated_item:
                        api_returns = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "See Also" in api_doc_formated_item:
                        api_see_also = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Note" in api_doc_formated_item:
                        api_notes = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and ("Example" in api_doc_formated_item or "Usage example".lower() in api_doc_formated_item.lower()):
                        api_examples += "\n"+normalize_text(api_doc_formated_item.split("******")[1])
                    else:
                        pass
                write_to_file(details_writer, [api_path, api_name, api_doc, api_signature, api_description, api_parameters, get_number_of_params(api_signature, library_name),\
                    api_returns, api_see_also, api_notes, api_examples])
        details_writer.close()
    elif library_name == "matplotlib":
        for line in tqdm(library_reader):
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                api_description, api_parameters, api_returns, api_see_also, api_notes, api_examples = "", "", "", "", "", ""
                if judge_none(api_doc):
                    continue
                api_doc_formated = api_doc
                for kw_line in all_key_words:
                    pattern = r"[\s]*{}[\n]*[\s]+[-]".format(kw_line) + r"{2,}"
                    api_doc_formated = re.sub(pattern, f"######{kw_line}******", api_doc_formated, flags=re.I)
                api_doc_formated_list = api_doc_formated.split("######")
                if len(api_doc_formated_list) < 1:
                    continue
                for api_doc_formated_item in api_doc_formated_list:
                    if "******" not in api_doc_formated_item and len(api_doc_formated_item) > 0:
                        api_description = normalize_text(api_doc_formated_item)
                    elif "******" in api_doc_formated_item and ("Parameters" in api_doc_formated_item or "Other Parameters" in api_doc_formated_item):
                        api_parameters += "\n"+normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Returns" in api_doc_formated_item:
                        api_returns = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "See Also".lower() in api_doc_formated_item.lower():
                        api_see_also = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Notes" in api_doc_formated_item:
                        api_notes = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Examples" in api_doc_formated_item:
                        api_examples = normalize_text(api_doc_formated_item.split("******")[1])
                    else:
                        pass
                write_to_file(details_writer, [api_path, api_name, api_doc, api_signature, api_description, api_parameters, get_number_of_params(api_signature, library_name),\
                    api_returns, api_see_also, api_notes, api_examples])
        details_writer.close()
    elif library_name == "beautifulsoup":
        for line in tqdm(library_reader):
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                api_description, api_parameters, api_returns, api_see_also, api_notes, api_examples = "", "", "", "", "", ""
                if judge_none(api_doc):
                    continue
                api_doc_formated = api_doc
                for kw_line in all_key_words:
                    pattern = r"[\n]+[\s]*{}:[\s]*[\n]*".format(kw_line)
                    api_doc_formated = re.sub(pattern, f"######{kw_line}******", api_doc_formated, flags=re.I)
                api_doc_formated_list = api_doc_formated.split("######")
                if len(api_doc_formated_list) < 1:
                    continue
                for api_doc_formated_item in api_doc_formated_list:
                    if "******" not in api_doc_formated_item and len(api_doc_formated_item) > 0:
                        api_description = normalize_text(api_doc_formated_item)
                    elif "******" in api_doc_formated_item and "Arguments" in api_doc_formated_item:
                        api_parameters = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Returns" in api_doc_formated_item:
                        api_returns = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "See Also" in api_doc_formated_item:
                        api_see_also = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Note" in api_doc_formated_item:
                        api_notes = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Example".lower() in api_doc_formated_item.lower():
                        api_examples = normalize_text(api_doc_formated_item.split("******")[1])
                    else:
                        pass
                write_to_file(details_writer, [api_path, api_name, api_doc, api_signature, api_description, api_parameters, get_number_of_params(api_signature, library_name),\
                    api_returns, api_see_also, api_notes, api_examples])
        details_writer.close()
    elif library_name == "jieba":
        for line in tqdm(library_reader):
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                api_description, api_parameters, api_returns, api_see_also, api_notes, api_examples = "", "", "", "", "", ""
                if judge_none(api_doc):
                    continue
                api_doc_formated = api_doc
                for kw_line in all_key_words:
                    # pattern = r"[\s]*{}[\n]*[\s]+[-]".format(kw_line) + r"{2,}"
                    pattern = r"[\n]{1,}[\s]*" + r"{}:[\s]*".format(kw_line)
                    api_doc_formated = re.sub(pattern, f"######{kw_line}******", api_doc_formated, flags=re.I)
                api_doc_formated_list = api_doc_formated.split("######")
                if len(api_doc_formated_list) < 1:
                    continue

                for api_doc_formated_item in api_doc_formated_list:
                    if "******" not in api_doc_formated_item and len(api_doc_formated_item) > 0:
                        api_description = normalize_text(api_doc_formated_item)
                    elif "******" in api_doc_formated_item and "Arguments" in api_doc_formated_item:
                        api_parameters = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Returns" in api_doc_formated_item:
                        api_returns = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "See Also".lower() in api_doc_formated_item.lower():
                        api_see_also = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Note" in api_doc_formated_item:
                        api_notes = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Example".lower() in api_doc_formated_item.lower():
                        api_examples = normalize_text(api_doc_formated_item.split("******")[1])
                    else:
                        pass
                write_to_file(details_writer, [api_path, api_name, api_doc, api_signature, api_description, api_parameters, get_number_of_params(api_signature, library_name),\
                    api_returns, api_see_also, api_notes, api_examples])
        details_writer.close()
    elif library_name == "transformers":
        for line in tqdm(library_reader):
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                api_description, api_parameters, api_returns, api_see_also, api_notes, api_examples = "", "", "", "", "", ""
                if judge_none(api_doc):
                    continue
                api_doc_formated = api_doc
                for kw_line in all_key_words:
                    pattern = r"[\n]{2,}[\s]*" + r"{}:[\n]+".format(kw_line)
                    api_doc_formated = re.sub(pattern, f"######{kw_line}******", api_doc_formated, flags=re.I)
                api_doc_formated_list = api_doc_formated.split("######")
                if len(api_doc_formated_list) < 1:
                    continue
                for api_doc_formated_item in api_doc_formated_list:
                    if "******" not in api_doc_formated_item and len(api_doc_formated_item) > 0:
                        api_description = normalize_text(api_doc_formated_item)
                    elif "******" in api_doc_formated_item and ("Args" in api_doc_formated_item or "Arguments" in api_doc_formated_item or "Parameters" in api_doc_formated_item):
                        api_parameters += "\n"+normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Returns" in api_doc_formated_item:
                        api_returns = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "See Also".lower() in api_doc_formated_item.lower():
                        api_see_also = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Note" in api_doc_formated_item:
                        api_notes = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and ("Examples" in api_doc_formated_item or "Some examples" in api_doc_formated_item or\
                        "More examples" in api_doc_formated_item or "Example" in api_doc_formated_item):
                        api_examples += "\n"+normalize_text(api_doc_formated_item.split("******")[1])
                    else:
                        pass
                write_to_file(details_writer, [api_path, api_name, api_doc, api_signature, api_description, api_parameters, get_number_of_params(api_signature, library_name),\
                    api_returns, api_see_also, api_notes, api_examples])
        details_writer.close()
    elif library_name == "sqlalchemy":
        for line in tqdm(library_reader):
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                api_description, api_parameters, api_returns, api_see_also, api_notes, api_examples = "", "", "", "", "", ""
                if judge_none(api_doc):
                    continue
                api_doc_formated = api_doc
                for kw_line in all_key_words:
                    pattern = r"[\n]{2,}[\s]*" + r"{}[:]+[\n]+".format(kw_line)
                    api_doc_formated = re.sub(pattern, f"######{kw_line}******", api_doc_formated, flags=re.I)
                api_doc_formated_list = api_doc_formated.split("######")
                if len(api_doc_formated_list) < 1:
                    continue
                for api_doc_formated_item in api_doc_formated_list:
                    if "******" not in api_doc_formated_item and len(api_doc_formated_item) > 0:
                        api_description = normalize_text(api_doc_formated_item)
                    elif "******" in api_doc_formated_item and "Arguments" in api_doc_formated_item:
                        api_parameters = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Returns" in api_doc_formated_item:
                        api_returns = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "See Also".lower() in api_doc_formated_item.lower():
                        api_see_also = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Notes" in api_doc_formated_item:
                        api_notes = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Examples" in api_doc_formated_item:
                        api_examples = normalize_text(api_doc_formated_item.split("******")[1])
                    else:
                        pass
                write_to_file(details_writer, [api_path, api_name, api_doc, api_signature, api_description, api_parameters, get_number_of_params(api_signature, library_name),\
                    api_returns, api_see_also, api_notes, api_examples])
        details_writer.close()
    elif library_name == "allennlp":
        for line in tqdm(library_reader):
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                api_description, api_parameters, api_returns, api_see_also, api_notes, api_examples = "", "", "", "", "", ""
                if judge_none(api_doc):
                    continue
                api_doc_formated = api_doc
                for kw_line in all_key_words:
                    pattern = r"[\n]{2,}[\s]*" + r"{}[:]+[\n]+".format(kw_line)
                    api_doc_formated = re.sub(pattern, f"######{kw_line}******", api_doc_formated, flags=re.I)
                api_doc_formated_list = api_doc_formated.split("######")
                if len(api_doc_formated_list) < 1:
                    continue
                for api_doc_formated_item in api_doc_formated_list:
                    if "******" not in api_doc_formated_item and len(api_doc_formated_item) > 0:
                        api_description = normalize_text(api_doc_formated_item)
                    elif "******" in api_doc_formated_item and ("Arguments" in api_doc_formated_item or "Parameters" in api_doc_formated_item):
                        api_parameters += "\n"+normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and ("Return" in api_doc_formated_item or "Returns" in api_doc_formated_item):
                        api_returns += "\n"+normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "See Also".lower() in api_doc_formated_item.lower():
                        api_see_also = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Note" in api_doc_formated_item:
                        api_notes = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Example".lower() in api_doc_formated_item.lower():
                        api_examples = normalize_text(api_doc_formated_item.split("******")[1])
                    else:
                        pass
                write_to_file(details_writer, [api_path, api_name, api_doc, api_signature, api_description, api_parameters, get_number_of_params(api_signature, library_name),\
                    api_returns, api_see_also, api_notes, api_examples])
        details_writer.close()
    elif library_name == "django":
        for line in tqdm(library_reader):
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                api_description, api_parameters, api_returns, api_see_also, api_notes, api_examples = "", "", "", "", "", ""
                if judge_none(api_doc):
                    continue
                api_doc_formated = api_doc
                for kw_line in all_key_words:
                    pattern = r"[\n]{2,}[\s]*" + r"{}[:]+[\n]+".format(kw_line)
                    api_doc_formated = re.sub(pattern, f"######{kw_line}******", api_doc_formated, flags=re.I)
                api_doc_formated_list = api_doc_formated.split("######")
                if len(api_doc_formated_list) < 1:
                    continue
                for api_doc_formated_item in api_doc_formated_list:
                    if "******" not in api_doc_formated_item and len(api_doc_formated_item) > 0:
                        api_description = normalize_text(api_doc_formated_item)
                    elif "******" in api_doc_formated_item and "Arguments" in api_doc_formated_item:
                        api_parameters = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and ("Return" in api_doc_formated_item or "Returns" in api_doc_formated_item):
                        api_returns += "\n"+normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "See Also".lower() in api_doc_formated_item.lower():
                        api_see_also = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Note" in api_doc_formated_item:
                        api_notes = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Examples".lower() in api_doc_formated_item.lower():
                        api_examples = normalize_text(api_doc_formated_item.split("******")[1])
                    else:
                        pass
                write_to_file(details_writer, [api_path, api_name, api_doc, api_signature, api_description, api_parameters, get_number_of_params(api_signature, library_name),\
                    api_returns, api_see_also, api_notes, api_examples])
        details_writer.close()
    elif library_name == "flask":
        for line in tqdm(library_reader):
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                api_description, api_parameters, api_returns, api_see_also, api_notes, api_examples = "", "", "", "", "", ""
                if judge_none(api_doc):
                    continue
                api_doc_formated = api_doc
                for kw_line in all_key_words:
                    pattern = r"[\n]{2,}[\s]*" + r"{}[:]+[\n]+".format(kw_line)
                    api_doc_formated = re.sub(pattern, f"######{kw_line}******", api_doc_formated, flags=re.I)
                    api_doc_formated = re.sub(r"[\n]*[\s]*:param[\s]*", f"######Parameters******", api_doc_formated, flags=re.I)
                api_doc_formated_list = api_doc_formated.split("######")
                if len(api_doc_formated_list) < 1:
                    continue
                for api_doc_formated_item in api_doc_formated_list:
                    if "******" not in api_doc_formated_item and len(api_doc_formated_item) > 0:
                        api_description = normalize_text(api_doc_formated_item)
                    elif "******" in api_doc_formated_item and "Parameters" in api_doc_formated_item:
                        api_parameters += "\n"+normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and ("Return" in api_doc_formated_item or "Returns" in api_doc_formated_item):
                        api_returns += "\n"+normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "See Also".lower() in api_doc_formated_item.lower():
                        api_see_also = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Note" in api_doc_formated_item:
                        api_notes = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Examples".lower() in api_doc_formated_item.lower():
                        api_examples = normalize_text(api_doc_formated_item.split("******")[1])
                    else:
                        pass
                write_to_file(details_writer, [api_path, api_name, api_doc, api_signature, api_description, api_parameters, get_number_of_params(api_signature, library_name),\
                    api_returns, api_see_also, api_notes, api_examples])
        details_writer.close()
    elif library_name == "PIL":
        for line in tqdm(library_reader):
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                api_description, api_parameters, api_returns, api_see_also, api_notes, api_examples = "", "", "", "", "", ""
                if judge_none(api_doc):
                    continue
                api_doc_formated = api_doc
                for kw_line in all_key_words:
                    pattern = r"[\n]+[\s]*" + r"{}[:]+[\n]*".format(kw_line)
                    api_doc_formated = re.sub(pattern, f"######{kw_line}******", api_doc_formated, flags=re.I)
                    api_doc_formated = re.sub(r"[\n]*[\s]*:param[\s]*", f"######Parameters******", api_doc_formated, flags=re.I)
                api_doc_formated_list = api_doc_formated.split("######")
                if len(api_doc_formated_list) < 1:
                    continue
                for api_doc_formated_item in api_doc_formated_list:
                    if "******" not in api_doc_formated_item and len(api_doc_formated_item) > 0:
                        api_description = normalize_text(api_doc_formated_item)
                    elif "******" in api_doc_formated_item and "Parameters" in api_doc_formated_item:
                        api_parameters += "\n"+normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Return" in api_doc_formated_item:
                        api_returns = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "See".lower() in api_doc_formated_item.lower():
                        api_see_also = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Note".lower() in api_doc_formated_item.lower():
                        api_notes = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Example".lower() in api_doc_formated_item.lower():
                        api_examples = normalize_text(api_doc_formated_item.split("******")[1])
                    else:
                        pass
                write_to_file(details_writer, [api_path, api_name, api_doc, api_signature, api_description, api_parameters, get_number_of_params(api_signature, library_name),\
                    api_returns, api_see_also, api_notes, api_examples])
        details_writer.close()
    elif library_name == "scrapy":
        for line in tqdm(library_reader):
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                api_description, api_parameters, api_returns, api_see_also, api_notes, api_examples = "", "", "", "", "", ""
                if judge_none(api_doc):
                    continue
                api_doc_formated = api_doc
                for kw_line in all_key_words:
                    pattern = r"[\n]+[\s]*" + r"{}[:]+[\n]*".format(kw_line)
                    api_doc_formated = re.sub(pattern, f"######{kw_line}******", api_doc_formated, flags=re.I)
                    api_doc_formated = re.sub(r"[\n]*[\s]*:param[\s]*", f"######Parameters******", api_doc_formated, flags=re.I)
                api_doc_formated_list = api_doc_formated.split("######")
                if len(api_doc_formated_list) < 1:
                    continue
                for api_doc_formated_item in api_doc_formated_list:
                    if "******" not in api_doc_formated_item and len(api_doc_formated_item) > 0:
                        api_description = normalize_text(api_doc_formated_item)
                    elif "******" in api_doc_formated_item and "Parameters" in api_doc_formated_item:
                        api_parameters += "\n"+normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Return" in api_doc_formated_item:
                        api_returns = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "See".lower() in api_doc_formated_item.lower():
                        api_see_also = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Note".lower() in api_doc_formated_item.lower():
                        api_notes = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Example".lower() in api_doc_formated_item.lower():
                        api_examples = normalize_text(api_doc_formated_item.split("******")[1])
                    else:
                        pass
                write_to_file(details_writer, [api_path, api_name, api_doc, api_signature, api_description, api_parameters, get_number_of_params(api_signature, library_name),\
                    api_returns, api_see_also, api_notes, api_examples])
        details_writer.close()
    elif library_name == "tokenizers":
        for line in tqdm(library_reader):
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                api_description, api_parameters, api_returns, api_see_also, api_notes, api_examples = "", "", "", "", "", ""
                if judge_none(api_doc):
                    continue
                api_doc_formated = api_doc
                for kw_line in all_key_words:
                    pattern = r"[\n]+[\s]*" + r"{}[:]+[\n]*".format(kw_line)
                    api_doc_formated = re.sub(pattern, f"######{kw_line}******", api_doc_formated, flags=re.I)
                    api_doc_formated = re.sub(r"[\n]*[\s]*:param[\s]*", f"######Parameters******", api_doc_formated, flags=re.I)
                api_doc_formated_list = api_doc_formated.split("######")
                if len(api_doc_formated_list) < 1:
                    continue
                for api_doc_formated_item in api_doc_formated_list:
                    if "******" not in api_doc_formated_item and len(api_doc_formated_item) > 0:
                        api_description = normalize_text(api_doc_formated_item)
                    elif "******" in api_doc_formated_item and ("Parameters" in api_doc_formated_item or "argument" in api_doc_formated_item):
                        api_parameters += "\n"+normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Return" in api_doc_formated_item:
                        api_returns = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "See".lower() in api_doc_formated_item.lower():
                        api_see_also = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Note".lower() in api_doc_formated_item.lower():
                        api_notes = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Example".lower() in api_doc_formated_item.lower():
                        api_examples = normalize_text(api_doc_formated_item.split("******")[1])
                    else:
                        pass
                write_to_file(details_writer, [api_path, api_name, api_doc, api_signature, api_description, api_parameters, get_number_of_params(api_signature, library_name),\
                    api_returns, api_see_also, api_notes, api_examples])
        details_writer.close()
    elif library_name == "mxnet":
        for line in tqdm(library_reader):
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                api_description, api_parameters, api_returns, api_see_also, api_notes, api_examples = "", "", "", "", "", ""
                if judge_none(api_doc):
                    continue
                api_doc_formated = api_doc
                for kw_line in all_key_words:
                    pattern = r"[\s]*{}[\n]*[\s]+[-]".format(kw_line) + r"{2,}"
                    api_doc_formated = re.sub(pattern, f"######{kw_line}******", api_doc_formated, flags=re.I)
                api_doc_formated_list = api_doc_formated.split("######")
                if len(api_doc_formated_list) < 1:
                    continue
                for api_doc_formated_item in api_doc_formated_list:
                    if "******" not in api_doc_formated_item and len(api_doc_formated_item) > 0:
                        api_description = normalize_text(api_doc_formated_item)
                    elif "******" in api_doc_formated_item and "Parameters" in api_doc_formated_item:
                        api_parameters = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Returns" in api_doc_formated_item:
                        api_returns = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "See also".lower() in api_doc_formated_item.lower():
                        api_see_also = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Notes".lower() in api_doc_formated_item.lower():
                        api_notes = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Example".lower() in api_doc_formated_item.lower():
                        api_examples = normalize_text(api_doc_formated_item.split("******")[1])
                    else:
                        pass
                write_to_file(details_writer, [api_path, api_name, api_doc, api_signature, api_description, api_parameters, get_number_of_params(api_signature, library_name),\
                    api_returns, api_see_also, api_notes, api_examples])
        details_writer.close()
    elif library_name == "imageio":
        for line in tqdm(library_reader):
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                api_description, api_parameters, api_returns, api_see_also, api_notes, api_examples = "", "", "", "", "", ""
                if judge_none(api_doc):
                    continue
                api_doc_formated = api_doc
                for kw_line in all_key_words:
                    pattern = r"[\s]*{}[\n]*[\s]+[-]".format(kw_line) + r"{2,}"
                    api_doc_formated = re.sub(pattern, f"######{kw_line}******", api_doc_formated, flags=re.I)
                api_doc_formated_list = api_doc_formated.split("######")
                if len(api_doc_formated_list) < 1:
                    continue
                for api_doc_formated_item in api_doc_formated_list:
                    if "******" not in api_doc_formated_item and len(api_doc_formated_item) > 0:
                        api_description = normalize_text(api_doc_formated_item)
                    elif "******" in api_doc_formated_item and "Parameters" in api_doc_formated_item:
                        api_parameters = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Returns" in api_doc_formated_item:
                        api_returns = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "See also".lower() in api_doc_formated_item.lower():
                        api_see_also = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Notes".lower() in api_doc_formated_item.lower():
                        api_notes = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Examples".lower() in api_doc_formated_item.lower():
                        api_examples = normalize_text(api_doc_formated_item.split("******")[1])
                    else:
                        pass
                write_to_file(details_writer, [api_path, api_name, api_doc, api_signature, api_description, api_parameters, get_number_of_params(api_signature, library_name),\
                    api_returns, api_see_also, api_notes, api_examples])
        details_writer.close()
    elif library_name == "pytest":
        for line in tqdm(library_reader):
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                api_description, api_parameters, api_returns, api_see_also, api_notes, api_examples = "", "", "", "", "", ""
                if judge_none(api_doc):
                    continue
                api_doc_formated = api_doc
                for kw_line in all_key_words:
                    pattern = r"[\n]+[\s]*{}:[\n]+".format(kw_line)
                    api_doc_formated = re.sub(pattern, f"######{kw_line}******", api_doc_formated, flags=re.I)
                    api_doc_formated = re.sub(r"[\n]*[\s]*:param[\s]*", f"######Parameters******", api_doc_formated, flags=re.I)
                api_doc_formated_list = api_doc_formated.split("######")
                if len(api_doc_formated_list) < 1:
                    continue
                for api_doc_formated_item in api_doc_formated_list:
                    if "******" not in api_doc_formated_item and len(api_doc_formated_item) > 0:
                        api_description = normalize_text(api_doc_formated_item.replace(":meta private:", ""))
                    elif "******" in api_doc_formated_item and ("Attributes" in api_doc_formated_item or "Parameters" in api_doc_formated_item):
                        api_parameters += "\n"+normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Returns" in api_doc_formated_item:
                        api_returns = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "See also".lower() in api_doc_formated_item.lower():
                        api_see_also = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Notes".lower() in api_doc_formated_item.lower():
                        api_notes = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Example".lower() in api_doc_formated_item.lower():
                        api_examples = normalize_text(api_doc_formated_item.split("******")[1])
                    else:
                        pass
                write_to_file(details_writer, [api_path, api_name, api_doc, api_signature, api_description, api_parameters, get_number_of_params(api_signature, library_name),\
                    api_returns, api_see_also, api_notes, api_examples])
        details_writer.close()
    elif library_name == "metpy":
        for line in tqdm(library_reader):
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                api_description, api_parameters, api_returns, api_see_also, api_notes, api_examples = "", "", "", "", "", ""
                if judge_none(api_doc):
                    continue
                api_doc_formated = api_doc
                for kw_line in all_key_words:
                    pattern = r"[\s]*{}[\n]*[\s]+[-]".format(kw_line) + r"{2,}"
                    api_doc_formated = re.sub(pattern, f"######{kw_line}******", api_doc_formated, flags=re.I)
                api_doc_formated_list = api_doc_formated.split("######")
                if len(api_doc_formated_list) < 1:
                    continue
                for api_doc_formated_item in api_doc_formated_list:
                    if "******" not in api_doc_formated_item and len(api_doc_formated_item) > 0:
                        api_description = normalize_text(api_doc_formated_item)
                    elif "******" in api_doc_formated_item and "Parameters" in api_doc_formated_item:
                        api_parameters = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Returns" in api_doc_formated_item:
                        api_returns = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "See also".lower() in api_doc_formated_item.lower():
                        api_see_also = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Notes".lower() in api_doc_formated_item.lower():
                        api_notes = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Examples".lower() in api_doc_formated_item.lower():
                        api_examples = normalize_text(api_doc_formated_item.split("******")[1])
                    else:
                        pass
                write_to_file(details_writer, [api_path, api_name, api_doc, api_signature, api_description, api_parameters, get_number_of_params(api_signature, library_name),\
                    api_returns, api_see_also, api_notes, api_examples])
        details_writer.close()
    elif library_name == "ansible":
        for line in tqdm(library_reader):
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                api_description, api_parameters, api_returns, api_see_also, api_notes, api_examples = "", "", "", "", "", ""
                if judge_none(api_doc):
                    continue
                api_doc_formated = api_doc
                for kw_line in all_key_words:
                    pattern = r"[\n]+[\s]*{}:[\n]+".format(kw_line)
                    api_doc_formated = re.sub(pattern, f"######{kw_line}******", api_doc_formated, flags=re.I)
                    api_doc_formated = re.sub(r"[\n]*[\s]*:param[\s]*", f"######Parameters******", api_doc_formated, flags=re.I)
                api_doc_formated_list = api_doc_formated.split("######")
                if len(api_doc_formated_list) < 1:
                    continue
                for api_doc_formated_item in api_doc_formated_list:
                    if "******" not in api_doc_formated_item and len(api_doc_formated_item) > 0:
                        api_description = normalize_text(api_doc_formated_item.replace(":meta private:", ""))
                    elif "******" in api_doc_formated_item and "Parameters" in api_doc_formated_item:
                        api_parameters += "\n"+normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Returns" in api_doc_formated_item:
                        api_returns = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "See also".lower() in api_doc_formated_item.lower():
                        api_see_also = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Notes".lower() in api_doc_formated_item.lower():
                        api_notes = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Example".lower() in api_doc_formated_item.lower():
                        api_examples = normalize_text(api_doc_formated_item.split("******")[1])
                    else:
                        pass
                write_to_file(details_writer, [api_path, api_name, api_doc, api_signature, api_description, api_parameters, get_number_of_params(api_signature, library_name),\
                    api_returns, api_see_also, api_notes, api_examples])
        details_writer.close()
    elif library_name == "requests":
        for line in tqdm(library_reader):
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                api_description, api_parameters, api_returns, api_see_also, api_notes, api_examples = "", "", "", "", "", ""
                if judge_none(api_doc):
                    continue
                api_doc_formated = api_doc
                for kw_line in all_key_words:
                    pattern = r"[\n]+[\s]*{}:[\n]+".format(kw_line)
                    api_doc_formated = re.sub(pattern, f"######{kw_line}******", api_doc_formated, flags=re.I)
                    api_doc_formated = re.sub(r"[\n]*[\s]*:param[\s]*", f"######Parameters******", api_doc_formated, flags=re.I)
                api_doc_formated_list = api_doc_formated.split("######")
                if len(api_doc_formated_list) < 1:
                    continue
                for api_doc_formated_item in api_doc_formated_list:
                    if "******" not in api_doc_formated_item and len(api_doc_formated_item) > 0:
                        api_description = normalize_text(api_doc_formated_item)
                    elif "******" in api_doc_formated_item and ("Parameters" in api_doc_formated_item or "Arguments" in api_doc_formated_item):
                        api_parameters += "\n"+normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Returns" in api_doc_formated_item:
                        api_returns = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "See also".lower() in api_doc_formated_item.lower():
                        api_see_also = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Notes".lower() in api_doc_formated_item.lower():
                        api_notes = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Example".lower() in api_doc_formated_item.lower():
                        api_examples = normalize_text(api_doc_formated_item.split("******")[1])
                    else:
                        pass
                write_to_file(details_writer, [api_path, api_name, api_doc, api_signature, api_description, api_parameters, get_number_of_params(api_signature, library_name),\
                    api_returns, api_see_also, api_notes, api_examples])
        details_writer.close()
    elif library_name == "datetime" or library_name == "zlib" or library_name == "random" or library_name == "math" or library_name == "sys" or library_name == "glob"\
        or library_name == "os" or library_name == "urllib" or library_name == "uuid" or library_name == "pprint" or library_name == "time" or library_name == "re"\
        or library_name == "json" or library_name == "unittest" or library_name == "collections" or library_name == "subprocess" or library_name == "copy"\
        or library_name == "functools" or library_name == "six" or library_name == "itertools" or library_name == "threading" or library_name == "tempfile"\
        or library_name == "io" or library_name == "pickle" or library_name == "pathlib" or library_name == "socket" or library_name == "struct" or library_name == "hashlib"\
        or library_name == "traceback" or library_name == "csv":
        for line in tqdm(library_reader):
            line_dict = eval(line)
            for api_path, [api_doc, api_name, api_signature] in line_dict.items():
                api_description, api_parameters, api_returns, api_see_also, api_notes, api_examples = "", "", "", "", "", ""
                if judge_none(api_doc):
                    continue
                api_doc_formated = api_doc
                for kw_line in all_key_words:
                    pattern = r"[\n]{2}[\s]*"+r"{}:".format(kw_line)
                    api_doc_formated = re.sub(pattern, f"######{kw_line}******", api_doc_formated, flags=re.I)
                    api_doc_formated = re.sub(r"[\n]*>>>", f"######Parameters******", api_doc_formated, flags=re.I)
                api_doc_formated_list = api_doc_formated.split("######")
                if len(api_doc_formated_list) < 1:
                    continue
                for api_doc_formated_item in api_doc_formated_list:
                    if "******" not in api_doc_formated_item and len(api_doc_formated_item) > 0:
                        api_description = normalize_text(api_doc_formated_item)
                    elif "******" in api_doc_formated_item and "Parameters" in api_doc_formated_item:
                        api_parameters += "\n"+normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Returns" in api_doc_formated_item:
                        api_returns = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "See also".lower() in api_doc_formated_item.lower():
                        api_see_also = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Notes".lower() in api_doc_formated_item.lower():
                        api_notes = normalize_text(api_doc_formated_item.split("******")[1])
                    elif "******" in api_doc_formated_item and "Example".lower() in api_doc_formated_item.lower():
                        api_examples = normalize_text(api_doc_formated_item.split("******")[1])
                    else:
                        pass
                write_to_file(details_writer, [api_path, api_name, api_doc, api_signature, api_description, api_parameters, get_number_of_params(api_signature, library_name),\
                    api_returns, api_see_also, api_notes, api_examples])
        details_writer.close()
    else:
        raise NotImplementedError(f"{library_name} not implemented")
    pass

def extract_details_for_one_library(
    output_dir,
    library_names,
    process_num
    ):
    """
    extract details for one library
    """
    for library_name in tqdm(library_names.split(",")):
        library_path = os.path.join(output_dir, library_name, f"{library_name}_apis_doc_True.jsonl") # True means the signature is included.
        if not os.path.exists(library_path):
            raise FileNotFoundError(f"{library_path} not found")

        details_output_path = os.path.join(output_dir, library_name, f"{library_name}_apis_doc_details.jsonl")
        if os.path.exists(details_output_path):
            logging.info(f"{details_output_path} already exists, we maybe conver it.")
        details_writer = open(details_output_path, "w+")

        all_key_words = get_key_words(library_path, library_name)
        # ipdb.set_trace()
        get_details(library_path, all_key_words, library_name, details_writer)
        pass

        logging.info(f"{library_name} done.")

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract details from apis, like the function name, description, and parameters...")

    parser.add_argument("-o", "--output_dir", type=str, help="The output directory.")
    parser.add_argument("-l", "--library", type=str, help="The library name.")
    parser.add_argument("-pn", "--process_num", type=int, default=1, help="The number of processes to run in parallel.")

    args = parser.parse_args()
    logging.info(f"the args are: {args}")

    extract_details_for_one_library(
        args.output_dir, 
        args.library, 
        args.process_num
    )

    pass