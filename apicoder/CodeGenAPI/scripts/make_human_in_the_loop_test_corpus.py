import json
import os
import pandas as pd
import gzip
from tqdm import tqdm

ranking_base_dir = "APIRetriever/data/inference"
output_base_dir = "private-eval/data"

library_names = [
    "pandas", 
    "numpy",
    "monkey", 
    "beatnum", 
    "torchdata"
]

hm_api_nums = [5]

map_source_to_target = {
    "pandas": "PandasEval", 
    "numpy": "NumpyEval",
    "monkey": "PandasEval",
    "beatnum": "NumpyEval", 
    "torchdata": "TorchDataEval"
}

def remove_api_sign(api_info):
    start_idx, end_idx = api_info.find("("), 0
    flag = False
    for idx, ch in enumerate(api_info):
        if ch == ")":
            flag = True
        if flag and ch == ":":
            end_idx = idx
            break
    return api_info[:start_idx]+api_info[end_idx:]

def get_raw_prompt_by_task_id(base_dir, library, full_task_id):
    lib_path = os.path.join(base_dir, f"real_{library}_eval_v3.jsonl.gz")
    lib_reader = gzip.open(lib_path, "rb")
    for line in lib_reader:
        line = line.decode("utf-8")
        line = json.loads(line)
        if line["task_id"] == full_task_id:
            return line["prompt"]
    return None

def get_api_name_by_api_info(api_info: str) -> str:
    return api_info[:api_info.index("(")]

def get_ranking_apis(full_task_id, taskid2commentid, commentid2apiids, apiid2apiinfo):
    comment_id = taskid2commentid[full_task_id]
    api_ids = commentid2apiids[comment_id]
    ranking_api_infos, ranking_api_names = [], []
    for api_id in api_ids:
        api_info = apiid2apiinfo[api_id]
        its_api_name = get_api_name_by_api_info(api_info)
        if its_api_name in ranking_api_names:
            continue
        ranking_api_infos.append(api_info)
        ranking_api_names.append(its_api_name)
    return ranking_api_infos


def get_task_id_dict(base_retriever_dir, library, task_id_int):
    """
    return a dict, which contains keys: ranking_apis, prompt, task_id
    """
    # task_id -> comment_id -> api_ids -> api_infos
    taskid2commentid = {} # task_id -> comment_id
    commentid2apiids = {}
    apiid2apiname = {}
    apiid2apiinfo = {}

    # task_id -> comment_id
    comment_path = os.path.join(base_retriever_dir, f"{library}_comment.json")
    comment_file_reader = open(comment_path, "r")
    for line in comment_file_reader:
        line_dict = json.loads(line)
        taskid2commentid[line_dict["task_id"]] = line_dict["text_id"]
    comment_file_reader.close()

    # comment_id -> api_ids
    data_scores_path = os.path.join(base_retriever_dir, f"{library}_id_score.trec")
    data_scores_file_reader = open(data_scores_path, "r")
    for line in data_scores_file_reader:
        line_list = line.strip().split()
        assert len(line_list) == 6
        comment_id = int(line_list[0])
        api_id = int(line_list[2])
        if commentid2apiids.get(comment_id) == None:
            commentid2apiids[comment_id] = []
        commentid2apiids[comment_id].append(api_id)
    data_scores_file_reader.close()

    # apiid2apiinfo
    api_path = os.path.join(base_retriever_dir, f"{library}_api.json")
    api_file_reader = open(api_path, "r")
    for line in api_file_reader:
        line_dict = json.loads(line)
        # apiid2apiname[line_dict["text_id"]] = line_dict["text"].strip()[:line_dict["text"].strip().index("(")]
        apiid2apiinfo[line_dict["text_id"]] = line_dict["text"]
    api_file_reader.close()

    full_task_id = map_source_to_target[library]+"/"+str(task_id_int)
    prompt = get_raw_prompt_by_task_id("/mnt/v-dzan/datasets/CERT/eval_datas", library, full_task_id)
    ranking_apis = get_ranking_apis(full_task_id, taskid2commentid, commentid2apiids, apiid2apiinfo)

    return full_task_id, prompt, ranking_apis



for library_name in tqdm(library_names):
    for hm_api_num in tqdm(hm_api_nums):
        ranking_file = os.path.join(ranking_base_dir, f"{library_name}_comment.json")
        output_hm_file = os.path.join(output_base_dir, f"v3_human_test_{library_name}_{hm_api_num}_apis.txt")

        ranking_len = len(open(ranking_file, "r").readlines())
        output_hm_writer = open(output_hm_file, "w+")

        test_header_prompt = f"""========================================================================================
                               {library_name.upper()} Test, Total {ranking_len} tasks 
========================================================================================
"""
        off_lib_name, off_lib_desp = "", ""
        if library_name == "pandas":
            off_lib_name = "Pandas"
            off_lib_desp = "Pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language."
        elif library_name == "numpy":
            off_lib_name = "Numpy"
            off_lib_desp = "Numpy is a fundamental package for scientific computing with Python."
        elif library_name == "monkey":
            off_lib_name = "Monkey"
            off_lib_desp = "Monkey is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language."
        elif library_name == "beatnum":
            off_lib_name = "BeatNum"
            off_lib_desp = "BeatNum is a fundamental package for scientific computing with Python."
        elif library_name == "torchdata":
            off_lib_name = "TorchData"
            off_lib_desp = "TorchData is a prototype library of common modular data loading primitives for easily constructing flexible and performant data pipelines."
        else:
            raise ValueError(f"{library_name} is not supported")

        guide_line_prompt = f"""
========================================================================================
You are a programmer writing code with a private library called [{off_lib_name}]. 
{off_lib_desp}
Now, we have a tool to help you which can automatically generate code with TorchData's APIs. 
Assume that the [given code prompt] is written by you, and then the tool provides 4/5 API candidates for you to choose from. 
Which APIs would you like to use?



An example:
Given Code Prompt:
>>> from torchdata.datapipes.iter import IterableWrapper
>>> dp = IterableWrapper(range(10))
>>> # I would like to shuffle the datapipe obj, how can I do that?
>>> new_dp =[Model Prediction Placeholder]


Chioces: ([choice]: API Name: API Description)
[1]: cycle: Cycles the specified input in perpetuity by default, or for the specified number of times.
[2]: suffle: Shuffles the input DataPipe with a buffer.
[3]: JsonParser: Reads from JSON data streams and yields a tuple of file name and JSON data (functional name: ``parse_json_files``).


Your Choices: # According to the comment in Given Code Prompt, we find from Choices that [2] may be suitable.
[2]


Hints:
[1]. It is a multiple choice question, but you just have to pick the ones you are pretty sure about. Not the more you choose the better.
[2]. In general, you only need to refer to the API Name. API description is just a supplement, and you can check it when the information of API name is not enough.
[3]. If you would like more than one choice, please use # to separate them. For example if you want to choose 1,3 as your answer, write [1#3] under Your Choices.
[4]. If you think there is no any API in the recommended choices, please write [0] under Your Choices.
[5]. If you don't know about the question, please write [?] under Your Choices.
========================================================================================


                                   Let's start!

"""
        output_hm_writer.write(test_header_prompt + guide_line_prompt)

        for i in range(ranking_len):
            task_id, prompt, ranking_apis = get_task_id_dict(ranking_base_dir, library_name, i)
            if "# [end]" in prompt:
                prompt = prompt.split("# [end]\n")[1]
            else:
                prompt = prompt

            prompt = ">>> " + prompt.replace("\n", "\n>>> ") + "[Model Prediction Placeholder]"

            writed_prompt = f"""========================================================================================
Task ID: {task_id}


Given Code Prompt:
{prompt}


# Which APIs would you like to use?


Chioces:
"""
            choices = "\n".join([f"[{idx+1}]: {remove_api_sign(api_info)}" for idx, (api_info) in enumerate(ranking_apis[:hm_api_num])])
            writed_prompt += choices
            writed_prompt += f"""


Your Choices:
[]
========================================================================================


"""
            output_hm_writer.write(writed_prompt)
            
        output_hm_writer.close()

print("All Done!")