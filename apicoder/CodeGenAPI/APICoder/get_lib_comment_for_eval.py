#!/usr/bin/env python
# coding=utf-8
# 
# @Author: Daoguang Zan, @Mentor: Bei Chen, Jian-Guang Lou
# @Copyright 2022 The Microsoft Research Asia (DKI Group). All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List
import json
import gzip
import os
import sys
sys.path.append("..")
from scripts.get_comments_from_evallibs import get_comments_from_code
# remove the sys path ".." to avoid the conflict with the other scripts
sys.path.remove("..")

def get_one_instance_by_lib_name(library_name: str, base_dir: str):
    """
    Get an iterative object based on lib_name
    """
    base_dir = os.path.join(base_dir, "eval_datas")
    library_path = os.path.join(base_dir, f"real_{library_name}_eval_v2.jsonl.gz")

    library_reader = gzip.open(library_path, "rb")
    for line in library_reader:
        line = line.decode("utf-8")
        line_dict = json.loads(line)
        yield line_dict

def get_code_and_comment_by_lib_name_and_task_id(
    library_name: str,
    query_task_id: str,
    base_dir: str
):
    """
    Get code, comments and solutions based on lib_name and task_id.
    """
    # base_dir = f"/mnt/v-dzan/datasets/CERT/eval_datas"
    base_dir = os.path.join(base_dir, "eval_datas")
    library_path = os.path.join(base_dir, f"real_{library_name}_eval_v3.jsonl.gz")

    library_reader = gzip.open(library_path, "rb")
    for line in library_reader:
        line = line.decode("utf-8")
        line_dict = json.loads(line)
        task_id = line_dict["task_id"]
        if task_id == query_task_id:
            code = line_dict["prompt"]
            solution = line_dict["canonical_solution"][0]
            code_comment = get_comments_from_code(code)
            library_reader.close()
            return [code, code_comment, solution]
    
    library_reader.close()
    return ["", "", ""]


if __name__ == "__main__":
    print("Passed!")
    pass