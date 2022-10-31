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
import json
import os
import sys
import re

from get_libs_info_from_code import (
    normalizer_api_desp,
    get_first_sentence_from_api_desp,
    extract_main_comment_from_code
)

def judge_is_what_type_annotation(code: str) -> str:
    type = ["pound", "inverted commas"] # pound: #, inverted commas: """
    if "#" in code:
        return type[0]
    else:
        return type[1]

def get_comments_from_code(code: str) -> str:
    """
    Get comments from code.
    ---
    Args:
        Code: raw code from PandasEval, NumpyEval, etc.
    Returns:
        Comments: comments from code.
    """
    comment_type = judge_is_what_type_annotation(code)
    if comment_type == "pound":
        code_splited = code.split("\n")
        code_comment_str = ""
        for line in code_splited:
            if "#" in line:
                code_comment_str += " " + line.replace("#", "").strip() if code_comment_str != "" else line.replace("#", "").strip()
        return normalizer_api_desp(code_comment_str)
    else:
        return normalizer_api_desp(extract_main_comment_from_code(code)).replace("\"\"\"", '').replace("\'\'\'", '').strip()


if __name__ == '__main__':
    pass