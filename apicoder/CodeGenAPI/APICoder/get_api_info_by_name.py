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

def get_api_name_4_api_sign_and_desps(library_name: str, base_dir: str):
    """
    According to library_name, get all the API info of this library in the format shown in the following format.
    """
    # load the library_name's all api info
    # base_dir = "/mnt/v-dzan/datasets/CERT/PrivateLibrary/Train"
    base_dir = os.path.join(base_dir, "PrivateLibrary", "Train")
    library_path = os.path.join(base_dir, library_name, f"{library_name}_apis_doc_details.jsonl")

    library_apis_reader = open(library_path, "r")
    api_name_4_api_sign_and_desps = {}
    # The api_name_4_api_sign_and_desps format is:
    # {
    #    "api_name": {
    #       api_path1: [api_sign1, api_desp1],
    #       api_path2: [api_sign2, api_desp2],
    #       ...
    #   }
    #   ...
    # }
    for line in library_apis_reader:
        api_info = json.loads(line)
        # (['api_path', 'api_name', 'api_doc', 'api_signature', 'api_description', 'api_parameters', 'api_parameters_number', 'api_returns', 'api_see_also', 'api_notes', 'api_examples'])
        api_path = api_info["api_path"]
        api_name = api_info["api_name"]
        api_signature = api_info["api_signature"]
        api_description = api_info["api_description"]
        tmp_api_path_api_info = {api_path: [api_signature, api_description]}
        if api_name_4_api_sign_and_desps.get(api_name) is None:
            api_name_4_api_sign_and_desps[api_name] = tmp_api_path_api_info
        else:
            api_name_4_api_sign_and_desps[api_name] = dict(api_name_4_api_sign_and_desps[api_name], **tmp_api_path_api_info)

    library_apis_reader.close()
    return api_name_4_api_sign_and_desps

def get_all_api_info_prompt_list_by_api_name(api_name_4_api_sign_and_desps, API_NAME):
    """
    Get a dictionary of all {API_path: API_signature, API_description} based on the name of the API
    """
    import sys
    from scripts.get_libs_info_from_code import (
        normalizer_api_desp,
        get_first_sentence_from_api_desp
    )

    result_api_path_info_dict = dict()
    for api_name, api_path_info_dict in api_name_4_api_sign_and_desps.items():
        if api_name == API_NAME:
            for api_path, api_info_list in api_path_info_dict.items():
                api_signature, api_description = api_info_list[0], get_first_sentence_from_api_desp(normalizer_api_desp(api_info_list[1]))

                result_api_path_info_dict[api_path] = [api_signature, api_description]
            break
    return result_api_path_info_dict
