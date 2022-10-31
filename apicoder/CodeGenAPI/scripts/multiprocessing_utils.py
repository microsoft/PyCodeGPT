import os
import json
from re import S
import time
import traceback
from typing import List, Callable, Dict
from collections import defaultdict
from dataclasses import dataclass
import logging

import multiprocessing
import multiprocessing.dummy

def get_logger(path: str=None) -> logging.Logger:
    from imp import reload # python 2.x don't need to import reload, use it directly
    reload(logging)

    handlers = [logging.StreamHandler()]
    if path is not None:
        handlers.append(logging.FileHandler(path))

    logging.basicConfig(
        level=logging.INFO,
        force=True,
        format='%(asctime)s\t%(levelname)  -8s %(message)s',
        datefmt='%m-%d %H:%M:%S',
        handlers=handlers
    )

    logger = logging.getLogger(__name__)
    return logger

def save_result_to_json(result: Dict, path: str):
    with open(path, 'w', encoding='utf-8') as fw:
        fw.write(json.dumps(result, indent=4) + '\n')
    pass

def load_result_from_json(path: str):
    with open(path, 'r', encoding='utf-8') as fr:
        return json.loads(fr.read())

def convert_result_to_string(result: Dict[str, int]):
    return "; ".join(["{} = {}".format(k, v) for k, v in result.items()])

@dataclass
class MultiprocessingArgs:
    process_idx: int = -1

    run_name: str = ''

    inputs: List[str] = None

    output_dir: str = None
    
    domain: str = None

    lib_name_api_name_api_paths: Dict[str, Dict[str, str]] = None

    api_path_api_signature_and_api_desp: Dict[str, Dict[str, str]] = None

    type_name: str = None

    is_debug: str = "False"

    perturbation_probability: float = 0.0

    stype: str = None
    
    logging_freq: int = 10000

    logger: logging.Logger = None

    processs_size: int = None
    node_idx: int = None
    node_size: int = None

    kwargs: Dict[str, object] = None
    
    def __post__init__(self):
        if self.logger is None:
            self.logger = get_logger()

        if self.kwargs is None:
            self.kwargs = {}
    
    @property
    def process_name(self) -> str:
        if self.node_idx is None:
            return "Process {}".format(self.process_idx)        
        else:
            return "Process {}({}/{})".format(self.process_idx, self.node_idx, self.node_size)

class MultiprocessingResult(dict):
    def update_item(self, key, val):
        if key not in self:
            self[key] = val
        else:
            self[key] += val
    
    def merge(self, other):
        for k, v in other.items():
            self.update_item(k, v)
    
    def save_to_json(self, path):
        for key, val in self.items():
            if isinstance(val, list):
                self[key] = list(sorted(val))
        save_result_to_json(self, path)
    
    @staticmethod
    def load_from_json(cls, path):
        result = load_result_from_json(path)
        return cls({ k : v for k, v in result.items() })

def merge_results_from_distributed_nodes(node_size: int, output_dir: str, run_name: str):
    all_result = MultiprocessingResult()
    merged_nodes = set([])
    while True:
        for node_idx in range(node_size):
            if node_idx in merged_nodes:
                continue

            result_path = os.path.join(output_dir, "{}.node_{}.result.json".format(run_name, node_idx))
            if not os.path.exists(result_path):
                continue

            result = load_result_from_json(result_path)
            all_result.merge(result)
            all_result.update_item('nodes', 1)

            merged_nodes.add(node_idx)

        if len(merged_nodes) == node_size:
            break
        
        logging.info("Merged {}/{} results, sleep 3 mins ...".format(len(merged_nodes), node_size))
        time.sleep(3 * 60)
    
    logging.info("Merge all results of {} from {} nodes over, {}.".format(run_name, node_size, convert_result_to_string(all_result)))

    all_result.save_to_json(os.path.join(output_dir, "{}.node_all.result.json".format(run_name)))
    logging.info("Save all merged result over!")
    pass

def run_with_multiprocessing(
    run_name: str = '',
    process_func: Callable = None,
    inputs: str = "",
    output_dir: str = "",
    domain: str = "",
    lib_name_api_name_api_paths: Dict[str, Dict[str, str]] = None,
    api_path_api_signature_and_api_desp: Dict[str, Dict[str, str]] = None,
    type_name: str = "",
    is_debug: str = "False",
    num_processes: int = 20,
    perturbation_probability: float = 0.0,
    stype: str = "v2",
    logging_freq: int = 10000,
    **kwargs,
):
    # detect if run on distribtued nodes
    node_idx, node_size = None, None
    if 'DistributedNodes' in os.environ:
        node_idx, node_size = map(int, os.environ['DistributedNodes'].split('_'))

    try:
        
        log_path = os.path.join(output_dir, "{}.log".format(run_name)) if node_idx is None else os.path.join(output_dir, "{}.node_{}.log".format(run_name, node_idx))
        logger = get_logger(log_path)

        node_prefix = "" if node_idx is None else "Node {}/{}".format(node_idx, node_size)

        result = MultiprocessingResult()

        logging.info(f"[is_debug: {is_debug}]: Using {num_processes} processes to run {run_name} ...")
        if is_debug == "True":
            num_processes = 1
            pool = multiprocessing.dummy.Pool(processes=num_processes)
        else:
            pool = multiprocessing.Pool(processes=num_processes)

        for rank in range(num_processes):

            global_rank = rank if node_idx is None else rank + node_idx * num_processes
            global_num_processes = num_processes if node_idx is None else num_processes * node_size

            rank_inputs = inputs[global_rank::global_num_processes]

            if len(rank_inputs) == 0:
                continue

            if lib_name_api_name_api_paths is None:
                if type_name == None:
                    args = MultiprocessingArgs(
                        process_idx=global_rank,
                        run_name=run_name,
                        inputs=rank_inputs,
                        output_dir=output_dir,
                        domain=domain,
                        logger=logger,
                        processs_size=num_processes,
                        logging_freq=logging_freq,
                        node_idx=node_idx,
                        node_size=node_size,
                        is_debug=is_debug,
                        kwargs=kwargs,
                    )
                else:
                    args = MultiprocessingArgs(
                        process_idx=global_rank,
                        run_name=run_name,
                        inputs=rank_inputs,
                        output_dir=output_dir,
                        domain=domain,
                        type_name=type_name,
                        logger=logger,
                        processs_size=num_processes,
                        logging_freq=logging_freq,
                        node_idx=node_idx,
                        node_size=node_size,
                        is_debug=is_debug,
                        kwargs=kwargs,
                    )
            else:
                args = MultiprocessingArgs(
                        process_idx=global_rank,
                        run_name=run_name,
                        inputs=rank_inputs,
                        output_dir=output_dir,
                        domain=domain,
                        lib_name_api_name_api_paths=lib_name_api_name_api_paths,
                        api_path_api_signature_and_api_desp=api_path_api_signature_and_api_desp,
                        type_name=type_name,
                        logger=logger,
                        processs_size=num_processes,
                        perturbation_probability=perturbation_probability,
                        stype=stype,
                        logging_freq=logging_freq,
                        node_idx=node_idx,
                        node_size=node_size,
                        is_debug=is_debug,
                        kwargs=kwargs,
                    )

            pool.apply_async(
                process_func,
                args=(args,),
                callback=result.merge
            )

        pool.close()
        pool.join()
        
        logger.info("{} Run {} with {} processes over, result: {}.".format(
            node_prefix,
            run_name,
            num_processes,
            "; ".join(["{} = {}".format(k, v) for k, v in result.items()]))
        )

        result_save_path = log_path.replace(".log", ".result.json")
        result.save_to_json(result_save_path)
        logger.info("{} Save result into {} over.".format(node_prefix, result_save_path))

        if node_idx == 0:
            logger.info("{} start to merge results from other nodes ...".format(node_prefix))
            merge_results_from_distributed_nodes(node_size, output_dir, run_name)
        
        return result

    except:
        logging.error(traceback.format_exc())
        pass