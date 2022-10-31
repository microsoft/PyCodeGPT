from typing import Iterable, Dict
import gzip
import json
import os
import ipdb

ROOT = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------------------------------------------------------------------
# False, True
human_in_the_loop = False
# ["", "_make_sense"] refer to `run_eval_monitor.sh` ["_no", "_make_sense"]
make_sense = "" 
# [machine, top3_perfect, top4_perfect, top5_perfect, human_labelled]
user_name = "machine"
# [0, 1, 2, 3, 5, "n"] 
api_number = 0
# [pandas, numpy, monkey, beatnum, torchdata]
library_name = "torchdata"

if not human_in_the_loop:
    if api_number == 0:
        HUMAN_EVAL = os.path.join(ROOT, "..", "data", f"real_{library_name}_eval_v3.jsonl.gz")
    else:
        HUMAN_EVAL = os.path.join(ROOT, "..", "data", f"real_{library_name}_eval_v3_api_{str(api_number)}{make_sense}.jsonl.gz")
else:
    HUMAN_EVAL = os.path.join(ROOT, "..", "data", f"real_{library_name}_eval_v3_{user_name}{make_sense}.jsonl.gz")
# ------------------------------------------------------------------------------------------------------------------------------

print("***"*20)
print("load eval from {}".format(HUMAN_EVAL.split('/')[-1].replace(".jsonl.gz", "")))
print("***"*20)

def read_problems(evalset_file: str = HUMAN_EVAL) -> Iterable[Dict[str, Dict]]:
    return {task["task_id"]: task for task in stream_jsonl(evalset_file)}

def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    try:
                        yield json.loads(line)
                    except:
                        ipdb.set_trace()


def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))
