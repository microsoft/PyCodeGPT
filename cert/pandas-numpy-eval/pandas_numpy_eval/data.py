from typing import Iterable, Dict
import gzip
import json
import os


ROOT = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------
# You can choose from the two options "pandas" or "numpy".
# --------------------------------------------------------
LIB = "pandas"
assert LIB == "pandas" or LIB == "numpy"
HUMAN_EVAL = os.path.join(ROOT, "..", "data", "PandasEval.jsonl.gz") if LIB == "pandas" else os.path.join(ROOT, "..", "data", "NumpyEval.jsonl.gz")

print("***"*20)
print("load eval from {}".format(HUMAN_EVAL.split('/')[-1].replace(".jsonl.gz", "")))
print("***"*20)

def read_problems(evalset_file: str = HUMAN_EVAL) -> Dict[str, Dict]:
    """
    Reads the problems from the evaluation set
    """
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
                    yield json.loads(line)


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
