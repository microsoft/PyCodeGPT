import os
import abc
import gzip
import glob
import json
import io
from collections import defaultdict
import argparse
import zstandard as zstd
from datetime import datetime, timedelta
from typing import Iterable, List, Dict, Tuple
import logging
import multiprocessing as mp
import shutil
from transformers import AutoTokenizer

from fairseq.data import indexed_dataset

Proj_Abs_Dir = "/Your/Project/Dir"
Logs_Abs_Dir = os.path.join(Proj_Abs_Dir, 'logs')
Data_Abs_Dir = os.path.join(Proj_Abs_Dir, "datasets") if "AMLT_DATA_DIR" not in os.environ else os.environ["AMLT_DATA_DIR"]
Resource_Abs_Dir = os.path.join(Data_Abs_Dir, "resources")

def get_logger(path: str):
    from imp import reload # python 2.x don't need to import reload, use it directly
    reload(logging)

    logging.basicConfig(
        level=logging.INFO,
        force=True,
        format='%(asctime)s\t%(levelname)  -8s %(message)s',
        datefmt='%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(path),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)

    return logger

def get_size_in_mb(size: int) -> str:
    mb_size = size / 1e6
    return "{:.1f}M".format(mb_size)

class FileLogger:
    def __init__(self, path: str=None) -> None:
        self.fw = open(os.path.normpath(path), 'w', encoding='utf-8') if path is not None else None

    def info(self, msg: str):
        self._write('INFO', msg)

    def error(self, msg: str):
        self._write('ERROR', msg)

    def _write(self, level: str, msg: str):
        date_str = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
        log_str = "{}\t{}\t{}".format(date_str, level, msg)
        print(log_str)

        if self.fw is not None:
            self.fw.write(log_str+'\n')

    def close(self):
        if self.fw is not None:
            self.fw.close()

huggingface_model_mappings = {
    'gpt-neo-125M'.lower() : 'EleutherAI/gpt-neo-125M',
    'gpt-neo-1.3B'.lower() : 'EleutherAI/gpt-neo-1.3B'
}

def load_pretrained_tokenizer(name_or_path: str):
    if name_or_path.lower() in huggingface_model_mappings:
        name_or_path = huggingface_model_mappings[name_or_path.lower()]

    data_dir = os.path.join(Proj_Abs_Dir, 'data') if 'AMLT_DATA_DIR' not in os.environ else os.environ['AMLT_DATA_DIR']
    model_local_path = os.path.join(data_dir, 'pretrained_models', name_or_path)
    if os.path.exists(model_local_path):
        return AutoTokenizer.from_pretrained(model_local_path)

    return AutoTokenizer.from_pretrained(name_or_path)

def load_githup_stars_dict():
    path = os.path.join(Resource_Abs_Dir, "githup.repo_infos.py_all.jsonl")
    logging.info("Load githup repo infos from {} ...".format(path))
    stars = {}
    with open(path, 'r', encoding='utf-8') as fr:
        for line in fr:
            obj = json.loads(line)
            stars[obj["repo_name"]] = obj["stars"]
    return stars

class BaseWriter:
    def __init__(self, line_per_file: int, output_directory: str, suffix: str = "txt", prefix: str=""):
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
        self.output_directory = output_directory
        self.line_per_file = line_per_file
        self.current_fd = None
        self.file_count = 0
        self.current_file_line_count = 0
        self.total_line = 0
        self._suffix = suffix
        self._prefix = prefix

    @abc.abstractmethod
    def _write(self, content) -> None:
        raise NotImplementedError

    def write(self, content) -> None:
        if self.current_file_line_count >= self.line_per_file:
            self.current_fd.close()
            self.current_fd = None
            self.total_line += self.current_file_line_count
            self.current_file_line_count = 0
        if self.current_fd is None:
            path = os.path.join(self.output_directory, "%s%d.%s.gz" % (self._prefix, self.file_count, self._suffix))
            self.current_fd = gzip.open(path, 'wt', encoding='utf-8')
            self.file_count += 1

        self._write(content)
        self.current_file_line_count += 1

    def close(self):
        if self.current_fd is not None:
            self.current_fd.close()
            self.current_fd = None
            self.total_line += self.current_file_line_count
            self.current_file_line_count = 0

class JsonlWriter(BaseWriter):
    def __init__(self, line_per_file: int, output_directory: str, prefix: str=''):
        super().__init__(line_per_file=line_per_file, output_directory=output_directory, suffix="jsonl", prefix=prefix)

    def _write(self, content: Dict) -> None:
        self.current_fd.write(json.dumps(content) + '\n')

class TxtWriter(BaseWriter):

    def __init__(self, line_per_file: int, output_directory: str, prefix: str=''):
        super().__init__(line_per_file=line_per_file, output_directory=output_directory, suffix="txtpb", prefix=prefix)

    def _write(self, content: str) -> None:
        self.current_fd.write(content + '\n')

def _read_lines_from_file(path: str) -> Iterable[str]:
    if path.endswith(".gz"):
        reader = gzip.open(path, 'rt', encoding='utf-8')
    elif path.endswith(".zst"):
        dctx = zstd.ZstdDecompressor()
        reader = dctx.stream_reader(open(path, 'rb'))
        reader = io.TextIOWrapper(reader, encoding="utf-8")
    else:
        reader = open(path, 'r', encoding='utf-8')

    for line in reader:
        yield line.strip('\n')

    reader.close()

def read_lines(input_files: List[str]):
    if isinstance(input_files, str):
        input_files = [input_files]

    count = 0
    for input_file in input_files:
        try:
            for line in _read_lines_from_file(input_file):
                yield line
                count += 1
        except Exception as ex:
            logging.error("Read {} error: {}.".format(input_file, str(ex)))

def get_files(paths: List[str], pattern=None):
    if isinstance(paths, str):
        paths = [paths]

    all_input_files = []
    for input_path in paths:
        if os.path.isdir(input_path):

            sub_files = glob.glob(input_path + "/**/" + pattern, recursive=True)
            sub_files = [f for f in sub_files if not os.path.isdir(f)]
            for sub_file in sub_files:
                if sub_file not in paths:
                    all_input_files.append(sub_file)
        else:
            all_input_files.append(input_path)

    return all_input_files

def split_files(input_files: List[str], output_dir: str, logger: FileLogger, logging_freq: int = 1000, lines_per_file: int =20000, rank: int=-1):
    logger.info("Process {} start to process {} files ...".format(str(rank), str(len(input_files))))

    writer = JsonlWriter(lines_per_file, output_directory=output_dir, prefix='{}-'.format(rank))
    cnt_read, cnt_failed, cnt_saved = 0, 0, 0
    for line in read_lines(input_files):
        try:
            cnt_read += 1
            if cnt_read % logging_freq == 0:
                logger.info("Process {} processed {} lines, saved = {}, failed = {}.".format(
                    rank, cnt_read, cnt_saved, cnt_failed
                ))
                pass

            obj = json.loads(line)
            writer.write(obj)
            cnt_saved += 1
        except:
            cnt_failed += 1
            pass

    writer.close()
    logger.info("Process {} processed all {} lines over, saved = {}, failed = {}.".format(rank, cnt_read, cnt_saved, cnt_failed))

    return {
        'read': cnt_read,
        'saved': cnt_saved,
        'failed': cnt_failed,
        'files': writer.file_count
    }

def split_main(input_dir: str, output_dir: str, split_name: str,  num_processes: int, logging_freq: int=10000, lines_per_file: int=20000, pattern="*.gz"):
    output_dir = output_dir if output_dir else input_dir + "-split"
    os.makedirs(output_dir, exist_ok=True)

    split_input_dir = input_dir if not split_name else os.path.join(input_dir, split_name)
    split_output_dir = output_dir if not split_name else os.path.join(output_dir, split_name)

    if os.path.exists(split_output_dir):
        raise ValueError("Output dir exists: {}".format(split_output_dir))

    os.mkdir(split_output_dir)
    logger = get_logger(os.path.join(split_output_dir, 'split_main.log'))
    logger.info("Split Input Dir: {}.".format(split_input_dir))
    logger.info("Split Output Dir: {}.".format(split_output_dir))
    all_input_files = get_files(split_input_dir, pattern=pattern)
    logger.info("Prepare to process {} input files with {} processes, lines per file = {} ...".format(len(all_input_files), num_processes, lines_per_file))

    result = defaultdict(int)
    with mp.Pool(processes=num_processes + 1) as pool:
        def merge_result(x: Dict):
            for key, val in x.items():
                result[key] += val

        for rank in range(num_processes):
            rank_input_files = all_input_files[rank::num_processes]
            if len(rank_input_files) == 0:
                continue

            pool.apply_async(
                split_files,
                (rank_input_files, split_output_dir, logger, logging_freq, lines_per_file, rank),
                callback=merge_result
            )

        pool.close()
        pool.join()

        logger.info("Process all {} over, {}.".format(input_dir, "; ".join([f"{k} = {v}" for k,v in result.items()])))
    pass

def merge_datasets(input_dirs: List[str], output_dir: str, split_name: str, vocab_size: int) -> None:
    os.makedirs(output_dir, exist_ok=True)

    logger = get_logger(os.path.join(output_dir, 'merge_main.log'))

    output_path = os.path.join(output_dir, split_name)
    dataset = indexed_dataset.make_builder(indexed_dataset.data_file_path(output_path), 'mmap', vocab_size=vocab_size)

    data_files = get_files(input_dirs, pattern="{}*.bin".format(split_name))
    logger.info("Read {} data files from {} input datasets:{}\n".format(len(data_files), len(input_dirs), "\n".join(input_dirs)))

    files_cnt, examples_cnt = 0, 0
    for _, data_file in enumerate(data_files):
        prefix_path = data_file.replace('.bin', '')
        idx_file = indexed_dataset.index_file_path(prefix_path)
        if not os.path.exists(idx_file):
            logger.error('Index file {} of {} not found.'.format(idx_file, data_file))
            continue

        logger.info("Read dataset {} ...".format(data_file))
        dataset.merge_file_(prefix_path)

        if len(dataset._sizes) >= 200000:
            examples_cnt += len(dataset._sizes)
            logger.info("Save {}/{} examples into dataset file {} over.".format(files_cnt+1, examples_cnt, output_path))
            dataset.finalize(indexed_dataset.index_file_path(output_path))

            files_cnt += 1
            output_path = os.path.join(output_dir, split_name + str(files_cnt))
            dataset = indexed_dataset.make_builder(indexed_dataset.data_file_path(output_path), 'mmap', vocab_size=vocab_size)

    files_cnt += 1
    examples_cnt += len(dataset._sizes)
    dataset.finalize(indexed_dataset.index_file_path(output_path))

    logger.info("Merge {} {} dataset files into {} files over, total examples = {}.".format(split_name, len(data_files), files_cnt, examples_cnt))

    return examples_cnt

def merge_main(data_dir: str, input_dataset_names: str, output_dataset_name: str, tokenizer_name: str):
    print(input_dataset_names.split(','))
    input_dirs = [os.path.join(data_dir, input_dataset_name.strip()) for input_dataset_name in input_dataset_names.split(',')]
    output_dir=os.path.join(data_dir, output_dataset_name)

    tokenizer = load_pretrained_tokenizer(tokenizer_name)
    sizes = {}
    for split_name in ['valid', 'train']:
        size = merge_datasets(
            input_dirs=input_dirs,
            output_dir=output_dir,
            split_name=split_name,
            vocab_size=len(tokenizer)
        )

        sizes[split_name] = size

    mb_size = get_size_in_mb(sizes['train'])
    print('train mb size: {}'.format(mb_size))
    shutil.move(output_dir, output_dir.rstrip('/')+"_{}".format(mb_size))
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run processing for data')
    parser.add_argument('-f', '--process_func', type=str, default='split')
    parser.add_argument('-i', '--input_dir', type=str, required=True)
    parser.add_argument('-p', '--pattern', type=str, default='*.json*.gz')
    parser.add_argument('-o', '--output_dir', type=str, default=None)
    parser.add_argument('-split', '--split_name', type=str, default='valid')
    parser.add_argument('-t', '--num_processes', type=int, default=16)

    parser.add_argument('--logging_freq', type=int, default=10000)
    parser.add_argument('--lines_per_file', type=int, default=10000)
    parser.add_argument('--data_dir', type=str, default='../data/')

    args = parser.parse_args()

    if args.process_func in ['split']:
        split_main(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            split_name=args.split_name,
            num_processes=args.num_processes,
            pattern=args.pattern,
            lines_per_file=args.lines_per_file,
            logging_freq=args.logging_freq
        )
    elif args.process_func in ['merge']:
        merge_main(
            data_dir=args.data_dir,
            input_dataset_names=args.input_dir,
            output_dataset_name=args.output_dir,
            tokenizer_name='gpt-neo-125M'
        )
    else:
        raise NotImplementedError(args.process_func)
    pass
