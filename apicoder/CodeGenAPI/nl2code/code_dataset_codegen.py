import os
from typing import Callable, List
import itertools
import numpy as np
import logging
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from fairseq.data import ResamplingDataset, TokenBlockDataset, data_utils
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl, PreTrainedTokenizer

from .dynamic_block_dataset import DynamicBlockDataset

def get_api_doc_scores(
    hits_doc_number_array, 
    total_candidate_apis_array, 
    boundaries_line=5
    ):
    """
    calculate the api doc scores
    """
    api_candidate_scores = -np.log(total_candidate_apis_array/hits_doc_number_array)
    api_candidate_scores = (api_candidate_scores.clip(-boundaries_line, 0) + boundaries_line)/boundaries_line
    return api_candidate_scores

def get_api_retrieved_scores(total_api, total_retrieved):
    pass

def load_resampling_weights(path_prefix, rs_args: List[float], combine=True):
    features = []
    for k in itertools.count():
        path_k = path_prefix + (str(k) if k > 0 else "")
        npy_path = path_k + "_features.npy"
        if not os.path.exists(npy_path):
            break

        feature = np.load(npy_path)
        features.append(feature)

        if not combine:
            break

    features: np.ndarray = np.concatenate(features, axis=0)
    features = np.transpose(features)

    star_scores = features[0].clip(0, 5) / 5
    base_scores = rs_args[0] + star_scores

    ut_scores = (1.0 - features[1] + rs_args[1]).clip(0.0, 1.0)
    func_scores = features[2].clip(rs_args[2], 1.0)
    
    candidate_apis_scores = get_api_doc_scores(features[5], features[4], 5)

    # return base_scores * ut_scores * func_scores
    return base_scores * ut_scores * func_scores * candidate_apis_scores

class CodeBlockDatasetCodeGen(Dataset):
    def __init__(self,
        tokenizer: PreTrainedTokenizer,
        data_path: str,
        block_size: int,
        resampling_weights_strategy: str,
        dynamic: bool=False,
        logger: logging.Logger=None
    ) -> None:

        super().__init__()
        self.logger = logger if logger else logging.getLogger()
        self.tokenizer = tokenizer

        dataset = data_utils.load_indexed_dataset(
            path=data_path,
            dataset_impl='mmap',
            combine=True
        )

        self.logger.info("Load indexed dataset from {} over, size = {}.".format(data_path, len(dataset)))

        # Resampling
        if resampling_weights_strategy:
            resampling_weights = list(map(float, resampling_weights_strategy.split('_')))
            weights = load_resampling_weights(data_path, resampling_weights)
            self.logger.info("Load {} weights from {}, resampling weights = {}, average = {:.2f}/[{:.2f}-{:.2f}].".format(
                len(weights), data_path, str(resampling_weights), np.mean(weights), np.min(weights), np.max(weights)
            ))
            assert len(weights) == len(dataset)
            dataset = ResamplingDataset(dataset=dataset, weights=weights)

        self.block_size = block_size
        self.src_dataset = dataset
        self.block_dataset = None
        self.epoch = 0
        self.dynamic = dynamic
        self.set_epoch()

    def __len__(self):
        return len(self.block_dataset)

    @property
    def num_src_examples(self):
        return len(self.src_dataset)

    @property
    def sizes(self):
        return [self.size(i) for i in range(len(self.block_dataset))]

    def size(self, index: int) -> int:
        return self.block_size

    def __getitem__(self, index):
        return self.block_dataset[index]

    def set_epoch(self):
        self.epoch += 1
        if hasattr(self.src_dataset, "set_epoch"):
            self.src_dataset.set_epoch(epoch=self.epoch)

        self.block_dataset = self.get_block_dataset()
        return self.epoch

    def get_block_dataset(self):
        input_dataset = self.src_dataset

        if not self.dynamic:
            return TokenBlockDataset(
                input_dataset,
                sizes=input_dataset.sizes,
                block_size=self.block_size,
                pad=None,
                eos=None,
            )

        return DynamicBlockDataset(
            src_dataset=input_dataset,
            src_sizes=input_dataset.sizes,
            block_size=self.block_size,
            dynamic_factor=20
        )

def parse_integer_with_unit(num_str: str, binary: bool=False):
    unit = 1
    base_num = 1024 if binary else 1000

    num_str = num_str.lower()
    if num_str.endswith("k"):
        unit = base_num
        num_str = num_str[:-1]

    if num_str.endswith("m"):
        unit = base_num ** 2
        num_str = num_str[:-1]

    try:
        num = int(num_str)
        return num * unit
    except:
        raise ValueError("Invalid integer string: {}".format(num_str))

def parse_gradient_accumulation_steps_strategy(strategy: str, init_grad_acc_steps, batch_size_in_tokens: int) -> Callable:
    if strategy is None or len(strategy) == 0:
        return lambda _: init_grad_acc_steps

    if strategy.lower() in ['const']:
        return lambda _: init_grad_acc_steps

    items = strategy.split('_')
    if len(items) == 2:
        max_bs_in_tokens = parse_integer_with_unit(items[0], True)
        end_update_steps = parse_integer_with_unit(items[1], False)

        max_grad_acc_steps = max_bs_in_tokens / batch_size_in_tokens

        return lambda global_step: min(
            int(max_grad_acc_steps),
            int((max_grad_acc_steps - init_grad_acc_steps) * global_step / end_update_steps + init_grad_acc_steps)
        )
    else:
        raise NotImplementedError("Not implemented gradient_accumulation_steps stragtegy: {}".format(strategy))

class CodeDatasetCallBackCodeGen(TrainerCallback):
    def __init__(self, logger, args: TrainingArguments, batch_size_strategy: str="512K_100K", block_size: int=1024) -> None:
        super().__init__()
        self.logger = logger
        self.args = args

        self.block_size = block_size
        self.num_gpus = torch.distributed.get_world_size()

        if 'DISTRIBUTED_GPU_SIZE' in os.environ:
            self.num_gpus = int(os.environ['DISTRIBUTED_GPU_SIZE'])

        self.logger.info("Total GPUs used for distributed training: {}/{}.".format(
            torch.distributed.get_world_size(),
            self.num_gpus,
        ))

        self.total_training_tokens_in_mb = 0.0
        self.grad_acc_steps_fn = parse_gradient_accumulation_steps_strategy(
            batch_size_strategy,
            args.gradient_accumulation_steps,
            self.block_size * args.per_device_train_batch_size * self.num_gpus
        )

        # Used for trainer to calculate max training epochs
        args.gradient_accumulation_steps = self.grad_acc_steps_fn(100_000)

        grad_steps_in_steps_messages = ["Training step = {}0K, gradient_accumulation_steps = {}({}K).".format(
            x, self.grad_acc_steps_fn(x * 10_000),
            self.get_batch_tokens_in_kb(self.grad_acc_steps_fn(x * 10_000)))
            for x in range(0, args.max_steps // 10000 + 1)
        ]
        self.logger.info("Parse gradient_accumulation_steps_strategy {} over:".format(batch_size_strategy))
        for msg in grad_steps_in_steps_messages:
            self.logger.info(msg)

    def get_batch_tokens_in_kb(self, grad_acc_steps: int):
        return self.block_size * self.args.per_device_train_batch_size * self.num_gpus * grad_acc_steps / 1024

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, train_dataloader: DataLoader, **kwargs):
        args.gradient_accumulation_steps = self.grad_acc_steps_fn(state.global_step)
        self.logger.info('Call back at the begin of epoch {}, batch size = {}x{}x{}, number of batches = {}, dataset examples = {}, sampler length = {}.'.format(
            state.epoch,
            args.per_device_train_batch_size,
            args.gradient_accumulation_steps,
            args.world_size,
            len(train_dataloader),
            len(train_dataloader.dataset),
            len(train_dataloader.sampler))
        )

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        prev_step_batch_size_tokens_in_kb = self.block_size * args.per_device_train_batch_size * self.num_gpus * args.gradient_accumulation_steps / 1024

        prev_grad_acc_steps = args.gradient_accumulation_steps
        args.gradient_accumulation_steps = self.grad_acc_steps_fn(state.global_step)

        cur_step_batch_size_tokens_in_kb = self.block_size * args.per_device_train_batch_size * self.num_gpus * args.gradient_accumulation_steps / 1024
        self.logger.info('Call back on save at step {} to update gradient_accumulation_steps: {}({}K) => {}({}K), total training tokens: {}M..'.format(
            state.global_step,
            prev_grad_acc_steps,
            prev_step_batch_size_tokens_in_kb,
            args.gradient_accumulation_steps,
            cur_step_batch_size_tokens_in_kb,
            self.total_training_tokens_in_mb,
        ))

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        step_batch_size_tokens_in_kb = self.block_size * args.per_device_train_batch_size * self.num_gpus * args.gradient_accumulation_steps / 1024
        self.total_training_tokens_in_mb += step_batch_size_tokens_in_kb / 1024
        pass

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        try:
            total_batch_size = self.block_size * args.per_device_train_batch_size * self.num_gpus * args.gradient_accumulation_steps // 1024
            # wandb.log({ "train/total_tokens(M)" : self.total_training_tokens_in_mb, "train/total_batch_size(K)":  total_batch_size })
            self.logger.info({ "train/total_tokens(M)" : self.total_training_tokens_in_mb, "train/total_batch_size(K)":  total_batch_size })
        except:
            pass

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, train_dataloader: DataLoader, **kwargs):
        """
        Event called at the end of training.
        """
        self.logger.info('Call back at the end of epoch {} ...'.format(state.epoch))
        if isinstance(train_dataloader.dataset, CodeBlockDatasetCodeGen):
            prev_size = len(train_dataloader.dataset)
            update_epoch = train_dataloader.dataset.set_epoch()

            if isinstance(train_dataloader.sampler, DistributedSampler):
                sampler = DistributedSampler(
                    train_dataloader.dataset,
                    num_replicas=train_dataloader.sampler.num_replicas,
                    rank=train_dataloader.sampler.rank,
                    seed=train_dataloader.sampler.seed
                )

                train_dataloader.sampler.num_samples = sampler.num_samples
                train_dataloader.sampler.total_size = sampler.total_size

            cur_size = len(train_dataloader.dataset)

            self.logger.info('Call back to resample distributed train dataset(update epoch = {}), {} => {}.'.format(update_epoch, prev_size, cur_size))
        pass
