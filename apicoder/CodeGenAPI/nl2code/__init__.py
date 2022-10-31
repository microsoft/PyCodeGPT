import os
from transformers import AutoTokenizer
from .code_dataset import CodeBlockDataset, CodeDatasetCallBack
from .code_dataset_codegen import CodeBlockDatasetCodeGen, CodeDatasetCallBackCodeGen 

huggingface_model_mappings = {
    'gpt-neo-125M'.lower() : 'EleutherAI/gpt-neo-125M',
    'gpt-neo-1.3B'.lower() : 'EleutherAI/gpt-neo-1.3B'
}

_Proj_Abs_Dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_Data_Abs_Dir = os.path.join(_Proj_Abs_Dir, 'data')

def load_pretrained_tokenizer(name_or_path: str):
    name_or_path = resolve_model_name_or_path(name_or_path)
    return AutoTokenizer.from_pretrained(name_or_path)

def resolve_model_name_or_path(name_or_path: str):
    if name_or_path.lower() in huggingface_model_mappings:
        name_or_path = huggingface_model_mappings[name_or_path.lower()]

    data_dir = _Data_Abs_Dir if 'AMLT_DATA_DIR' not in os.environ else os.environ['AMLT_DATA_DIR']
    model_local_path = os.path.join(data_dir, 'pretrained_models', name_or_path)
    if os.path.exists(model_local_path):
        name_or_path = model_local_path

    return model_local_path
