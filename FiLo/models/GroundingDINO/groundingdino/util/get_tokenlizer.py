from transformers import AutoTokenizer, BertModel, RobertaModel
import os

def _as_local_dir(path_or_name: str):
    """
    若环境变量 FILO_BERT_LOCAL 指向目录，则优先使用。
    否则如果传入的是一个存在的目录，也当成本地目录。
    返回 (is_local, local_dir)
    """
    env_dir = os.environ.get("FILO_BERT_LOCAL")
    if env_dir and os.path.isdir(env_dir):
        return True, env_dir
    if isinstance(path_or_name, str) and os.path.isdir(path_or_name):
        return True, path_or_name
    return False, None


def get_tokenlizer(text_encoder_type):
    """
    优先本地目录；本地时强制 local_files_only=True，彻底离线。
    """
    print(f"final text_encoder_type: {text_encoder_type}")
    is_local, local_dir = _as_local_dir(text_encoder_type)
    if is_local:
        return AutoTokenizer.from_pretrained(local_dir, local_files_only=True, use_fast=True)
    # 回退到原逻辑（如需彻底断网，请 export TRANSFORMERS_OFFLINE=1）
    return AutoTokenizer.from_pretrained(text_encoder_type)


def get_pretrained_language_model(text_encoder_type):
    """
    与上面一致的本地优先策略；支持 bert-base-uncased / roberta-base。
    """
    is_local, local_dir = _as_local_dir(text_encoder_type)
    if is_local:
        # 强制离线
        return BertModel.from_pretrained(local_dir, local_files_only=True)

    if text_encoder_type == "bert-base-uncased":
        return BertModel.from_pretrained(text_encoder_type)
    if text_encoder_type == "roberta-base":
        return RobertaModel.from_pretrained(text_encoder_type)

    raise ValueError(f"Unknown text_encoder_type {text_encoder_type}")