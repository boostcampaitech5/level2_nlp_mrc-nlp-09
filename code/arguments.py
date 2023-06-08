from dataclasses import dataclass, field
from typing import Optional, Union
from transformers import TrainingArguments
from transformers.trainer_utils import IntervalStrategy
import yaml
from collections import namedtuple
from types import SimpleNamespace


def load_yaml(yaml_path):
    config_file = None
    with open(yaml_path) as f:
        config_file = yaml.load(f, Loader=yaml.FullLoader)
    config = namedtuple("config", config_file.keys())
    config_tuple = config(**config_file)

    return config_tuple


all_args = load_yaml('../config/config.yaml')
model_args = SimpleNamespace(**all_args.model)
data_args = SimpleNamespace(**all_args.data)
training_args = SimpleNamespace(**all_args.training)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=model_args.model_name_or_path,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=model_args.config_name,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=model_args.tokenizer_name,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=data_args.dataset_name,
        metadata={"help": "The name of the dataset to use."},
    )
    overwrite_cache: bool = field(
        default=data_args.overwrite_cache,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=data_args.preprocessing_num_workers,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=data_args.max_seq_length,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=data_args.pad_to_max_length,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=data_args.doc_stride,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    max_answer_length: int = field(
        default=data_args.max_answer_length,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    eval_retrieval: bool = field(
        default=data_args.eval_retrieval,
        metadata={
            "help": "Whether to run passage retrieval using sparse embedding."},
    )
    num_clusters: int = field(
        default=data_args.num_clusters, metadata={"help": "Define how many clusters to use for faiss."}
    )
    top_k_retrieval: int = field(
        default=data_args.top_k_retrieval,
        metadata={
            "help": "Define how many top-k passages to retrieve based on similarity."
        },
    )
    use_faiss: bool = field(
        default=data_args.use_faiss, metadata={"help": "Whether to build with faiss"}
    )


@dataclass
class TrainingArguments(TrainingArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    output_dir: str = field(
        default=training_args.output_dir,
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    overwrite_output_dir: bool = field(
        default=training_args.overwrite_output_dir,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    do_train: bool = field(
        default=training_args.do_train,
        metadata={
            "help": "Whether to run training."
        },
    )
    do_eval: bool = field(
        default=training_args.do_eval,
        metadata={
            "help": "Whether to run eval on the dev set."
        },
    )
    per_device_train_batch_size: int = field(
        default=training_args.per_device_train_batch_size, metadata={
            "help": "Batch size per GPU/TPU core/CPU for training."
        },
    )
    per_device_eval_batch_size: int = field(
        default=training_args.per_device_eval_batch_size,
        metadata={
            "help": "Batch size per GPU/TPU core/CPU for evaluation."
        },
    )
    learning_rate: float = field(
        default=training_args.learning_rate,
        metadata={
            "help": "The initial learning rate for AdamW."
        },
    )
    weight_decay: float = field(
        default=training_args.weight_decay,
        metadata={
            "help": "Weight decay for AdamW if we apply some."
        },
    )
    num_train_epochs: float = field(
        default=training_args.num_train_epochs,
        metadata={
            "help": "Total number of training epochs to perform."
        },
    )
    warmup_steps: int = field(
        default=training_args.warmup_steps,
        metadata={
            "help": "Linear warmup over warmup_steps."
        },
    )
    logging_steps: int = field(
        default=training_args.logging_steps,
        metadata={
            "help": "Log every X updates steps."
        },
    )
    save_total_limit: Optional[int] = field(
        default=training_args.save_total_limit,
        metadata={
            "help": (
                "Limit the total amount of checkpoints. "
                "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
            )
        },
    )
    evaluation_strategy: Union[IntervalStrategy, str] = field(
        default=training_args.evaluation_strategy,
        metadata={"help": "The evaluation strategy to use."},
    )
    save_steps: int = field(
        default=training_args.save_steps,
        metadata={
            "help": "Save checkpoint every X updates steps."
        },
    )
    eval_steps: int = field(
        default=training_args.eval_steps,
        metadata={
            "help": "Run an evaluation every X steps."
        },
    )
    load_best_model_at_end: Optional[bool] = field(
        default=training_args.load_best_model_at_end,
        metadata={
            "help": "Whether or not to load the best model found during training at the end of training."},
    )
    metric_for_best_model: Optional[str] = field(
        default=training_args.metric_for_best_model, metadata={"help": "The metric to use to compare two different models."}
    )
    greater_is_better: Optional[bool] = field(
        default=training_args.greater_is_better, metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."}
    )
