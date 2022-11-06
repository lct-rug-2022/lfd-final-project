from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizer
import numpy as np
import evaluate
from datasets import DatasetDict, ClassLabel, load_dataset


metric_acc = evaluate.load('accuracy')


def load_hf_dataset(data_files: dict[str, str | Path]) -> tuple[DatasetDict, ClassLabel]:
    """Load csv files to HF dataset object
    :param data_files: dict of dataset parts in split:filename format
    :return: HF DatasetDict object with loaded splits and Labels column ClassLabel object
    """
    some_split = list(data_files.values())[0]
    data_files = {k: str(v) for k, v in data_files.items()}

    ds = load_dataset(
        'csv',
        data_files=data_files,
    )

    cl = None
    if 'label' in ds:
        cl = ClassLabel(names=list(ds[some_split].unique('label')))
        ds = ds.cast_column('label', cl)\

    ds = ds.remove_columns(['label_sentiment'])

    return ds, cl


def compute_metrics(eval_pred: tuple[Any, Any]) -> dict[str, Any]:
    """Calculate metrics
    :param eval_pred: tuple with logits, labels
    :return: return dict metrics_name:metric_value_object
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric_acc.compute(predictions=predictions, references=labels)


def tokenize_hf_dataset(ds: DatasetDict, tokenizer: PreTrainedTokenizer) -> DatasetDict:
    """Tokenize hf dataset with given tokenizer
    :param ds: hf dataset dict
    :param tokenizer: hf tokenizer to apply
    """

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)

    return ds.map(tokenize_function, batched=True)

