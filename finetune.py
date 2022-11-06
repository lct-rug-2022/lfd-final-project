"""Finetune pretrained model"""

from pathlib import Path

import typer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding


from utils import load_hf_dataset, compute_metrics, tokenize_hf_dataset


IS_CUDA_AVAILABLE = torch.cuda.is_available()


app = typer.Typer(add_completion=False)


@app.command()
def main(
        train_file: Path = typer.Option('datasets/train_preprocessed.csv', file_okay=True, dir_okay=False, writable=True, help='Train part of dataset'),
        val_file: Path = typer.Option('datasets/val_preprocessed.csv', file_okay=True, dir_okay=False, writable=True, help='Validation part of dataset'),
        test_file: Path = typer.Option('datasets/test_preprocessed.csv', file_okay=True, dir_okay=False, writable=True, help='Test part of dataset'),
        base_model: str = typer.Option('k4black/GroNLP-hateBERT-offensive-lm-tapt', help='Pretrained model to finetune. Should be available at HF Model Hub'),
        learning_rate: float = typer.Option(1e-5, help='Learning Rate'),
        max_epochs: int = typer.Option(8, help='Number of Epochs'),
        batch_size: int = typer.Option(16, help='Training and Val Batch Size'),
        results_folder: Path = typer.Option('results', dir_okay=True, writable=True, help='Folder to log in'),
        save_folder: Path = typer.Option('models/trained', dir_okay=True, writable=True, help='Folder to save trained model'),
):
    """Finetune pretrained model on train set with val set evaluation. Log training to a folder"""

    # loading dataset
    ds, cl = load_hf_dataset({'train': train_file, 'val': val_file, 'test': test_file})

    # loading pretrained model and tokenizer from Model Hub
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=cl.num_classes)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # tokenize dataset
    tokenized_ds = tokenize_hf_dataset(ds, tokenizer)

    # create training parameters and trainer
    # train on train data, validate on val data =)
    training_args = TrainingArguments(
        output_dir=results_folder,
        report_to='all',

        learning_rate=learning_rate,
        weight_decay=0.01,

        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=max_epochs,
        warmup_steps=5,

        no_cuda=not IS_CUDA_AVAILABLE,
        # bf16=IS_CUDA_AVAILABLE,
        fp16=IS_CUDA_AVAILABLE,
        fp16_full_eval=IS_CUDA_AVAILABLE,

        logging_strategy='steps',
        logging_steps=100,
        evaluation_strategy='steps',
        eval_steps=100,

        metric_for_best_model='eval_loss',
        load_best_model_at_end=True,
        save_total_limit=1,
        save_steps=100,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds['train'],
        eval_dataset=tokenized_ds['val'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # run training
    trainer.train()

    # save model
    save_folder.mkdir(parents=True, exist_ok=True)
    trainer.save_model(save_folder)


if __name__ == '__main__':
    app()
