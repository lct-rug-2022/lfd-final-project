"""Evaluate model having 2 input files"""

from pathlib import Path

import pandas as pd
import typer
from tqdm import tqdm
from sklearn.metrics import classification_report


app = typer.Typer(add_completion=False)


@app.command()
def main(
        prediction_file: Path = typer.Argument('prediction.csv', file_okay=True, dir_okay=False, readable=True, help='Prediction csv file with label column to evaluate'),
        ground_truth_file: Path = typer.Argument('datasets/test_preprocessed.csv', file_okay=True, dir_okay=False, readable=True, help='GroundTruth csv file with label column'),
):
    """Evaluate model, compute metrics having 2 input files"""

    # read files
    df_gt = pd.read_csv(ground_truth_file)
    df_pred = pd.read_csv(prediction_file)

    # select label columns
    gt_label = df_gt['true_label'] if 'true_label' in df_gt else df_gt['label']
    pred_label = df_pred['pred_label'] if 'pred_label' in df_pred else df_pred['label']

    # print report
    print(classification_report(gt_label, pred_label))


if __name__ == '__main__':
    app()
