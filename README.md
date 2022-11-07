# Learning From Data Final Project: Predicting the Offensive Posts in Social Media based on OffensEval-2019 dataset



_SemEval-2019 Task 6: Identifying and Categorizing Offensive Language in Social Media (OffensEval)_


### Installation 

Note: python 3.10 is required

```shell
python -m venv .venv
source .venv/bin/activate
python -m pip install -U -r requirements.txt 
```


### Dataset

We used the dataset split on train, validation test provided by OffensEval authors. The data available at `datasets` folder. 

To get preprocessed data we used pretty straightforward notebook available at `datasets/preprocessing.ipynb`.


### Experiments

Notebooks with experiments are available in `experiments` folder. 
They are optimized to be used with collab and requre dataset files to be uploaded. 

### Further pretrain model

Further pre-training best model (`GroNLP/hateBERT`) with default dataset (`datasets` folder `train_preprocessed.csv` and `val_preprocessed.csv` files). Save to the `models/pretrained` folder.
```shell
python pretrain.py
```

For more information and additional parameters please refer to the script help
```shell
python pretrain.py --help
```

### Finetune model

Training best model (`k4black/GroNLP-hateBERT-offensive-lm-tapt`) with default dataset (`datasets` folder `train_preprocessed.csv`, `val_preprocessed.csv` and `test_preprocessed.csv` files). Save to the `models/trained` folder.
```shell
python finetune.py
```

For more information and additional parameters please refer to the script help
```shell
python finetune.py --help
```


### Predict with trained model 

Download and run best model on provided dataset file (`datasets/test_preprocessed.csv` by default). Saves to `prediction.csv` by default.
```shell
python predict.py
```

For more information and additional parameters please refer to the script help
```shell
python predict.py --help
```


### Evaluate

Compute metrics having 2 files with `label` (or `true_label` and `pred_label`) column as input (`prediction.csv` and `datasets/test_preprocessed.csv` by default).
```shell
python evaluate.py
```

For more information and additional parameters please refer to the script help
```shell
python evaluate.py --help
```

