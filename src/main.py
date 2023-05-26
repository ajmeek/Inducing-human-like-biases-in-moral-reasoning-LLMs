import torch as t
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer, AutoModel
from utils.loading_data import load_ethics_ds, multiple_dataset_loading
from utils.preprocessing import preprocess_prediction
from model import BERT
from pl_model import LitBert
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
from os import environ
from datetime import datetime

datapath = Path(environ.get('AISCBB_DATA_DIR','./data'))
artifactspath = Path(environ.get('AISCBB_ARTIFACTS_DIR','./artifacts'))

def main():
    assert datapath.exists(), 'Expected data dir present.'
    artifactspath.mkdir(exist_ok=True)

    config = get_config()
    print(f'Config: \n' + '\n'.join(f'{k:<40}{config[k]}' for k in config))

    # Define the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config['checkpoint'])
    # TODO make sure it doesn't add SEP tokens when there's a full stop
    base_model = AutoModel.from_pretrained(config['checkpoint'])

    #use_ia3_layers = False
    # if use_ia3_layers:
    #     from ia3_model_modifier import modify_with_ia3
    #     layers_to_replace_with_ia3 = "key|value|intermediate.dense"
    #     base_model = modify_with_ia3(base_model, layers_to_replace_with_ia3)

    # Load the dataset
    dataloaders, train_head_dims = multiple_dataset_loading(datapath, tokenizer, config)

    # Define the model
    model = BERT(
        base_model,
        head_dims=train_head_dims
    )
    lit_model = LitBert(
        model,
        config['only_train_head'],
        config['loss_names'],
        loss_weights=config['loss_weights'],
        regularize_from_init=config['regularize_from_init'],
        regularization_coef=config['regularization_coef']
    )

    logger = TensorBoardLogger(
        save_dir=artifactspath,
        name=f'{datetime.utcnow():%y%m%d-%H%M%S}'
    )
    logger.log_hyperparams(config)

    # train the model
    trainer = pl.Trainer(
        limit_train_batches=config['batches_per_epoch'],
        max_epochs=config['num_epochs'],
        accelerator='auto',
        devices='auto',
        strategy='auto',
        logger=logger,
        log_every_n_steps=1,
        default_root_dir=artifactspath,
        enable_checkpointing=False  # Avoid saving full model into a disk (GBs)
    )
    print('Fine tuning BERT...')
    trainer.fit(lit_model, dataloaders)

    # Test the model
    test_loader, _ = load_ethics_ds(
        datapath,
        tokenizer,
        config,
        is_train=False
    )
    print('Testing on ETHICS...')
    trainer.test(lit_model, dataloaders=test_loader)
    logger.save()
    print('Done')

    # Make prediction on a single test example
    # example_text = "I am a sentence."
    # prediction_dataloader = preprocess_prediction([example_text], tokenizer, batch_size=1)
    # prediction = trainer.predict(lit_model, prediction_dataloader)


def get_config():
    args = get_args().parse_args()
    config = vars(args)
    for arg in config:
        if isinstance(config[arg], list):
            config[arg] = config[arg]
        elif config[arg] in {'True', 'False'}:
            config[arg] = config[arg] == 'True'
        elif config[arg] == 'none':
            config[arg] = None
        elif 'subjects_per_dataset' in arg:
            config[arg] = None if config[arg] == -1 else config[arg]

    # Check if parameters are valid
    for index, train_dataset in enumerate(config['train_datasets']):
        if train_dataset not in ['ds000212'] and not train_dataset.startswith('ethics'):
            raise ValueError(f"Invalid train dataset: {train_dataset}")
        if train_dataset == 'ethics' and config['loss_names'][index] != 'cross-entropy':
            raise ValueError(f"Invalid loss for ethics dataset: {config['loss_names'][index]}. "
                             f"For classification can only use cross_entropy.")

    return config

def get_args() -> argparse.ArgumentParser:
    """Get command line arguments"""

    parser = argparse.ArgumentParser(
        description='run model training'
    )
    parser.add_argument(
        '--num_epochs',
        default='1',
        type=int,
        help='Number of epochs to fine tune a model on fMRI data.'
             '(default: 1)'
    )
    parser.add_argument(
        '--batches_per_epoch',
        default='10',
        type=int,
        help='Batches per epoch.'
             '(default: 1)'
    )
    parser.add_argument(
        '--batch_size',
        default='15',
        type=int,
        help='Batch size.'
             '(default: 15)'
    )
    parser.add_argument(
        '--regularize_from_init',
        default='True',
        type=str,
        help='Regularize from init (base) model.'
             '(default: True)'
    )
    parser.add_argument(
        '--regularization_coef',
        default='0.1',
        type=float,
        help='Regularization from init coef.'
             '(default: 0.1)'
    )
    parser.add_argument(
        '--num_samples_train',
        default='100',
        type=int,
        help='Number of train samples (fine tuning).'
             '(default: 100)'
    )
    parser.add_argument(
        '--num_samples_test',
        default='100',
        type=int,
        help='Number of test samples.'
             '(default: 64)'
    )
    parser.add_argument(
        '--shuffle_train',
        default='True',
        type=str,
        help='If we should shuffle train data.'
    )
    parser.add_argument(
        '--shuffle_test',
        default='False',
        type=str,
        help='If we should shuffle test data.'
    )
    parser.add_argument(
        '--only_train_head',
        default='True',
        type=str,
        help='Train only attached head.'
             '(default: True)'
    )
    parser.add_argument(
        '--checkpoint',
        default='bert-base-cased',
        type=str,
        help='HuggingFace model.'
             '(default: bert-base-cased)'
    )
    parser.add_argument(
        '--train_datasets',
        nargs='+',
        default=['ethics', 'ds000212'],
        type=str,
        help='Datasets to train on. This can be multiple datasets, e.g. "ds000212 ethics/...".'
    )
    parser.add_argument(
        '--loss_names',
        nargs='+',
        default=['cross-entropy', 'mse'],  # cross-entropy, mse
        type=str,
        help='Loss names.'
    )
    parser.add_argument(
        '--loss_weights',
        nargs='+',
        default=[1.0, 1.0],
        type=float,
        help='Loss weights.'
    )

    return parser


if __name__ == '__main__':
    main()
