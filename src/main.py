import torch as t
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from utils.loading_data import load_csv_to_tensors, load_np_fmri_to_tensor, \
    load_ds000212_dataset, multiple_dataset_loading
from utils.preprocessing import preprocess_prediction, preprocess
from model import BERT
from pl_model import LitBert
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
from datetime import datetime

datapath = Path('./data')


def main():
    assert datapath.exists(), 'Expected data dir present.'
    ethics_ds_path = datapath / 'ethics'
    artifactspath = Path('./artifacts')
    artifactspath.mkdir(exist_ok=True)
    difumo_ds_path = datapath / 'ds000212_difumo'

    config = get_config()

    # determine the best device to run on
    if t.cuda.is_available():
        device = 'cuda'
    elif t.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"{device=}")
    print(f'Config: {config}')

    # Define the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config['checkpoint'])
    # TODO make sure it doesn't add SEP tokens when there's a full stop
    base_model = AutoModel.from_pretrained(config['checkpoint'])

    # use_ia3_layers = False
    # if use_ia3_layers:
    #     from ia3_model_modifier import modify_with_ia3
    #     layers_to_replace_with_ia3 = "key|value|intermediate.dense"
    #     base_model = modify_with_ia3(base_model, layers_to_replace_with_ia3)

    # Load the train dataset
    train_dataloaders, val_dataloaders, train_head_dims = \
        multiple_dataset_loading(datapath, tokenizer, config)

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
        regularization_coef=config['regularization_coef'],
        dataset_names=config['train_datasets'],
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
        accelerator=device,
        devices=1,
        logger=logger,
        log_every_n_steps=1,
        default_root_dir=artifactspath,
        check_val_every_n_epoch=config['check_val_every_n_epoch'],
    )
    print('Fine tuning BERT...')
    # See documentation on multiple dataloaders here: https://pytorch-lightning.readthedocs.io/en/1.3.8/advanced/multiple_loaders.html
    trainer.fit(lit_model,
                train_dataloaders=train_dataloaders,
                val_dataloaders=val_dataloaders)

    # Test the model
    test_dataset_path = ethics_ds_path / config['test_set']
    tokens, masks, targets = load_csv_to_tensors(
        test_dataset_path, tokenizer, num_samples=config['num_samples_test'])
    data = TensorDataset(tokens, masks, targets)
    test_loader = DataLoader(
        data, batch_size=config['batch_size'], shuffle=config['shuffle_test'])

    print('Testing on ETHICS...')
    trainer.test(lit_model, dataloaders=test_loader)
    logger.save()
    print('Done')

    # Make prediction on a single test example
    # example_text = "I am a sentence."
    # prediction_dataloader = preprocess_prediction(
    #     [example_text], tokenizer, batch_size=1)
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
        if train_dataset not in ['ds000212'] and \
                not train_dataset.startswith('ethics'):
            raise ValueError(f"Invalid train dataset: {train_dataset}")
        if train_dataset == 'ethics' \
                and config['loss_names'][index] != 'cross-entropy':
            raise ValueError(f"Invalid loss for ethics dataset: "
                             f"{config['loss_names'][index]}. "
                             f"For classification can only use cross_entropy.")

    return config


def get_args() -> argparse.ArgumentParser:
    """Get command line arguments"""

    parser = argparse.ArgumentParser(
        description='run model training'
    )
    parser.add_argument(
        '--train_datasets',
        nargs='+',
        default=['ethics/commonsense/cm_train.csv'],
        type=str,
        help='Datasets to train on. This can be multiple datasets, '
             'e.g. "ds000212 ethics/...".'
             'Note that the same datasets are used for validation.'
    )

    parser.add_argument(
        '--normalize_fmri',
        default='True',
        type=str,
        help='Normalize fMRI data.'
    )
    parser.add_argument(
        '--num_epochs',
        default='10',
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
        default='32',
        type=int,
        help='Batch size.'
             '(default: 32)'
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
        '--shuffle_train',
        default='True',
        type=str,
        help='If we should shuffle train data.'
    )
    parser.add_argument(
        '--num_samples_test',
        default='100',
        type=int,
        help='Number of test samples.'
             '(default: 64)'
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
        '--test_set',
        default='commonsense/cm_test.csv',
        type=str,
        help='Path to test set starting from data/ethics directory.'
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
    parser.add_argument(
        '--fraction_train',
        default='0.9',
        type=float,
        help='Fraction of data to use for training.'
    )

    parser.add_argument(
        '--check_val_every_n_epoch',
        default='2',
        type=int,
        help='Check validation every n epochs.'
    )

    return parser


if __name__ == '__main__':
    main()
