import torch as t
from transformers import AutoTokenizer, AutoModel
from utils.loading_data import load_csv_to_tensors, load_np_fmri_to_tensor, load_ds000212_dataset
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
    if t.cuda.is_available(): device = 'cuda'
    elif t.backends.mps.is_available(): device = 'mps'
    else: device = 'cpu'
    print(f"{device=}")

    # Define the tokenizer and model
    checkpoint='bert-base-cased'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # TODO make sure it doesn't add SEP tokens when there's a full stop
    base_model = AutoModel.from_pretrained(checkpoint)

    #use_ia3_layers = False
    # if use_ia3_layers:
    #     from ia3_model_modifier import modify_with_ia3
    #     layers_to_replace_with_ia3 = "key|value|intermediate.dense"
    #     base_model = modify_with_ia3(base_model, layers_to_replace_with_ia3)

    tokens, masks, targets = load_ds000212_dataset(datapath, tokenizer, config['num_samples_train'], normalize=False)
    train_head_dims = [e.shape[1] for e in targets] #[64]  # Classification head and regression head, for example [2, (10, 4)]
    model = BERT(
        base_model,
        head_dims=train_head_dims
    )
    loss_names = ['mse'] #['cross-entropy']  # cross-entropy, mse
    lit_model = LitBert(
        model,
        config['only_train_head'],
        loss_names,
        loss_weights=[1.0],
        regularize_from_init=config['regularize_from_init'],
        regularization_coef=config['regularization_coef']
    )

    # Get training dataloader
    # if train_head_dims[0] == 2:  # TODO: this is a bit hacky, not sure when we want to use what.
    #     # For now if the first head has two outputs we use the ethics dataset and otherwise the fMRI dataset.
    #     tokens, masks, targets = load_csv_to_tensors(ethics_ds_path / 'commonsense/cm_train.csv', tokenizer, num_samples=num_samples_train)
    # else:
    #     tokens, masks, targets = load_np_fmri_to_tensor(difumo_ds_path, tokenizer, num_samples=num_samples_train)
    train_loader = preprocess(tokens, masks, targets, train_head_dims, config['batch_size'], shuffle=False)

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
        default_root_dir=artifactspath
    )
    print('Fine tuning BERT...')
    trainer.fit(lit_model, train_loader)

    # Use base model with new head for testing.
    trained_base_model = trainer.model.model.base
    test_head_dims = [2]
    model = BERT(trained_base_model, head_dims=test_head_dims)
    lit_model = LitBert(model, config['only_train_head'])  # losses are not needed for testing

    # Test the model
    test_dataset_path = ethics_ds_path / 'commonsense/cm_train.csv'
    tokens, masks, targets = load_csv_to_tensors(test_dataset_path, tokenizer, num_samples=config['num_samples_test'])
    test_loader = preprocess(tokens, masks, targets, head_dims=test_head_dims, batch_size=config['batch_size'], shuffle=False)

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
        if config[arg] in {'True', 'False'}:
            config[arg] = config[arg] == 'True'
        elif config[arg] == 'none':
            config[arg] = None
        elif 'subjects_per_dataset' in arg:
            config[arg] = None if config[arg] == -1 else config[arg]
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
        default='1',
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
        '--num_samples_test',
        default='64',
        type=int,
        help='Number of test samples.'
             '(default: 64)'
    )
    parser.add_argument(
        '--only_train_head',
        default='True',
        type=str,
        help='Train only attached head.'
             '(default: True)'
    )

    return parser

if __name__ == '__main__':
    main()
