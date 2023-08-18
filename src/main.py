from datetime import datetime
from lightning.pytorch.loggers import WandbLogger
from model import BERT
from os import environ
from pathlib import Path
from pl_model import LitBert
from pprint import pprint
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from utils.EthicsDataset import EthicsDataset
from utils.constants import Sampling
from utils.loading_data import multiple_dataset_loading, DEFAULT_DATASETS

import argparse
import lightning.pytorch as pl
import wandb

def train(context):
    pprint('Context:')
    pprint(context, indent=2)
    # Define the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(context['checkpoint'])
    # TODO make sure it doesn't add SEP tokens when there's a full stop
    base_model = AutoModel.from_pretrained(context['checkpoint'])

    #use_ia3_layers = False
    # if use_ia3_layers:
    #     from ia3_model_modifier import modify_with_ia3
    #     layers_to_replace_with_ia3 = "key|value|intermediate.dense"
    #     base_model = modify_with_ia3(base_model, layers_to_replace_with_ia3)

    # Load the dataset
    dataloaders, train_head_dims = multiple_dataset_loading(tokenizer, context)

    # Define the model
    model = BERT(
        base_model,
        head_dims=train_head_dims
    )
    lit_model = LitBert(model, context)

    logger = WandbLogger(save_dir=context['artifactspath'], project="AISC_BB")
    logger.log_hyperparams(context)

    # train the model
    trainer = pl.Trainer(
        limit_train_batches=context['batches_per_epoch'],
        max_epochs=context['num_epochs'],
        accelerator='auto',
        devices='auto',
        strategy='auto',
        logger=logger,
        log_every_n_steps=1,
        default_root_dir=context['artifactspath'],
        enable_checkpointing=False  # Avoid saving full model into a disk (GBs)
    )
    print('Fine tuning BERT...')
    trainer.fit(lit_model, dataloaders)

    # Test the model
    data = EthicsDataset(context, tokenizer, is_train=False)
    test_loader = DataLoader( data, batch_size=context['batch_size'], shuffle=context['shuffle_test'])
    print('Testing on ETHICS...')
    trainer.test(lit_model, dataloaders=test_loader)
    logger.save()
    wandb.finish()
    trainer.save_checkpoint(context['artifactspath'] / f'model-{datetime.utcnow().isoformat(timespec="minutes").replace(":","")}.ckpt')


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
        if train_dataset == 'ethics' and config['loss_names'][index] != 'cross-entropy':
            raise ValueError(f"Invalid loss for ethics dataset: {config['loss_names'][index]}. "
                             f"For classification can only use cross_entropy.")

    return config

def get_args() -> argparse.ArgumentParser:
    """Get command line arguments"""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='run model training'
    )
    parser.add_argument(
        '--num_epochs',
        default='1',
        type=int,
        help='Number of epochs to fine tune a model on fMRI data.'
    )
    parser.add_argument(
        '--batches_per_epoch',
        default='15',
        type=int,
        help='Batches per epoch.'
    )
    parser.add_argument(
        '--batch_size',
        default='15',
        type=int,
        help='Batch size.'
    )
    parser.add_argument(
        '--regularize_from_init',
        default='False',
        type=str,
        help='Regularize from init (base) model.'
    )
    parser.add_argument(
        '--regularization_coef',
        default='0.1',
        type=float,
        help='Regularization from init coef.'
    )
    parser.add_argument(
        '--num_samples_train',
        default='100',
        type=int,
        help='Number of train samples (fine tuning).'
    )
    parser.add_argument(
        '--num_samples_test',
        default='100',
        type=int,
        help='Number of test samples.'
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
    )
    parser.add_argument(
        '--checkpoint',
        default='bert-base-cased',
        type=str,
        help='HuggingFace model.'
    )
    parser.add_argument(
        '--train_datasets',
        nargs='+',
        default=DEFAULT_DATASETS,
        type=str,
        help='Datasets to train on.'
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
        '--checkpointing',
        nargs='+',
        default=False,
        type=bool,
        help='Save checkpoints of model after fine tuning. Enable to compare and contrast with brain scores later'
    )
    parser.add_argument(
        '--calculate_brain_scores',
        nargs='+',
        default=False,
        type=bool,
        help='By default this will calculate brain scores on the latest saved checkpoint.'
    )

    parser.add_argument(
        '--lr',
        default=0.0006538379548447884,
        type=float,
        help='Learning rate'
    )
    parser.add_argument(
        '--sampling_method',
        default=Sampling.LAST.name,
        choices=Sampling,
        type=lambda v: Sampling[v.replace('Sampling.', '')],
        help='Method for sampling fMRI data.'
    )
    parser.add_argument(
        '--datapath',
        default=Path(environ.get('AISCBB_DATA_DIR','./data')),
        type=str,
        help='Path to the folder with datasets.'
    )
    parser.add_argument(
        '--artifactspath',
        default=Path(environ.get('AISCBB_ARTIFACTS_DIR','./artifacts')),
        type=str,
        help='Path to the folder for artifacts.'
    )
    return parser


if __name__ == '__main__':
    context = get_config()
    assert context['datapath'].exists(), 'Expected data dir present.'
    context['artifactspath'].mkdir(exist_ok=True)
    train(context)
    print('Done')
