import torch as t
from transformers import AutoTokenizer, AutoModel
from utils.loading_data import load_csv_to_tensors, load_np_fmri_to_tensor
from utils.preprocessing import preprocess_prediction, preprocess
from model import BERT
from pl_model import LitBert
import lightning.pytorch as pl
from pathlib import Path
import pandas as pd
import numpy as np

datapath = Path('./data')

def main():
    assert datapath.exists(), 'Expected data dir present.'
    ethics_ds_path = datapath / 'ethics'
    artifactspath = Path('./artifacts')
    artifactspath.mkdir(exist_ok=True)
    difumo_ds_path = datapath / 'ds000212_difumo'
    # Hyperparameters #
    # Training parameters
    num_epochs = 10
    batches_per_epoch = 1
    batch_size = 32

    # Model parameters
    checkpoint = 'bert-base-cased'  # Hugging Face model we'll be using
    train_head_dims = [64]  # Classification head and regression head, for example [2, (10, 4)]
    test_head_dims = [2]
    only_train_head = True
    use_ia3_layers = False
    layers_to_replace_with_ia3 = "key|value|intermediate.dense"

    # Loss parameters
    loss_names = ['cross-entropy']  # cross-entropy, mse
    loss_weights = [1]

    # Dataset parameters
    # train_dataset_path = ethics_ds_path / 'commonsense/cm_train.csv'
    train_dataset_path = difumo_ds_path
    num_samples_train = 32
    shuffle_train = True  # Set to False in order to get deterministic results and test overfitting on a small dataset.
    test_dataset_path = ethics_ds_path / 'commonsense/cm_train.csv'
    # test_dataset_path = difumo_ds_path
    num_samples_test = 32
    shuffle_test = False

    # Logging
    log_every_n_steps = 1

    # determine the best device to run on
    if t.cuda.is_available(): device = 'cuda'
    elif t.backends.mps.is_available(): device = 'mps'
    else: device = 'cpu'
    print(f"{device=}")

    # Define the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # TODO make sure it doesn't add SEP tokens when there's a full stop
    base_model = AutoModel.from_pretrained(checkpoint)
    # if use_ia3_layers:
    #     from ia3_model_modifier import modify_with_ia3
    #     base_model = modify_with_ia3(base_model, layers_to_replace_with_ia3)

    ds000212 = load_ds000212_dataset()    
    ds000212_shape = ds000212['outputs'].shape

    model = BERT(base_model, head_dims=train_head_dims)
    lit_model = LitBert(model, only_train_head, loss_names, loss_weights)

    # Get training dataloader
    if train_head_dims[0] == 2:  # TODO: this is a bit hacky, not sure when we want to use what.
        # For now if the first head has two outputs we use the ethics dataset and otherwise the fMRI dataset.
        tokens, masks, targets = load_csv_to_tensors(train_dataset_path, tokenizer, num_samples=num_samples_train)
    else:
        tokens, masks, targets = load_np_fmri_to_tensor(train_dataset_path, tokenizer, num_samples=num_samples_train)
    train_loader = preprocess(tokens, masks, targets, train_head_dims, batch_size, shuffle=shuffle_train)

    # train the model
    trainer = pl.Trainer(
        limit_train_batches=batches_per_epoch,
        max_epochs=num_epochs,
        accelerator=device,
        devices=1,
        default_root_dir=artifactspath / 'lightning_logs',
        log_every_n_steps=log_every_n_steps
    )
    trainer.fit(lit_model, train_loader)

    # Use base model with new head for testing.
    trained_base_model = trainer.model.model.base
    model = BERT(trained_base_model, head_dims=test_head_dims)
    lit_model = LitBert(model, only_train_head)  # losses are not needed for testing

    # Test the model
    tokens, masks, targets = load_csv_to_tensors(test_dataset_path, tokenizer, num_samples=num_samples_test)
    test_loader = preprocess(tokens, masks, targets, head_dims=test_head_dims, batch_size=batch_size, shuffle=shuffle_test)

    trainer.test(lit_model, dataloaders=test_loader)

    # Make prediction on a single test example
    example_text = "I am a sentence."
    prediction_dataloader = preprocess_prediction([example_text], tokenizer, batch_size=1)
    prediction = trainer.predict(lit_model, prediction_dataloader)


def load_ds000212_dataset():
    assert datapath.exists()
    scenarios = []
    fmri_items = []
    for subject_dir in Path(datapath / 'functional_flattened').glob('sub-*'):
        for runpath in subject_dir.glob('[0-9]*.npy'):
            scenario_path = runpath.parent / f'labels-{runpath.name}'
            fmri_items += np.load(runpath.resolve()).tolist()
            scenarios += np.load(scenario_path.resolve()).tolist()
    assert len(scenarios) == len(fmri_items), f'Expected: {len(scenarios)} == {len(fmri_items)}'
    # Drop those of inconsistent len:
    from collections import Counter
    counts = Counter(len(e) for e in fmri_items)
    most_common_len = counts.most_common()[0][0]
    indeces = [i for i, e in enumerate(fmri_items) if len(e) == most_common_len] 
    scenarios = [e for i,e in enumerate(scenarios) if i in indeces]
    fmri_items = [e for i,e in enumerate(fmri_items) if i in indeces]
    return {'inputs': scenarios, 'outputs': t.tensor(fmri_items)}



if __name__ == '__main__':
    main()
