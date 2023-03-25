import torch as t
from transformers import AutoTokenizer, AutoModel
from utils.loading_csv import load_csv_to_tensors
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
    # Hyperparameters #
    # Training parameters
    num_epochs = 10
    batches_per_epoch = 12
    batch_size = 4

    # Model parameters
    checkpoint = 'bert-base-cased'  # Hugging Face model we'll be using
    layers_to_replace_with_ia3 = "key|value|intermediate.dense"

    # Dataset parameters
    train_dataset = ethics_ds_path / 'commonsense/cm_train.csv'
    num_samples_train = 100
    est_dataset = ethics_ds_path / 'commonsense/cm_test.csv'
    num_samples_test = 10

    # determine the best device to run on
    if t.cuda.is_available(): device = 'cuda'
    elif t.backends.mps.is_available(): device = 'mps'
    else: device = 'cpu'
    print(f"{device=}")

    # Define the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # TODO make sure it doesn't add SEP tokens when there's a full stop
    base_model = AutoModel.from_pretrained(checkpoint)
    # use_ia3_layers = False
    # if use_ia3_layers:
    #     from ia3_model_modifier import modify_with_ia3
    #     base_model = modify_with_ia3(base_model, layers_to_replace_with_ia3)

    ds000212 = load_ds000212_dataset()    
    ds000212_shape = ds000212['outputs'].shape


    head_dims = [2, ds000212_shape[1]]  # Classification head and regression head
    loss_names = ['cross-entropy', 'mse']
    loss_weights = [1, 1]
    model = BERT(base_model, head_dims=head_dims)
    only_train_head = True
    lit_model = LitBert(model, only_train_head, loss_names, loss_weights)

    # Get training dataloader
    tokens, masks, targets = load_csv_to_tensors(train_dataset, tokenizer, num_samples=num_samples_train)

    # Input for ds000212 data (scenarios,fmri):
    # Scenarios:
    tokenized_scenarios = tokenizer(ds000212['inputs'], padding='max_length', truncation=True)
    scenario_tokens = t.tensor(tokenized_scenarios['input_ids'])
    scenario_masks = t.tensor(tokenized_scenarios['attention_mask'])
    tokens = t.concat((tokens, scenario_tokens))
    masks = t.concat((masks, scenario_masks))
    # fMRI:
    targets.append(t.zeros((targets[0].shape[0], ds000212_shape[1])))
    targets[0] = t.concat((targets[0], t.zeros((ds000212_shape[0],)).int()))
    targets[1] = t.concat((targets[1], ds000212['outputs']))

    train_loader = preprocess(tokens, masks, targets, head_dims, batch_size, shuffle=True)

    # train the model
    trainer = pl.Trainer(
        limit_train_batches=batches_per_epoch,
        max_epochs=num_epochs,
        accelerator=device,
        devices=1,
        default_root_dir=artifactspath / 'lightning_logs'
    )
    trainer.fit(lit_model, train_loader)

    # Test the model
    tokens, masks, targets = load_csv_to_tensors(test_dataset, tokenizer, num_samples=num_samples_test)
    test_loader = preprocess(tokens, masks, targets, head_dims=[2], batch_size=1, shuffle=False)  # only test the classification head
    metrics = trainer.test(lit_model, dataloaders=test_loader, verbose=True)
    print(metrics)
    # prediction_dataloader = preprocess_prediction([example_text], tokenizer, batch_size=1)
    true_num = 0
    count = 0
    for in_batch, labels_batch in test_loader:
        predictions = trainer.predict(lit_model, test_loader)
        true_num += (predictions.max(axis=0).indeces == labels_batch).sum()
        count += len(labels_batch)
    print(f'Accuracty: {true_num / count}')


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