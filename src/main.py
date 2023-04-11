import torch
from transformers import AutoTokenizer, AutoModel
from utils.loading_data import load_fmri_with_labels, load_np_fmri_to_tensor, load_csv_to_tensors
from utils.preprocessing import preprocess_prediction, preprocess
from model import BERT
from pl_model import LitBert
import lightning.pytorch as pl
from pathlib import Path


def main():
    data_path = Path('./data')
    assert data_path.exists(), 'Expected data dir present.'
    # Hyperparameters #
    # Training parameters
    train_with_cls_labels = True
    num_epochs = 16
    num_epochs_head_training = 4
    batches_per_epoch = 128
    batch_size = 8
    regularize = True
    regularization_coef = 1e-5

    # Model parameters
    checkpoint = 'bert-base-cased'  # Hugging Face model we'll be using
    if train_with_cls_labels:
        train_head_dims = [64, 3] # fMRI and classification (neutral, accidental, intentional)
    else:
        train_head_dims = [64]
    test_head_dims = [2]
    only_train_head = False
    use_ia3_layers = False
    layers_to_replace_with_ia3 = "key|value|intermediate.dense"

    # Loss parameters
    loss_names = ['mse', 'cross-entropy']  # cross-entropy, mse

    # Dataset parameters
    num_samples_train = 4096
    shuffle_train = True  # Set to False in order to get deterministic results and test overfitting on a small dataset.
    head_train_dataset_path = data_path / 'ethics/commonsense/cm_train.csv'
    test_dataset_path = data_path / 'ethics/commonsense/cm_test.csv'
    num_samples_test = 32

    # Logging
    log_every_n_steps = 1

    # determine the best device to run on
    if torch.cuda.is_available(): device = 'cuda'
    elif torch.backends.mps.is_available(): device = 'mps'
    else: device = 'cpu'
    print(f"{device=}")

    # Define the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # TODO make sure it doesn't add SEP tokens when there's a full stop
    base_model = AutoModel.from_pretrained(checkpoint)
    if use_ia3_layers:
        from ia3_model_modifier import modify_with_ia3
        base_model = modify_with_ia3(base_model, layers_to_replace_with_ia3)

    model = BERT(base_model, head_dims=train_head_dims)
    lit_model = LitBert(model, only_train_head, loss_names,
                        regularize_from_init=regularize,
                        regularization_coef=regularization_coef)

    # Get training dataloader
    if train_with_cls_labels:
        tokens, masks, targets = load_fmri_with_labels(data_path, tokenizer,
                                                       num_samples_train)
    else:
        tokens, masks, targets = load_np_fmri_to_tensor(data_path / 'ds000212_difumo',
                                                        tokenizer,
                                                        num_samples_train)
    train_loader = preprocess(tokens, masks, targets, train_head_dims, batch_size,
                              shuffle=shuffle_train)

    # train the model
    trainer = pl.Trainer(
        limit_train_batches=batches_per_epoch,
        max_epochs=num_epochs,
        accelerator=device,
        devices=1,
        log_every_n_steps=log_every_n_steps,
    )
    trainer.fit(lit_model, train_loader)

    # Use base model with new head for testing.
    model = BERT(base_model, test_head_dims)
    lit_model = LitBert(model, only_train_head=True)

    # Fine-tune the head on the ethics training set
    tokens, masks, targets = load_csv_to_tensors(head_train_dataset_path,
                                                 tokenizer, num_samples=1000)
    head_train_loader = preprocess(tokens, masks, targets, test_head_dims,
                              batch_size, shuffle=True)
    trainer = pl.Trainer(
        limit_train_batches=batches_per_epoch,
        max_epochs=num_epochs_head_training,
        accelerator=device,
        devices=1,
        log_every_n_steps=log_every_n_steps,
    )
    trainer.fit(lit_model, head_train_loader) 

    # Test the model
    tokens, masks, targets = load_csv_to_tensors(test_dataset_path, tokenizer,
                                                 num_samples=num_samples_test)
    test_loader = preprocess(tokens, masks, targets,
                             head_dims=test_head_dims, batch_size=batch_size)

    trainer.test(lit_model, dataloaders=test_loader)

    # Make prediction on a single test example
    example_text = "I am a sentence."
    prediction_dataloader = preprocess_prediction([example_text], tokenizer, batch_size=1)
    prediction = trainer.predict(lit_model, prediction_dataloader)


if __name__ == '__main__':
    main()
