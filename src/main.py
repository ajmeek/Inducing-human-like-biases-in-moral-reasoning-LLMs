import torch
from transformers import AutoTokenizer, AutoModel
from utils.loading_csv import load_csv_to_tensors
from utils.preprocessing import preprocess_prediction, preprocess
from model import BERT
from pl_model import LitBert
import lightning.pytorch as pl
from pathlib import Path


def main():
    datapath = Path('./data')
    assert datapath.exists(), 'Expected data dir present.'
    ethics_ds_path = datapath / 'ethics'
    # Hyperparameters #
    # Training parameters
    num_epochs = 10
    batches_per_epoch = 12
    batch_size = 4

    # Model parameters
    checkpoint = 'bert-base-cased'  # Hugging Face model we'll be using
    head_dims = [2, (4, 20)]  # Classification head and regression head
    only_train_head = True
    use_ia3_layers = True
    layers_to_replace_with_ia3 = "key|value|intermediate.dense"

    # Loss parameters
    loss_names = ['cross-entropy', 'mse']
    loss_weights = [1, 1]
    regularize_from_init = True
    regularization_coef = 1e-2

    # Dataset parameters
    train_dataset = ethics_ds_path / 'commonsense/cm_train.csv'
    num_samples_train = 100
    test_dataset = ethics_ds_path / 'commonsense/cm_test.csv'
    num_samples_test = 10

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

    model = BERT(base_model, head_dims=head_dims)
    lit_model = LitBert(model, only_train_head, loss_names, loss_weights,
                        regularize_from_init=regularize_from_init, regularization_coef=regularization_coef)

    # Get training dataloader
    tokens, masks, targets = load_csv_to_tensors(train_dataset, tokenizer, num_samples=num_samples_train)
    train_loader = preprocess(tokens, masks, targets, head_dims, batch_size, shuffle=True)

    # train the model
    trainer = pl.Trainer(
        limit_train_batches=batches_per_epoch,
        max_epochs=num_epochs,
        accelerator=device,
        devices=1,
    )
    trainer.fit(lit_model, train_loader)

    # Test the model
    tokens, masks, targets = load_csv_to_tensors(test_dataset, tokenizer, num_samples=num_samples_test)
    test_loader = preprocess(tokens, masks, targets, head_dims=[2], batch_size=1, shuffle=False)  # only test the classification head

    trainer.test(lit_model, dataloaders=test_loader)

    # Make prediction on a single test example
    example_text = "I am a sentence."
    prediction_dataloader = preprocess_prediction([example_text], tokenizer, batch_size=1)
    prediction = trainer.predict(lit_model, prediction_dataloader)

if __name__ == '__main__':
    main()