import torch
from transformers import AutoTokenizer, AutoModel
from data_placeholders import load_cm_with_reg_placeholder, load_cm_df
from model import BERT
from train import train_model, preprocess
from eval import test_accuracy, classifier
from utils.preprocessing import preprocess_prediction, preprocess_test

if __name__ == '__main__':
    # hyperparams
    num_epochs = 0
    batches_per_epoch = 12
    batch_size = 4
    checkpoint = 'bert-base-cased'  # Hugging Face model we'll be using
    regression_out_dims = (4, 20)
    head_dims = [2, regression_out_dims]  # Classification head and regression head
    only_train_head = True
    use_ia3_layers = True
    layers_to_replace_with_ia3 = "key|value|intermediate.dense"

    # determine the best device to run on
    if torch.cuda.is_available(): device = 'cuda'
    elif torch.backends.mps.is_available(): device = 'mps'
    else: device = 'cpu'
    print(f"{device=}")

    # load the data
    inputs, targets = load_cm_with_reg_placeholder(regression_out_dims)

    # define the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # TODO make sure it doesn't add SEP tokens when there's a full stop
    base_model = AutoModel.from_pretrained(checkpoint)
    if use_ia3_layers:
        from ia3_model_modifier import modify_with_ia3
        base_model = modify_with_ia3(base_model, layers_to_replace_with_ia3)

    model = BERT(base_model, head_dims=head_dims)

    # train the model
    trainer, lit_model = train_model(model, tokenizer, inputs, targets,
                loss_names=['cross-entropy', 'mse'],   # BE CAREFUL: If this doesn't have the same length as the number of heads, not all losses will be used and thus not all heads will be trained.
                only_train_head=only_train_head,
                num_epochs=num_epochs,
                batches_per_epoch=batches_per_epoch,
                batch_size=batch_size,
                device=device)

    # test the model and see same output.
    df = load_cm_df('test')
    inputs = df['input'].tolist()
    targets = df['label']
    loader = preprocess_test(inputs, targets, tokenizer, num_samples=10, batch_size=1)

    trainer.test(lit_model, dataloaders=loader)
    test_accuracy(model, tokenizer, max_samples=10, log_all=True)

    example_text = "I am a sentence."
    prediction = classifier(example_text, model, tokenizer)
    print(prediction)
    prediction_dataloader = preprocess_prediction([example_text], tokenizer, batch_size=1)
    prediction = trainer.predict(lit_model, prediction_dataloader)
    print(prediction)
