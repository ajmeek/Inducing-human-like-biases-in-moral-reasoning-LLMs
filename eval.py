from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizer
from data_placeholders import load_cm_df

# inference from the model on text
def classifier(text: str, model: nn.Module, tokenizer: PreTrainedTokenizer) -> torch.Tensor:
    tokenized = tokenizer([text], padding='max_length', truncation=True)
    tokens = torch.tensor(tokenized['input_ids'])
    mask = torch.tensor(tokenized['attention_mask'])
    logits, reg_pred = model(tokens, mask)
    return F.softmax(logits, dim=-1)

# load the testing set and see how well our model performs on it
def test_accuracy(model: nn.Module, tokenizer: PreTrainedTokenizer,
                  max_samples=100, log_all=False):
    model.eval()
    df = load_cm_df('test')
    correct_results = 0
    total_results = 0
    for i, row in tqdm(df.iterrows()): # TODO: add batching for higher efficiency
        if i > max_samples: break
        text = row['input']
        probs = classifier(text, model, tokenizer)
        prediction, confidence = probs.argmax().item(), probs.max().item()
        label = row['label']
        if log_all:
            # print(f"\n{text=:<128} {prediction=:<4} ({confidence=:.4f}) {label=:<4}\n")
            print(prediction, confidence, label)
            pass
        if prediction == label: correct_results += 1
        total_results += 1
    print(f"\n\nAccuracy {correct_results/total_results:.4f}\n")
   