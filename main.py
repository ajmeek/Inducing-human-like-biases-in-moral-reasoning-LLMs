import transformers
from transformers import pipeline
# from ethics.commonsense import tune
import torch

if __name__ == '__main__':

    # Check if GPU is available
    print(torch.cuda.is_available())

    # Example usage of hugging face transformers
    # In BERT we do classification. We take a sentence, and we classify it as acceptable or not.
    classifier = pipeline("sentiment-analysis", model='bert-base-uncased')
    result = classifier("The man worked as a carpenter")
    print(result)
