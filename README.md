# FineTune-DistilBERT
HuggingFace Transformers:  Fine-tuning DistilBERT for Binary Classification Tasks

## About
Creating high-performing natural language models is as time-consuming as it is expensive,
but recent advances in transfer learning as applied to the domain of NLP have made it easy
for companies to use pretrained models for their natural language tasks. 

In this repository, we propose code to be used as a reference point for fine-tuning pretrained models
from the [HuggingFace Transformers Library](https://github.com/huggingface/transformers) on binary classification tasks using TF 2.0.

Specifically, we will be using:
1. [Comet.ml](https://www.comet.ml/site/) as our experimentation framework
2. [DistilBERT](https://medium.com/huggingface/distilbert-8cf3380435b5)--a distilled version of BERT

to predict toxic comments on a modified version of the [Jigsaw Toxic Comment](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) dataset on Kaggle.
