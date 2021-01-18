# FineTune-DistilBERT :hugs:
HuggingFace Transformers:  Fine-tuning DistilBERT for Binary Classification Tasks

## About
Creating high-performing natural language models is as time-consuming as it is expensive,
but recent advances in transfer learning as applied to the domain of NLP have made it easy
for companies to use pretrained models for their natural language tasks. 

In this repository, we propose code to be used as a reference point for fine-tuning pretrained models
from the [HuggingFace Transformers Library](https://github.com/huggingface/transformers) on binary classification tasks using TF 2.0.

Specifically, we will be using:
1. [Comet.ml](https://www.comet.ml/site/) as our experimentation framework
2. [nlpaug](https://github.com/makcedward/nlpaug) for data augmentation
2. [DistilBERT](https://medium.com/huggingface/distilbert-8cf3380435b5)--a distilled version of BERT

to predict toxic comments on a modified version of the [Jigsaw Toxic Comment](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) dataset on Kaggle.

## Datasets

As mentioned previously, the datasets used to train our models are based on the Jigsaw Toxic Comment dataset found on Kaggle.
This dataset has labels intended for a multi-label classification task (e.g. Toxic, Severe Toxic, Obscene, Threat, Insult, Identity Hate),
but we decided against using these labels due to their subjectivity.  

Instead, we converted the original dataset to a binary classification task where labels are either classified as 
toxic ('isToxic == 1) or non-toxic ('isToxic == 0).  Toxic comments make up 9.58% of this intermediate dataset.

To get our `unbalanced` dataset, we undersampled the majority class of this intermediate dataset until toxic comments make up 20.15% of all data.

To get our `balanced` dataset, we used [nlpaug](https://github.com/makcedward/nlpaug) to augment the minority class of the `unbalanced` dataset
until we reached a 50-50 class distribution.  Text augmentation was performed with synonym replacement using BERT embeddings.

*Any files or folders with `unbalanced` or `balanced` in the name is in relation to these two datasets.*

## Results

Both models follow the same architecture.  That is:

> [DistilBERT CLS Token Layer] + [Dense 256] + [Dense 32] + [Single Output Layer]

The only difference is whether they were trained on the `balanced` dataset with `binary_crossentropy` loss
or the `unbalanced` dataset with `focal_loss`.

The results are as follows:

**balanced_model**

1. Test Accuracy:  0.8801 
2. Test AUC-ROC:   0.9656

**unbalanced_model**

1. Test Accuracy:  0.9218
2. Test AUC-ROC:   0.9691

## Credits

@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}

@misc{ma2019nlpaug,
  title={NLP Augmentation},
  author={Edward Ma},
  howpublished={https://github.com/makcedward/nlpaug},
  year={2019}
}

@misc{Kaggle,
  title={Toxic Comment Classification Challenge},
  author={Jigsaw / Conversation AI},
  howpublished={https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data},
  year={2017}
}
