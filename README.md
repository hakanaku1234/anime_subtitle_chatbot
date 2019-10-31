# Anime Subtitle Chatbot

`tensor2tensor` implementation of using the [Anime Subtitles Dataset](https://www.kaggle.com/jef1056/anime-subtitles).

Based on `tensor2tensor`'s official and well-written [poetry generation](https://cloud.google.com/blog/products/gcp/cloud-poetry-training-and-hyperparameter-tuning-custom-text-models-on-cloud-ml-engine) tutorial. To understand this repository it is useful to read that tutorial because it explains most of the code.

## Pre-trained Model

A [pre-trained model](https://www.kaggle.com/waifuai/anime-subtitle-chatbot-pretrained-model) is hosted on Kaggle. Warning: This current version behaves very poorly and only says "I'm sorry" or "What's wrong", because those two phrases occur very frequently in the dataset it was trained on. However, we still uploaded it because it is a useful reference for anyone who was wondering what would it look like if someone trained a model based on the dataset.
