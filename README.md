### Transfromer From Scratch with Pytorch

The transfromer model architecture is in the `model.py` file. The dataset file contains the data loader to train the transfromer and the `train.py` is the code used to train it. We trained the model on a translation task using `opus_books` dataset from Hugging Face.
You can choose many languages to train the model on, you just have to edit the config file with the src and target language.