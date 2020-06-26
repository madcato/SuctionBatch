# Train a pre-trained model

Is this directory there are code to train a already trained model [following this script from transformers github repo](https://github.com/huggingface/transformers#Quick-tour-TF-20-training-and-PyTorch-interoperability)

Use distilled-bert in order to train this model in a **`GeForce GTX 1660 6GB` GPU**.

The texts to train this model are `man` command help texts. To find this files see `man man`.

## Install 

    $ pip3 install tensorflow-gpu==2.0.0
    $ pip3 install transformers
    $ pip3 install tensorflow_datasets
    $ pip3 install pytorch-pretrained-bert

## Configure
  
First copy `man` help texts into `../data` directory.

## Run

    $ python3 train.py
