# SuctionBatch

This project is an adaptation of the GTP2 ML algorithm to learn code instea of news articles.

## Prerequesites

- Ruby
- [octokit](http://octokit.github.io/octokit.rb/)

### Isntallation

Install Octokit:

    gem install octokit

## Configuration

### Download code to learn from Github

    $ cd code
    $ ./downloadCodeFromGithub.rb 10000 Ruby

Change extension to txt

    $ find . -name *.rb -exec mv {} {}.txt \;

You must create three directories: 'train', 'test', 'valid' and distribute the codes in them.


## transformer-lm adaptation

- [Transformer language model (GPT-2) with sentencepiece tokenizer](https://github.com/lopuhin/transformer-lm#id1)

### Prepare data fro training

Change directory to:

    $ cd transformer-lm
cd 
Configure project:

    $ pip3 install -r requirements.txt
    $ python3 setup.py develop


Train:

    $ sp-train ../code/data/ruby_small ruby_small.txt ../code/data/ruby_small/ruby_small-model

This generate files:
- transformer-lm/ruby_small.txt  -> Can be deleted
- code/data/ruby_small/ruby_smal-encoded.model
- code/data/ruby_small/ruby_smal-encoded.vocab

Encode:

    $ sp-encode ../code/data/ruby_small ../code/data/ruby_small/ruby_small-model.model ../code/data/ruby_small/encoded

This step generates files: 
- Saving encoded split train to ../code/data/ruby_small/encoded/train.npy
- Saving encoded split valid to ../code/data/ruby_small/encoded/valid.npy
- Saving encoded split test to ../code/data/ruby_small/encoded/test.npy

### Train model

    $ gpt-2 run-root ../code/data/ruby_small/encoded ../code/data/ruby_small/ruby_small-model.model

On Bolt with GPU:

gpt-2 run-root ../code/data/encoded ../code/data/swift1-model.model --batch-size 2 --g-accum-gradients 1 --n_ctx 400

## Try!

    $ gpt-2-gen run-root "def uri_for page"

# GPT2 

- [Transformer language model (GPT-2) with sentencepiece tokenizer](https://github.com/lopuhin/transformer-lm#id1)
- [Code for the paper "Language Models are Unsupervised Multitask Learners"](https://github.com/nshepperd/gpt-2)
- [How To Make Custom AI-Generated Text With GPT-2](https://minimaxir.com/2019/09/howto-gpt2/)
- [Transformers: State-of-the-art Natural Language Processing for TensorFlow 2.0 and PyTorch.](https://github.com/huggingface/transformers) w547 s26.1 f6.2

# Doc

- [Documentation for Ruby: 2.7.0](https://docs.ruby-lang.org/en/2.7.0/)
- [Ruby stdlib: 2.7.0](https://ruby-doc.org/stdlib-2.7.1/)
- [Ruby core: 2.7.1](https://ruby-doc.org/core-2.7.1/)