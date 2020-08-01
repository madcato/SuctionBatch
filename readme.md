# SuctionBatch

This project is an adaptation of the GTP2 ML algorithm to learn code instea of news articles.

* Follow this link to see a sample implementation of Trasnformer: [Transformer model for language understanding](https://www.tensorflow.org/tutorials/text/transformer)

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

### Prepare data for training

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

# sqlite3

## Build sqlite3 with R-Tree

    $ ./configure --enable-rtree
    $ make

## Usage of R-Tree

- [The SQLite R*Tree Module](https://www.sqlite.org/rtree.html)

### Create table

    CREATE VIRTUAL TABLE <name> USING rtree(<column-names>);

This create four tables.

Sample:

    CREATE VIRTUAL TABLE demo_index USING rtree(
       id,              -- Integer primary key
       minX, maxX,      -- Minimum and maximum X coordinate
       minY, maxY       -- Minimum and maximum Y coordinate
    );

Insert data:

    INSERT INTO demo_index VALUES(
        1,                   -- Primary key -- SQLite.org headquarters
        -80.7749, -80.7747,  -- Longitude range
        35.3776, 35.3778     -- Latitude range
    );
    INSERT INTO demo_index VALUES(
        2,                   -- NC 12th Congressional District in 2010
        -81.0, -79.6,
        35.0, 36.2
    );

Query: 

    SELECT id FROM demo_index
     WHERE minX>=-81.08 AND maxX<=-80.58
       AND minY>=35.00  AND maxY<=35.44;

Use a circle to query: 

    SELECT id FROM demo_index WHERE id MATCH circle(45.3, 22.9, 5.0)

circle inposition x: 45.3 y: 22.9 radius: 5.0


# Submodulo transformers

This submodule is a BERT transformer adaptation to detect toxic phrases.

    $ git submodule add git@github.com:madcato/transformers.git

The technique in this repository is to use a BERT model, adapting its inputs/outputs to solve a text classification model, wich originally was not trained.

# Transformer-man

This project is an investigation to solve a **Q&A** in the Linux manual context (`man` utility). Maybe can be presented as a **Q%A** solutions or a **translate** or other system. Also I must investigate how to train `man` manuals: one possibility is to train a BERT from zero, another is adapt a BERT with the manuals (adding more texts to model), another option would be to put the manual as part of the input (this requires to find the manual page before adding it)

This is a **BERT** repo implemented portable **C++** -> [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)

# NLP with C++

This is a set of investigations of several NLP projects made in C++. All are made in try (.gitignored)

[NLP C++ projects list on Github](https://git- hub.com/keon/awesome-nlp#user-content-c++)

## MITIE: library and tools for information extraction
This project contains good utilities and it seems easy to integrate with other C++ projects. It proncipal operation is the **NER* *(Near Entity Recognition)*, it detects words like: organizatoins, places, names, and oother types of entities.

[github(mit-nlp/MITIE)](https://github.com/mit-nlp/MITIE)

Comercial use allowed. It has training code, lacks a bit of documentation and some help to run the samples.

### sample ner_stream

From:

    A Pegasus Airlines plane landed at an Istanbul airport Friday after a passenger "said that there was a bomb on board" 
    and wanted the plane to land in Sochi, Russia, the site of the Winter Olympics, said officials with Turkey's 
    Transportation Ministry.

Generates:

    A [ORGANIZATION Pegasus Airlines] plane landed at an [LOCATION Istanbul] airport Friday after a passenger " 
    said that there was a bomb on board " and wanted the plane to land in [LOCATION Sochi] , [LOCATION Russia] , 
    the site of the Winter Olympics , said officials with [LOCATION Turkey] 's [ORGANIZATION Transportation Ministry] . 

## Unicode tokeniser. Ucto tokenizes text files
This project to **tokenization**.

[github(LanguageMachines/ucto)](https://github.com/LanguageMachines/ucto)

It has some dependencies.

## meta: A Modern C++ Data Sciences Toolkit
This project has some utilities: **tokenization**, **classification**,...

[github(meta-toolkit/meta)](https://github.com/meta-toolkit/meta)

[meta: more info](https://meta-toolkit.org)

Compiling problems

## Starspace

[facebookresearch/StarSpace](https://github.com/facebookresearch/StarSpace)

Require boost

# Fine-tuning BERT
In directory *./fine-tuning* there are 5 different projects that realize a fine-tuning of the BERT model. All try to use **DistilBERT** to make learning fast.

# Q&A
En el directorio `qa` incluyo un proyecto para realizar un **Q&A** destinado a servir como buscador de información de TMB. Este proyecto se usa en el proyecto TMBInfo como motor de búsqueda.

- [Sample code from which generated my code](https://github.com/huggingface/transformers/tree/master/examples/question-answering)

## Run 

    export SQUAD_DIR=./SQUAD
    
    python3 run_squad.py \
      --model_type bert \
      --model_name_or_path bert-base-uncased \
      --do_train \
      --do_eval \
      --do_lower_case \
      --train_file $SQUAD_DIR/train-v1.1.json \
      --predict_file $SQUAD_DIR/dev-v1.1.json \
      --per_gpu_train_batch_size 12 \
      --learning_rate 3e-5 \
      --num_train_epochs 2.0 \
      --max_seq_length 384 \
      --doc_stride 128 \
      --output_dir ./debug_squad/

## Sample usage

### Load
- [From this post](https://qa.fastforwardlabs.com/pytorch/hugging%20face/wikipedia/bert/transformers/2020/05/19/Getting_Started_with_QA.html)

  Run `$ python3 sample.py`
- [Find pre-trained models here](https://huggingface.co/models)

