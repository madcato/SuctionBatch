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

Este submodulo es una adaptación de un transformador BERT para detectar frases tóxicas.

    $ git submodule add git@github.com:madcato/transformers.git

La técnica que presenta este repositorio es usar un modelo BERT, adaptando sus salidas y entradas para resolver un problema de clasificación, para el que originalmente no estaba entrenado.

# Transformer-man

Este proyecto es una investigación para ver cómo resolver el problema de **Q&A** en el contexto de ayuda de Linux (la utilidad `man`). Quizá se pueda presentar el problema como **Q&A** o como **translate** u otro sistemas. También hay que investigar cómo enseñar los manuales de `man`: una opción es entrenar de cero un BERT, otra sería reentrenar el BERT con los manuales(añadir más textos al modelo), la otra opción sería meter los manuales como entrada de la pregunta (esto requeriría detectar en un paso previo qué manual requiere la pregunta)

Este repo es un **BERT** implementado en **C++** portable -> [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)

# NLP with C++

This is a set of investigations of several NLP projects made in C++. All are made in try (.gitignored)

[NLP C++ projects list on Github](https://git- hub.com/keon/awesome-nlp#user-content-c++)

## MITIE: library and tools for information extraction
Este proyecto tiene buenas utilidades y parece fácil de compilar he integrar con otras apps C++. Su principal función es el **NER** *(Near Entity Recognition)*, detecta palabras que son: organizaciones, localizaciones, nombres, y otros tipos de entidades.

[github(mit-nlp/MITIE)](https://github.com/mit-nlp/MITIE)

La licencia permite el uso comercial. Tienes herramientas de entrenamiento, falta algo de documentación y ayuda para correr los ejemplos.

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
Este proyecto hace tokenizacion

[github(LanguageMachines/ucto)](https://github.com/LanguageMachines/ucto)

Tiene varias dependencias. Es mejor evitar integrar esto en AAI

## meta: A Modern C++ Data Sciences Toolkit
Este proyecto tiene multiples utilidaes: tokenizacion, clasificación...

[github(meta-toolkit/meta)](https://github.com/meta-toolkit/meta)

[meta: more info](https://meta-toolkit.org)

Da problemas de compilación.

## Starspace
Este proyecto merece la pena estudiarlo, al menos por su red neural en c++ (si realmente la tiene)

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

