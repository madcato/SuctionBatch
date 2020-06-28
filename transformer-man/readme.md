# Transformer-man

Este proyecto es una investigación para ver cómo resolver el problema de **Q&A** en el contexto de ayuda de Linux (la utilidad `man`). Quizá se pueda presentar el problema como **Q&A** o como **translate** u otro sistemas. También hay que investigar cómo enseñar los manuales de `man`: una opción es entrenar de cero un BERT, otra sería reentrenar el BERT con los manuales(añadir más textos al modelo), la otra opción sería meter los manuales como entrada de la pregunta (esto requeriría detectar en un paso previo qué manual requiere la pregunta)

## Datos para entrenar 

#### En macOs

Ejecutar `$ man man` para buscar el archivo de configuración de man. (En macOS está en `/private/etc/man.conf`)

Directories are:

    MANPATH /usr/share/man
    MANPATH /usr/local/share/man
    MANPATH /usr/X11/man
    MANPATH /Library/Apple/usr/share/man

#### En Linux

    $ cp -r /usr/share/man/* .
 
## Opciones a probar

### Adaptaciones del model

- **Q&A** <-- *Opción a investigar actualmente*
- **Translate\_X\_Y**

### Usos de BERT

- Usar un BERT preentrenado.
- Usar un BERT preentrenado y reentrenarlo con los manuales.  <-- *Opción a investigar actualmente*
- Entrenar un BERT nuevo con los manuales.

# Tasks

- [ ] Buscar un ejemplo sobre como reentrenar el BERT
    Directorio: `./train-a-pre-trained-model`
- [ ] Buscar un ejemplo sobre Q&A

# BERT Adaptations

## Documentation links

- [Towards Machine Learning](https://towardsml.com/2019/09/17/bert-explained-a-complete-guide-with-theory-and-tutorial/)
- [BERT NLP — How To Build a Question Answering Bot](https://towardsdatascience.com/bert-nlp-how-to-build-a-question-answering-bot-98b1d1594d7b)
- [Notebook showing how to use the Transformers library for NLP](https://github.com/Spain-AI/transformers/blob/master/transformers-multiclass.ipynb)
- [Transformers: State-of-the-art Natural Language Processing for Pytorch and TensorFlow 2.0.](https://github.com/huggingface/transformers)

## Pipelines

`Pipeline` are high-level objects which automatically handle tokenization, running your data through a transformers model and outputting the result in a structured object.

- `feature-extraction`: Generates a tensor representation for the input sequence
- `ner`: Generates named entity mapping for each word in the input sequence.
- `sentiment-analysis`: Gives the polarity (positive / negative) of the whole input sequence.
- `text-classification`: Initialize a TextClassificationPipeline directly, or see sentiment-analysis for an example.
- `question-answering`: Provided some context and a question refering to the context, it will extract the answer to the question in the context.
- `fill-mask`: Takes an input sequence containing a masked token (e.g. <mask>) and return list of most probable filled sequences, with their probabilities.
- `summarization`
- `translation_xx_to_yy`