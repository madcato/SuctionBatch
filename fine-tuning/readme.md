# Fine-tuning BERT

In this directory there are 5 different projects that realize a fine-tuning of the BERT model. All try to use **DistilBERT** to make learning fast.

All use this project: [github(huggingface/transformers)](https://github.com/huggingface/transformers)

- First install and read the instruction to train models from: [transformers installation](https://github.com/huggingface/transformers#run-the-examples)

## Projects

List of projects and its subdirectory:

- [transformers-multiclass](https://github.com/Spain-AI/transformers/blob/master/transformers-multiclass.ipynb) ->  *spain-ai-multiclass-transformer*.
- [Question Answering with a Fine-Tuned BERT](https://mccormickml.com/2020/03/10/question-answering-with-a-fine-tuned-BERT/) -> *q&a*
- [seq2seq to summarization](https://github.com/huggingface/transformers/tree/master/examples/seq2seq) -> *seq2seq-summarization*
- [text-classification](https://github.com/huggingface/transformers/tree/master/examples/text-classification) -> *text-classification*
- [text-generation](https://github.com/huggingface/transformers/tree/master/examples/text-generation) -> *text-generation*

### transformers-multiclass


### Question Answering with a Fine-Tuned BERT (Q&A)

- There is another example of [question-answering here](https://github.com/huggingface/transformers/tree/master/examples/question-answering)
- Another [long form question answering here](https://github.com/huggingface/transformers/tree/master/examples/longform-qa)

### seq2seq to summarization

### text-classification

### text-generation

## How to fine-tuning transformers

1. First select one model:
  
        #          model_class      | tokenizer_class    | pretrained_weights
        MODELS = [(BertModel,       BertTokenizer,       'bert-base-uncased'),
                  (OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt'),
                  (GPT2Model,       GPT2Tokenizer,       'gpt2'),
                  (CTRLModel,       CTRLTokenizer,       'ctrl'),
                  (TransfoXLModel,  TransfoXLTokenizer,  'transfo-xl-wt103'),
                  (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),
                  (XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024'),
                  (DistilBertModel, DistilBertTokenizer, 'distilbert-base-cased'),
                  (RobertaModel,    RobertaTokenizer,    'roberta-base'),
                  (XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-base'),
                 ]

2. Each of this has its own pre-trained models and tokenizers:
  
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights)

3. Encode text:
  
        input_ids = torch.tensor([tokenizer.encode("Here is some text to encode", add_special_tokens=True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
        with torch.no_grad():
            last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples

4. To do fine-tuning select one adapter: (each architecture has its own fine-tuning classes)
  
        BERT_MODEL_CLASSES = [BertModel, BertForPreTraining, BertForMaskedLM, BertForNextSentencePrediction,
                          BertForSequenceClassification, BertForTokenClassification, BertForQuestionAnswering]

5. Load pre-trained model (for fine-tuning) with hidden-states and attention:
  
        model = model_class.from_pretrained(pretrained_weights,
                                            output_hidden_states=True,
                                            output_attentions=True)

6. Encode texts:
  
        input_ids = torch.tensor([tokenizer.encode("Let's see all hidden-states and attentions on this text")])
            all_hidden_states, all_attentions = model(input_ids)[-2:]

