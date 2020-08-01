import torch
from transformers import *

# Transformers has a unified API
# for 1 transformer architectures and 30 pretrained weights.
#          Model          | Tokenizer          | Pretrained weights shortcut
MODEL = (DistilBertModel, DistilBertTokenizer, 'distilbert-base-cased')

# To use TensorFlow 2.0 versions of the models, simply prefix the class names with 'TF', e.g. `TFRobertaModel` is the TF 2.0 counterpart of the PyTorch model `RobertaModel`

# Let's encode some text in a sequence of hidden-states using each model:
(model_class, tokenizer_class, pretrained_weights) = MODEL
# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)
print(tokenizer)
print(model)
exit() 
# Encode text
input_ids = torch.tensor([tokenizer.encode("Here is some text to encode", add_special_tokens=True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
with torch.no_grad():
    last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples

# Each architecture is provided with several class for fine-tuning on down-stream tasks, e.g.
BERT_MODEL_CLASS = BertForQuestionAnswering

# All the classes for an architecture can be initiated from pretrained weights for this architecture
# Note that additional weights added for fine-tuning are only initialized
# and need to be trained on the down-stream task
fine_model_class = BERT_MODEL_CLASS
# Load pretrained model/tokenizer
model = fine_model_class.from_pretrained(pretrained_weights)

# Models can return full list of hidden-states & attentions weights at each layer
model = model_class.from_pretrained(pretrained_weights,
                                    output_hidden_states=True,
                                    output_attentions=True)
input_ids = torch.tensor([tokenizer.encode("Let's see all hidden-states and attentions on this text")])
all_hidden_states, all_attentions = model(input_ids)[-2:]

# Models are compatible with Torchscript
model = model_class.from_pretrained(pretrained_weights, torchscript=True)
traced_model = torch.jit.trace(model, (input_ids,))

# Simple serialization for models and tokenizers
model.save_pretrained('./directory/to/save/')  # save
model = model_class.from_pretrained('./directory/to/save/')  # re-load
tokenizer.save_pretrained('./directory/to/save/')  # save
tokenizer = BertTokenizer.from_pretrained('./directory/to/save/')  # re-load

# SOTA examples for GLUE, SQUAD, text generation...