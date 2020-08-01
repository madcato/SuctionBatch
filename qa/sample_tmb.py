import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# executing these commands for the first time initiates a download of the
# model weights to ~/.cache/torch/transformers/
# Choose one of this spanish-pytorch-question-answering finetunned model
finetunned = "mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
# finetunned = "mrm8488/bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
# finetunned = "mrm8488/electricidad-small-finetuned-squadv1-es"

tokenizer = AutoTokenizer.from_pretrained(finetunned)
model = AutoModelForQuestionAnswering.from_pretrained(finetunned)


question = "Cuál es el detalle"

context = """

Saludos

Copied from TMB APP Android - Feature #62276: imetro - detalle de una estación de metro y ocupación en tiempo real	Resolved	26/06/2020	23/07/2020	
History

#1 Updated by Miguel Angel Hermida Perez 25 days ago

Copied from Feature #62276: imetro - detalle de una estación de metro y ocupación en tiempo real added
#2 Updated by David Moreno Ferrera 25 days ago

A la hora de calcular el tiempo que falta y mostrarlo en la vista de detalle de estación de metro, hay que restarle al tiempo aproximado (campo "temps_arribada" del json) el tiempo actual. Si sale positivo, hay que mostrar los minutos y segundos que faltan para que llegue el metro a la estación. Si el resultado es 0 o negativo, se mostrará el literal "Inminente".

Listo y subido con la versión 9.2.0,
Saludos
"""


# 1. TOKENIZE THE INPUT
# note: if you don't include return_tensors='pt' you'll get a list of lists which is easier for 
# exploration but you cannot feed that into a model. 
inputs = tokenizer.encode_plus(question, context, return_tensors="pt") 

# 2. OBTAIN MODEL SCORES
# the AutoModelForQuestionAnswering class includes a span predictor on top of the model. 
# the model returns answer start and end scores for each word in the text
answer_start_scores, answer_end_scores = model(**inputs)
answer_start = torch.argmax(answer_start_scores)  # get the most likely beginning of answer with the argmax of the score
answer_end = torch.argmax(answer_end_scores) + 1  # get the most likely end of answer with the argmax of the score

# 3. GET THE ANSWER SPAN
# once we have the most likely start and end tokens, we grab all the tokens between them
# and convert tokens back to words!
result = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
print(result)