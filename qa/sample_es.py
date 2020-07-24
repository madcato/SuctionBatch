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


question = "Como está construido"
# question = "En qué época existió"

context = """
Parte del túmulo alargado Coldrum
El túmulo alargado Coldrum (en inglés Coldrum Long Barrow), también conocido como piedras Coldrum (Coldrum Stones) o piedras Adscombe (Adscombe Stones),
 es un túmulo alargado con cámara ubicado cerca del pueblo de Trottiscliffe del condado de Kent, en el sudeste de Inglaterra. Probablemente construido en el
  cuarto milenio antes de Cristo, durante el período Neolítico inicial de Gran Bretaña, se encuentra en estado de ruina.
Construido con tierra y alrededor de cincuenta megalitos de piedra sarsen local, consistía en un túmulo de tierra trapezoidal, casi rectangular, 
rodeado de bordillos. Dentro del extremo oriental del túmulo había una cámara de piedra en la que se depositaron restos humanos en, al menos, dos ocasiones 
separadas en el tiempo durante el Neolítico inicial. El análisis osteoarqueológico de estos restos ha demostrado que son de no menos de diecisiete individuos, 
una mezcla de hombres, mujeres y niños. Uno, sino más, de los cuerpos había sido desmembrado antes del entierro, lo que podría reflejar una tradición funeraria 
de excarnación e inhumación secundaria. Al igual que con otros túmulos, Coldrum ha sido interpretado como una tumba para albergar los restos de los muertos, 
tal vez como parte de un sistema de creencias que involucraba la veneración de los antepasados, aunque los arqueólogos han sugerido que también podría haber 
tenido más connotaciones religiosas, rituales y usos culturales."""


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