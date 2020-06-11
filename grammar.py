import nltk

groucho_grammar = nltk.CFG.fromstring("""
S -> NP VP
PP -> P NP
NP -> Det N | Det N PP | 'I'
VP -> V NP | VP PP
Det -> 'an' | 'my'
N -> 'elephant' | 'pajamas'
V -> 'shot'
P -> 'in'
""")


sent = ['I', 'shot', 'an', 'elephant', 'in', 'my', 'pajamas']
parser = nltk.ChartParser(groucho_grammar)
for tree in parser.parse(sent):
    print(tree)


quizás con ntlk podría crear grmáticas libres de contexto que pudieran dar ordenes a un sistema.

La entrada al algoritmo Rick serían estructuras de arbol. Estos arboles contendrían los las palabras parseadas y sus tipos.
