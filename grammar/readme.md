# Grammar parsing

NLTK

- https://stackoverflow.com/questions/42322902/how-to-get-parse-tree-using-python-nltk
- https://stackoverflow.com/questions/6115677/english-grammar-for-parsing-in-nltk

## Try Stanford NLP Grammar

Install -> https://pypi.org/project/pycorenlp/

Download folder `stanford-corenlp-full` from https://stanfordnlp.github.io/CoreNLP/download.html

### Launch

    $ cd stanford-corenlp-4.0.0/
    $ export CLASSPATH="`find . -name '*.jar'`"
    $ java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer [port?]  # run server

REQUIRES JAVA