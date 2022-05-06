# Testing different embeddings on measuring sentence similarity

## Usage
`python3 s_similarity.py --model_name`

Or

```
from s_similarity import SentenceSimilarity

ss = SentenceSimilarity(model="gpt") #available models = ["gpt","gpt2","bert","word2vec","fasttext","glove"]
print(s_similarity("Hi there, how are you?","Howdy!"))
```

## Requisites

### The transformer models will download automatically on first run, but you'd need to manually download and place the data in the LanguageModels folder for now.
fast_text_data_download = https://fasttext.cc/docs/en/english-vectors.html

glove_data_download = https://nlp.stanford.edu/projects/glove/


## Some simple tests

GPT : 

```
=========
Hi there, how are you?    |vs|    Howdy!
tensor(0.6723)
=========
Hi there, how are you?    |vs|    Hi.
tensor(0.6685)
=========
Hi there, how are you?    |vs|    Hello.
tensor(0.6738)
=========
Hi there, how are you?    |vs|    Good morning
tensor(0.5629)
=========
Hi there, how are you?    |vs|    Greetings.
tensor(0.5543)
=========
Hi there, how are you?    |vs|    I like the movie soylent greens.
tensor(0.4912)
=========
Hi there, how are you?    |vs|    I like javascript.
tensor(0.4024)
=========
Hi there, how are you?    |vs|    Have you read the book 1984? I haven't.
tensor(0.4541)
```

BERT: 

```
=========
Hi there, how are you?    |vs|    Howdy!
tensor(0.6272)
=========
Hi there, how are you?    |vs|    Hi.
tensor(0.6614)
=========
Hi there, how are you?    |vs|    Hello.
tensor(0.6990)
=========
Hi there, how are you?    |vs|    Good morning
tensor(0.7214)
=========
Hi there, how are you?    |vs|    Greetings.
tensor(0.6741)
=========
Hi there, how are you?    |vs|    I like the movie soylent greens.
tensor(0.5625)
=========
Hi there, how are you?    |vs|    I like javascript.
tensor(0.5950)
=========
Hi there, how are you?    |vs|    Have you read the book 1984? I haven't.
tensor(0.6398)
```

Word2Vec:  (Spacy default)

```
=========
Hi there, how are you?    |vs|    Howdy!
tensor(0.4630)
=========
Hi there, how are you?    |vs|    Hi.
tensor(0.6965)
=========
Hi there, how are you?    |vs|    Hello.
tensor(0.6991)
=========
Hi there, how are you?    |vs|    Good morning
tensor(0.6722)
=========
Hi there, how are you?    |vs|    Greetings.
tensor(0.5281)
=========
Hi there, how are you?    |vs|    I like the movie soylent greens.
tensor(0.7077)
=========
Hi there, how are you?    |vs|    I like javascript.
tensor(0.7433)
=========
Hi there, how are you?    |vs|    Have you read the book 1984? I haven't.
tensor(0.8725)
```



