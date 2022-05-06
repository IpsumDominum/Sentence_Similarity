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


## Some tests
