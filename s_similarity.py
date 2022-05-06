import torch
from transformers import BertTokenizer, BertModel
from transformers import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
#XL Net
from transformers import TransfoXLTokenizer, TransfoXLModel
from word_vectors import read
import spacy
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.ERROR)
import io
import os
import pickle
import yaml

"""
From here:
https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
here:
https://fasttext.cc/docs/en/english-vectors.html
here:
https://github.com/huggingface/transformers
"""
def load_intent(path,enforce_version_3=True):            
    p = os.path.join(path)
    data = yaml.safe_load(open(p))
    if enforce_version_3 and not data.get("version") == "3.0":
        raise Exception(f"File not version 3.0: {p!r}")
    intents = {}
    for nlu in data.get("nlu", []):
        name = nlu["intent"]
        if name in intents:
            raise Exception(f"Duplicate intent {name}")
        intents[name] = []
        examples = nlu["examples"]
        if isinstance(examples, str):
            examples = yaml.safe_load(examples)
            intents[name].append(examples)
    return intents
            

def replace_all(text,tokens):
    for t in tokens:
        text = text.replace(t,"")
    return text    
root_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),"LanguageModels")
from tqdm import tqdm
class SentenceSimilarity:
    def __init__(self,MODEL="fasttext"):
        self.MODEL = MODEL.lower()
        if(MODEL=="gpt2"):
            # Load pre-trained model tokenizer (vocabulary)
            self.tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
            # Load pre-trained model (weights)
            self.model = GPT2Model.from_pretrained('distilgpt2')
            self.model.eval()
        elif(MODEL=="glove"):
            print("Glove Will take a while to set up.")
            exit()
            self.glove_root_dir = os.path.join(root_dir,"glove_text_vectors")
            WRITE = True
            if(WRITE):
                print("Loading Glove Model from txt file. This will take a while...")
                os.mkdir(self.glove_root_dir)
                self.word_map, word_vectors = read(os.path.join(root_dir,"glove.840B.300d.txt"))
                with tqdm(total=len(word_vectors)) as progress_bar:
                    for idx,vec in enumerate(word_vectors):                        
                        with open(os.path.join(self.glove_root_dir,str(idx)),"w") as file:
                            file.write(str(vec))
                        progress_bar.update(1)
                pickle.dump(self.word_map,open(os.path.join(root_dir,"glove_word_map.pkl"),"wb"))
            else:
                print("loading saved Glove vectors from hard storage")
                self.word_map = pickle.load(open("glove_word_map.pkl","rb"))
            print("Done.")
        elif(MODEL=="word2vec"):                      
            self.model = spacy.load("en_core_web_lg")  # make sure to use larger package!
        elif(MODEL=="gpt"):
            # Load pre-trained model tokenizer (vocabulary)
            self.tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
            # Load pre-trained model (weights)
            self.model = OpenAIGPTModel.from_pretrained('openai-gpt')
            self.model.eval()            
        elif(MODEL=="fasttext"):
            self.fasttext_root_dir = os.path.join(root_dir,"fast_text_vectors")
            if(os.path.isdir(self.fasttext_root_dir)):
                print("loading fasttext word vectors from hard storage")
                self.word_map = pickle.load(open("fast_text_word_map.pkl","rb"))
                print("done.")
            else:
                os.mkdir(self.fasttext_root_dir)
                print("loading and pre-writing fasttext word vectors to storage. This will take a while...")
                fin = io.open(os.path.join(root_dir,"fasttext.vec"), 'r', encoding='utf-8', newline='\n', errors='ignore')
                n, d = map(int, fin.readline().split())
                self.word_map = {}
                self.fasttext_vecs = []
                with tqdm(total=n) as progress_bar:
                    for idx,line in enumerate(fin):
                        tokens = line.rstrip().split(' ')
                        self.word_map[tokens[0]] = idx
                        with open(os.path.join(self.fasttext_root_dir,str(idx)),"w") as file:
                            file.write(str(list(map(float, tokens[1:]))))
                        progress_bar.update(1)
                pickle.dump(self.word_map,open(os.path.join(root_dir,"fast_text_word_map.pkl"),"wb"))
                print("loaded")
        elif(MODEL=="bert"):
            print("Using Bert")
            # Load pre-trained model tokenizer (vocabulary)
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')        
            self.model = BertModel.from_pretrained('bert-base-uncased',
                                            output_hidden_states = True, # Whether the model returns all hidden-states.
                                            )
            # Put the model in "evaluation" mode, meaning feed-forward operation.
            self.model.eval()
        else:
            raise AttributeError("Error::Invalid model {}, please choose from {}",MODEL,["fasttext","BERT","spacy"])
    def cosine_similarity(self,a,b):
        return torch.dot(a,b) / (torch.norm(a) *torch.norm(b))
    def __call__(self,a,b):        
        a_embedding = self.forward(a.lower())
        b_embedding = self.forward(b.lower())
        return self.cosine_similarity(a_embedding,b_embedding)
    def aggregate_hidden_states(self,hidden_states):
        vec_embedding = torch.zeros(len(hidden_states),hidden_states[-1][0].shape[1])
        div = 1
        for idx,layer in enumerate(hidden_states):
            scaling_factor = (idx+1)*(idx+1)/2
            div *= scaling_factor
            vec_embedding[idx] = scaling_factor*torch.mean(layer[0], dim=0)
        # Calculate the average of all 22 token vectors.
        return torch.mean(vec_embedding,dim=0)/div
    def forward(self,sentence):
        if(self.MODEL=="gpt"):            
            # Tokenized input
            tokenized_text = self.tokenizer.tokenize(sentence)
            # Convert token to vocabulary indices
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor([indexed_tokens])
            # Predict hidden states features for each layer
            with torch.no_grad():
                outputs = self.model(tokens_tensor)
                hidden_states = outputs[0]
            return self.aggregate_hidden_states(hidden_states.unsqueeze(0))
        elif(self.MODEL=="bert"):
            # Add the special tokens.
            marked_text = "[CLS] " + sentence + " [SEP]"
            # Split the sentence into tokens.
            tokenized_text = self.tokenizer.tokenize(marked_text)
            # Map the token strings to their vocabulary indeces.
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            segments_ids = [1] * len(tokenized_text)
            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])
            # Run the text through BERT, and collect all of the hidden states produced
            # from all 12 layers. 
            with torch.no_grad():
                outputs = self.model(tokens_tensor, segments_tensors)
                hidden_states = outputs[2]
            return self.aggregate_hidden_states(hidden_states)
        elif(self.MODEL=="fasttext"):            
            embedding = torch.zeros((300))
            num_words = 0
            for word in replace_all(sentence,["!","?",".",",",";","'","\""]).split(" "):
                if(word in self.word_map.keys()):
                    embedding_id = self.word_map[word]
                    embedding += torch.tensor(eval(open(os.path.join(self.fasttext_root_dir,str(embedding_id))).read()))
                    num_words +=1
            return embedding/num_words
        elif(self.MODEL=="word2vec"):
            return torch.from_numpy(self.model(sentence).vector)
        elif(self.MODEL=="glove"):
            embedding = torch.zeros((300))
            num_words = 0        
            for word in replace_all(sentence,["!","?",".",",",";","'","\""]).split(" "):
                if(word in self.word_map.keys()):
                    embedding_id = self.word_map[word]
                    embedding += torch.tensor(eval(open(os.path.join(self.glove_root_dir,str(embedding_id))).read()))
                    num_words +=1
            return embedding/num_words
        elif(self.MODEL=="gpt2"):
            # Predict hidden states features for each layer
            # Encode some inputs
            indexed_tokens = self.tokenizer.encode(sentence)
            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor([indexed_tokens])
            with torch.no_grad():
                outputs = self.model(tokens_tensor)
                hidden_states = outputs[0]
            return self.aggregate_hidden_states(hidden_states.unsqueeze(0))


if(__name__=="__main__"):
    from sys import argv
    if(len(argv)<2):
        print("usage: python3 test.py --model_name")
        exit()
    print("Loading Models...")
    s_similarity = SentenceSimilarity(argv[1])
    print("Done.")
    intents = load_intent(os.path.join(root_dir,"general_nlu.yml"))
    def score_intent(intent_name, check_val):
        return max((s_similarity(e,check_val) for e in intents[intent_name][0]), default=0)        

    while True:
        print(">",end="")
        inputx = input()
        best_intent = ""
        best_intent_score = 0
        for intent_name in intents.keys():
            intent_score = score_intent(intent_name,inputx)
            if(intent_score>=best_intent_score):
                best_intent_score = intent_score
                best_intent = intent_name
        print("Best Matched Intent: ",best_intent," Score : ",best_intent_score)
                    
    """
    print(s_similarity("Hi there, how are you?","Howdy!"))
    print(s_similarity("Hi there, how are you?","Hi."))
    print(s_similarity("Hi there, how are you?","Hello."))
    print(s_similarity("Hi there, how are you?","Good morning"))
    print(s_similarity("Hi there, how are you?","Greetings."))
    print(s_similarity("Hi there, how are you?","I like the movie soylent greens."))
    print(s_similarity("Hi there, how are you?","I like javascript."))
    print(s_similarity("Hi there, how are you?","Have you read the book 1984? I haven't."))
    """
