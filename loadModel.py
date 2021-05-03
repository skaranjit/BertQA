import transformers
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
import numpy as np

class QAPipe():
    def __init__(self,context):
        #self.nlp = pipeline("question-answering")
        self.context = context
        self.input_ids = None
        self.text_tokens = None
        self.outputs = None
        self.segment_ids = None
        self.tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        self.model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    def __init_tokenizer__(self):
        self.tokenizer = BertTokenizer.from_pretrained("deepset/bert-large-uncased-whole-word-masking-squad2")
    
    def __init_model__(self):
        self.model = BertForQuestionAnswering.from_pretrained("deepset/bert-large-uncased-whole-word-masking-squad2")
     
    def get_ids(self):
        return self.input_ids
        
        
    def generate_text_from_token(self):
        return self.tokenizer.convert_ids_to_tokens(self.input_ids)

    def get_output(self,question):
        self.input_ids = self.tokenizer.encode(question,self.context,max_length=2015,truncation=True)
        tokens = self.generate_text_from_token()
        SEP_index = self.input_ids.index(102) #Getting index of first SEP token
        len_question = SEP_index+1
        len_answer = len(self.input_ids)-len_question
        segment_ids = [0]*len_question + [1]*len_answer #Segment Ids will 0 for question and 1 for answer
        self.segment_ids = segment_ids
        #Getting start and end scores for answer and converting input arrays to tensor before passing to the module
        start_token_scores = self.model(torch.tensor([self.input_ids]),token_type_ids=torch.tensor([segment_ids]))[0]
        end_token_scores = self.model(torch.tensor([self.input_ids]),token_type_ids=torch.tensor([segment_ids]))[1]
        #Converting scores tensor to numpy arrays so that we can use numpy functions
        start_token_scores =start_token_scores.detach().numpy().flatten()
        end_token_scores = end_token_scores.detach().numpy().flatten()
        #picking start and end index of an answer based on start/end indices with highest scores
        answer_start_index = np.argmax(start_token_scores)
        answer_end_index = np.argmax(end_token_scores)
         #Getting Scores for start and end token of the answer. 
        start_token_score = np.round(start_token_scores[answer_start_index],2)
        end_token_score = np.round(end_token_scores[answer_end_index],2)
        #Cleaning answer text
        
        answer = tokens[answer_start_index]
        for i in range(answer_start_index+1,answer_end_index+1):
            if tokens[i][0:2] == '##':
                answer+=tokens[i][2:]
            else:
                answer+= ' '+tokens[i]
        if(answer_start_index ==0) or (start_token_score<0) or (answer == '[SEP]') or (answer_end_index < answer_start_index):
            answer = "Sorry!, Not able to answer your question. Try a different question related to the context."
        
        return (answer_start_index,answer_end_index,start_token_score,end_token_score,start_token_scores,end_token_scores,answer)
