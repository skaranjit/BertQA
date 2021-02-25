import transformers
from transformers import ElectraTokenizer, ElectraForQuestionAnswering
import torch


class QAPipe():
    def __init__(self,context):
        #self.nlp = pipeline("question-answering")
        self.context = context
        self.input_ids = None
        self.text_tokens = None
        self.outputs = None
        self.answer_start_scores = None
        self.answer_end_scores = None
        self.answer_start = None
        self.answer_end = None
        self.tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-discriminator")
        self.model = ElectraForQuestionAnswering.from_pretrained("google/electra-small-discriminator")
    def __init_tokenizer__(self):
        self.tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-discriminator")
    
    def __init_model__(self):
        self.model = ElectraForQuestionAnswering.from_pretrained("google/electra-small-discriminator")
     
    def get_ids(self,inputs):
        return ((inputs["input_ids"].tolist()[0]))
        
        
    def generate_text_from_token(self):
        return self.tokenizer.convert_ids_to_tokens(self.input_ids)

    def get_output(self,question):
        self.input_ids = self.tokenizer.encode(question,self.context,max_length=2015,truncation=True)
        tokens = self.generate_text_from_token()
        import IPython; IPython.embed(); exit(1);
        # inputs = self.tokenizer(question,context,add_special_tokens=True,return_tensors="pt")
        # self.input_ids = self.get_ids(inputs)
        # #text_tokens = generate_text_from_token(input_ids)
        # outputs = self.model(**inputs)
        # self.answer_start_scores = outputs.start_logits
        # self.answer_end_scores = outputs.end_logits
        # self.answer_start = torch.argmax(
        #     self.answer_start_scores
        # )  # Get the most likely beginning of answer with the argmax of the score
        # self.answer_end = torch.argmax(self.answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
        # answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(self.input_ids[self.answer_start:self.answer_end]))
        # return answer
        result = self.nlp(question=question,context=self.context)
        return result


