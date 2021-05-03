There are many machine learning algorithms that tries to solve the question answering using the natural language. In old day, the bag of words was popular amongst others which tries to answer the question that are pre-defined by the developers. Using this method, developers have to spend a lot of time writing the questions and answer for the particular questions. This method was very useful for the chat bots but was not able to answer the questions for the huge database. Modern Natural Language Processor are dominated by the models called transformers. Using these transformers library, the Bert QA model was introduced by HuggingFace which reads through the text context provided by the user and tries to answer the questions related to that text context. This model has been promising in answering the complex question from a large document. For example, if one company have a report regarding their financial years that been fed through the model, user can just ask question regarding the certain year or the profits they made for particular year. And without scrolling through the documents the answer can be found with the matter of seconds.




Setting up the environment
# pip install -r requirement.txt
This will install all the required packages for the project. There are 2 main python modules that I have written that are in the heart of this project.
1. loadModel.py : This module will load the BERT model process inputs and generates outputs. This module has a class QAPipe which we will use to interact with the BERT model.
2. create_dataset.py: This module will create a dataset from either text or article from web address. It has a class named Create_DS which is used to generate the clean context from either a text passage or article from the internet.
Also, I have included the jupyter notebook that anyone can use to better understand the model and how it works.
