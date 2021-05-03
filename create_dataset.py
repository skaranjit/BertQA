'''
This class will get context from the users and return as a raw data.
User can choose to upload context from textfile, pdf, or even internet article.
'''
#Importing necessary Libraries
import os
from newspaper import Article
import nlp


class Create_DS():
    def __init__(self):
        self.ds = None

    def loadPdf(self,filename):
        print("I am working in progress")
    def loadTxt(self,text):
        self.ds = self.clean_text(text)
    def loadArticle(self,filename):
        article = Article(filename)
        article.download()
        article.parse()
        article.nlp()
        print(article.text)
        self.ds = self.clean_text(article.text)

    def clean_text(self,text):
        text = text.replace("]", " ] ")
        text = text.replace("[", " [ ")
        text = text.replace("\n", " ")
        text = text.replace("''", '" ').replace("``", '" ')

        return text
