import streamlit as st
import pandas as pd
import nltk
import numpy as np
import string
import warnings
import requests
import pickle
import random

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import json
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from responses import *
from data import *

# Lemmitization

lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def Normalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

vectorizer = TfidfVectorizer(tokenizer=Normalize,stop_words = stopwords.words('french'))



def load_doc(jsonFile):
    with open(jsonFile) as file:
        Json_data = json.loads(file.read())
    return Json_data


#data = load_doc('data.json')
book = load_doc('book.json')
eclf= joblib.load('eclf.pkl')
df = pd.DataFrame(data, columns = ["Text","Intent"])
x = df['Text']
y= df['Intent']
X= vectorizer.fit_transform(x)
eclf.fit(X, y)


# To get responnse

def response(user_response):
    text_test = [user_response]
    X_test = vectorizer.transform(text_test)
    prediction = eclf.predict(X_test)
    reply = random.choice(responses[prediction[0]]['response'])
    return reply

# To get indent
def intent(user_response):
    text_intent = [user_response]
    X_test_intent = vectorizer.transform(text_intent)
    predicted_intent = eclf.predict(X_test_intent)
    intent_predicted = responses[predicted_intent[0]]['intent']
    return intent_predicted

def bot_initialize(user_msg):
    flag=True
    while(flag==True):
        user_response = user_msg
        user_intent = intent(user_response)
        
        if (user_intent !=''):
            if (user_response == '/start'):
                resp = """Salut je  suis HSEbot une intelligence artificielle qui t'aide à identifier les dangers et les risques ainsi qu'à les prévenirs.Mon créateur est Dahou Renaud L:https://www.linkedin.com/in/dahou-renaud-louis-8958599a/\n\nComment puis-je t'aider ?\n\nTapez Bye pour quitter."""
                return resp
            
            elif (user_intent == 'salutation'):
                resp = str(random.choice(responses[0]['response'])) + ", comment puis-je vous aider?"
                return resp
        
            elif (user_intent == 'conaissance'):
                resp = str(random.choice(responses[1]['response']))+ ", comment puis-je vous aider?"
                return resp
            
            elif (user_intent == 'fin_conversation'):
                resp = random.choice(responses[2]['response'])
                return resp

            elif (user_intent == 'Merci'):
                resp = random.choice(responses[3]['response'])
                return resp

            elif (user_intent == 'but'):
                resp = random.choice(responses[5]['response'])
                return resp

            elif (user_intent == 'conaissance'):
                resp = random.choice(responses[1]['response'])
                return resp
            
            elif (user_intent == "question"):
                user_response=user_response.lower()
                resp =  response(user_response)
                return resp #+ "\n\n🎁CADEAU SURPRISE.🎁\nJe t'offre ce document HSE qui te servira un jour.😊:\n"+random.choice(book)

            else:
                resp = "Désolé je ne comprend pas mon vocabulaire est en amélioration.Envoie ta question à mon créateur @Renaud17" #random.choice(responses[4]['response'])
                return resp
                   
        else:
            flag = False
            resp = "Mais vous ne m'avez posé aucune question"+ ", comment puis-je vous aider?" #random.choice(responses[2]['response'])
            return resp
        
        
        
import logging
from typing import NoReturn
from time import sleep

import telegram
from telegram.error import NetworkError, Unauthorized

def make_reply(msg):     # user input will go here
    if msg is not None:
        reply = bot_initialize(msg)     # user input will start processing to bot_initialize function
        return reply

UPDATE_ID = None


def main() -> NoReturn:
    """Run the bot."""
    global UPDATE_ID
    # Telegram Bot Authorization Token
    bot = telegram.Bot('1836903308:AAHtERNcpC-aJjb6J86k2AUzzUu_rxlT53k')

    # get the first pending update_id, this is so we can skip over it in case
    # we get an "Unauthorized" exception.
    try:
        UPDATE_ID = bot.get_updates()[0].update_id
    except IndexError:
        UPDATE_ID = None

    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    while True:
        try:
            echo(bot)
        except NetworkError:
            sleep(1)
        except Unauthorized:
            # The user has removed or blocked the bot.
            UPDATE_ID += 1


def echo(bot: telegram.Bot) -> None:
    """Echo the message the user sent."""
    global UPDATE_ID
    # Request updates after the last update_id
    for update in bot.get_updates(offset=UPDATE_ID, timeout=10):
        UPDATE_ID = update.update_id + 1
        try:
            message = item["message"]["text"]
            #print(message)
        except:
            message = None
        from_ = item["message"]["from"]["id"]
        #print(from_)
        reply = make_reply(message)
        bot.send_message(reply,from_)

if __name__ == '__main__':
    main()
