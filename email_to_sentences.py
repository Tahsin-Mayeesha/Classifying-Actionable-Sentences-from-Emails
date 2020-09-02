import numpy as np # linear algebra
import pandas as pd
import email # for processing emails

emails = pd.read_csv("data/enron-email-dataset/emails.csv")

def get_content_from_email(message):
    """ Extracts content text from emails """
    message = email.message_from_string(message) # creates a email message object from text
    for part in message.walk():
        text = part.get_payload() # extracts the plain text content portion
    return text

def test_filter_words(text):
    """ Checks if the sentences contain any filter words or not"""
    for filter_word in filter_words:
        if filter_word in text:
            return False
    else:
        return True


def preprocess_email(email):
    """ Preprocessing function to tokenize email text to sentences """
    result = []
    # Split the paragraphs first
    email = email.split("\n\n")
    paragraph_sentences = []
    for item in email:
        paragraph_sentences.extend(item.split("."))
    for text in paragraph_sentences:
        text = text.strip()
        if text != '' and test_filter_words(text) and len(text) >= text_len :
            result.append(text)
    return result

filter_words = ["Forwarded","Subject","<",">","From","To","cc","--","Report as of"] # these sentences don't have action words
text_len = 5 # to remove small sentences like 'gtg','lol'

email_texts = []

for message in emails['message']:
    # First we extract the text from the raw emails
    email_texts.append(get_content_from_email(message))

email_sentences = []
for email in email_texts:
    email_sentences.extend(preprocess_email(email))

pd.Series(email_sentences).to_csv("email_sentences.csv")

