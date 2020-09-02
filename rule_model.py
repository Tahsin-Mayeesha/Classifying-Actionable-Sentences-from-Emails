import pandas as pd
import spacy
import textacy
# this thing should take an argument a file and produce a pandas dataframe with targets
keywords = ["find email","send email","respond","check","plan","create reminder",
            "find calendar entry","search","add agenda item","create calendar entry","open agenda",
            "send email","find email","make call","open setting","assign","deliver",
            "suggest","order","request","create assignment","forward","tag","todo","call me",
            "call us","give us a call","email"]

def data_cleanup(text):
    return text.replace("\n","").replace("\t","")


#email_sentences = pd.read_csv("email_sentences.csv")
#email_sentences.columns = ['unnamed','text']
#email_sentences = email_sentences['text']
#email_sentences = email_sentences.map(data_cleanup)

# Create spacy model instance
nlp = spacy.load("en_core_web_sm")

email_sentences = ["suggested materials should be sent","hello world"]

def rule_based_model(text):
    # First check for action keywords
    text = data_cleanup(text)
    for keyword in keywords:
        if keyword in text:
            return True
    # If the sentence is a question and long then it might be an action sentence
    if "?" in text and len(text.split()) >= 5:
        return True
    # create spacy object
    doc = nlp(text)
    # get subject verb object triplets, if its non-empty then the sentence is actionable
    sub_vrb_obj = list(textacy.extract.subject_verb_object_triples(doc))
    if sub_vrb_obj != []:
        return True
    else:
        return False


#labels = []

#for email in email_sentences:
#    labels.append(rule_based_model(email))
#print(labels)
#email_labelled = pd.DataFrame({"emails":email_sentences,"target":labels})
#pd.to_csv("email_labelled.to_csv",index=False)