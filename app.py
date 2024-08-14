import streamlit as st
import pickle
import nltk

nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

tfidf= pickle.load(open("vectorizer.pkl", "rb"))
model= pickle.load(open("model.pkl", "rb"))

st.title("Email/SMS Spam Classifier")

input_sms=st.text_area("Enter the message")

def text_processor(text):
    stemmer = PorterStemmer()
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    filtered_tokens = []
    for word in tokens:
        if word.isalnum() and word not in stopwords.words("english"):
            stemmed_word = stemmer.stem(word)
            filtered_tokens.append(stemmed_word)
    
    return " ".join(filtered_tokens)

if st.button("Predict"):
    
    # 1.preprocess
    transformed_sms=text_processor(input_sms)

    # 2.vectorize
    vector_input=tfidf.transform([transformed_sms])

    # 3.predict
    result=model.predict(vector_input)[0]

    # 4.display
    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")


