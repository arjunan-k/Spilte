import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# st.set_page_config(page_title="Spilter - Arjunan K", page_icon=None, layout="centered",
#                    initial_sidebar_state="auto", menu_items=None)
st.markdown("<h1 style='text-align: center; font-size: 100px; padding-bottom: 10px;'>SPILTER</h1>",
            unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; font-size: 20px; padding-top: 0px;'>Spam email "
            "and sms classifier made by Arjunan K.</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; font-size: 0px; padding-bottom: 10px;'></h1>", unsafe_allow_html=True)

# st.title("Spam Email and SMS Filter")
input_sms = st.text_area("Enter the message")

if st.button("Predict"):

    # 1. Preprocess
    def transform_text(text):
        ps = PorterStemmer()

        # Lower case
        text = text.lower()

        # Tokenization
        text = nltk.word_tokenize(text)
        y = []
        for i in text:
            if i.isalnum():
                y.append(i)

        # Remove Special Characters and stop words
        text = y.copy()
        y.clear()

        for i in text:
            if i not in stopwords.words("english") and i not in string.punctuation:
                y.append(i)

        # Stemming
        text = y.copy()
        y.clear()
        for i in text:
            y.append(ps.stem(i))

        return " ".join(y)

    transform_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_input = tfidf.transform([transform_sms])

    # 3. Predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
hide_decoration_bar_style = '''
    <style>
        header {visibility: hidden;}
    </style>
'''
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)
st.markdown(hide_streamlit_style, unsafe_allow_html=True)