import streamlit as st
from PIL import Image
import nltk
import re
import string
import pickle
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize.toktok import ToktokTokenizer
tokenizer = ToktokTokenizer()
import nltk
nltk.download('stopwords')
stopwords_list = nltk.corpus.stopwords.words('english')
stopwords_list.remove('no')
stopwords_list.remove('not')
model = pickle.load(open('SentimentAnalysis.p','rb'))


nav = st.sidebar.radio("Navigation",["Home", "Analyise","Contact"])

if nav == "Home":
    st.title("Sentiment Analysis")
    image = Image.open("image.jpeg")
    st.image(image,width = 400)
    st.write("Opinion information is very important for businesses and manufacturers. They often want to know in time what consumers and the public  think of their products and services. However, it is not realistic to manually read every post on the website and extract useful viewpoint information from it. If you do it manually, there is too much data. Sentiment analysis allows large-scale processing of data in an efficient and cost-effective manner. This project used a dataset of the Amazon reviews and then built a model to predict the sentiment of the comment given the comment declaration by using Python and machine learning algorithm- Naïve Bayes.")
if nav == "Analyise":
    st.title("Sentiment Analysis")
    st.subheader("Enter Text to analyise: ")
    text = st.text_input(" ")
    st.write("Note: For accurate Prediction, please enter minimum of 10-20 words as model is trained with such reviews.Please click Enter after typing the text and then proceed to click Predict button")

    st.write('Text after proceesing: ',text)
    text = [text]
    y_out = model.predict(text)

    if st.button("Predict"):

        if (y_out == "Positive"):
            image = Image.open("happy.jpeg")
            st.image(image,width = 250)
            st.header("WOW!! That's Positive review")
        elif(y_out == "Negative"):
            image = Image.open("sad.jpeg")
            st.image(image,width = 250)
            st.header("That's Negative review")
if nav == "Contact":
    st.title("Name: D Deepak Prasanna")
    st.header("Email: prasanna333.d@gmail.com")
