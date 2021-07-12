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

st.title("Sentiment Analysis")

nav = st.sidebar.radio("Navigation",["Home", "Prediction"])

if nav == "Home":
    image = Image.open(r"https://github.com/DDeepakPrasanna/SentimentAnalysis/blob/main/image.jpeg")
    st.image(image,width = 500)
    st.write("Opinion information is very important for businesses and manufacturers. They often want to know in time what consumers and the public  think of their products and services. However, it is not realistic to manually read every post on the website and extract useful viewpoint information from it. If you do it manually, there is too much data. Sentiment analysis allows large-scale processing of data in an efficient and cost-effective manner. This project used a dataset of the Amazon reviews and then built a model to predict the sentiment of the comment given the comment declaration by using Python and machine learning algorithm- Naïve Bayes.")
if nav == "Prediction":
    st.subheader("Enter Text: ")
    text = st.text_input(" ")
    st.write("Please click Enter after typing the text and then proceed to click Predict button")

    def remove_sp(text):
            text = text.lower()
            text = re.sub('\[.*?\]',"",text)
            text = re.sub('[%s]' %re.escape(string.punctuation), "", text)
            text = re.sub('\w*\d\w',"",text)
            text = re.sub('[''""_]', "", text)
            text = re.sub('\n',"", text)
            return text

        #remove stopwords
    def remove_stopwords(text):
      tokens = tokenizer.tokenize(text)
      tokens = [token.strip() for token in tokens]
      filtered_tokens = [token for token in tokens if token not in stopwords_list]
      filtered_text = ' '.join(filtered_tokens)
      return filtered_text

    text = text.lower()
    text = remove_sp(text)
    text = remove_stopwords(text)
    st.write('Text after proceesing: ',text)
    text = [text]
    y_out = model.predict(text)

    if st.button("Predict"):

        st.write(f'PREDICTED OUTPUT: {y_out}')

        if (y_out == "Positive"):
            st.write("😊")
        else:
            st.write("😞")

        CATEGORIES = ['Negative', 'Positive']
        q = model.predict_proba(text)
        for index, item in enumerate(CATEGORIES):
              st.write(f'{item} : {q[0][index]*100}%')
