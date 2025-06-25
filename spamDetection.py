import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re

nltk.download('stopwords')
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub(r'\W', ' ', text)  # Remove special chars
    text = text.lower()
    words = text.split()
    filtered = [ps.stem(w) for w in words if w not in stop_words]
    return ' '.join(filtered)

data = pd.read_csv("C:/Users/NOUNOU/Documents/spamDetection/spam.csv")

data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham','spam'],['Not Spam','Spam'])


data['Cleaned'] = data['Message'].apply(preprocess)

mess = data['Cleaned']
cat = data['Category']

(mess_train, mess_test, cat_train, cat_test) = train_test_split(mess, cat, test_size=0.2)

cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)
       
# Creating model
model = MultinomialNB()
model.fit(features, cat_train)  

# Test model
features_test = cv.transform(mess_test)
print(model.score(features_test, cat_test))

# Predict data
def predict(message):
    processed = preprocess(message)
    input_message = cv.transform([processed]).toarray()
    result = model.predict(input_message)
    return result

# Streamlit interface
st.header('Spam Detection App')

input_mess = st.text_input('Enter message here:')

if st.button('Validate'):
    prediction = predict(input_mess)
    st.markdown(prediction[0])
