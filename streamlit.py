import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as func
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import RomanianStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import pandas as pd

# Descarcă resursele necesare
nltk.download('stopwords')
nltk.download('punkt')

# Definește clasa modelului
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(4 * (5000 // 2), 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = func.relu(self.conv1(x))
        x = self.pool(x)
        x = func.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 4 * (5000 // 2))
        x = func.relu(self.fc1(x))
        x = self.dropout(x)
        x = func.relu(self.fc2(x))
        x = self.dropout(x)
        x = func.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        x = func.sigmoid(x)
        return x

# Încarcă modelul și vectorizatorul
model = NN()
model.load_state_dict(torch.load("text_classification_model.pth"))
model.eval()

vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Funcția de preprocesare a textului
def preprocess_text(text):
    stopwords_RO = stopwords.words('romanian')
    stemmer = RomanianStemmer()
    
    if type(text) != str:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    stop_words = set(stopwords_RO)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

st.title("Satire Detection")

input_title = st.text_input("Title")
input_content = st.text_area("Content")

if st.button("Classify"):
    processed_title = preprocess_text(input_title)
    processed_content = preprocess_text(input_content)
    input_text = processed_title + " " + processed_content

    X_input = vectorizer.transform([input_text])
    X_input = torch.tensor(X_input.toarray(), dtype=torch.float32)

    with torch.no_grad():
        prediction = model(X_input)
        prediction = torch.round(prediction).item()

    if int(prediction) == 0:
        st.write("The text is classified as class 0.")
    else:
        st.write("The text is classified as class 1.")
