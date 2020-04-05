from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer

from nltk.stem import PorterStemmer
lemmatizer = PorterStemmer()

base_path = "./resources/review_polarity/txt_sentoken/"

def load_files(path, label):
    results = defaultdict(list)
    for file in Path(path).iterdir():
        with open(file, "r") as file_open:
            text = file_open.read()
            results["text"].append(text)
            results["label"] = label
    df = pd.DataFrame(results)
    return df


df_neg = load_files(base_path + 'neg/', 'negative')
df_pos = load_files(base_path + 'pos/', 'positive')

df_all = pd.concat([df_neg, df_pos])


stop_words = stopwords.words('english')
def process_text(text):
    tokens = word_tokenize(text)
    words = [word for word in tokens if word.isalpha()]
    words = [w for w in words if not w in stop_words]
    words = [lemmatizer.stem(w) for w in words]
    return " ".join(words)


sample = 10
df_all = pd.concat([df_all.head(sample), df_all.tail(sample)])
df_all['processed'] = df_all.apply(lambda x: process_text(x['text']), axis=1)

cv = CountVectorizer()

df_all['frecuencies'] = df_all.apply(lambda x: FreqDist(x['processed'].split()), axis=1)

from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer()
text_tf = tf.fit_transform(df_all['processed'])

df_all['vectorized'] = list(text_tf.toarray())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_all['vectorized'].tolist(), df_all['label'], test_size=0.3, random_state=3)

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted = clf.predict(X_test)
print(y_test)
print("MultinomialNB Accuracy:", metrics.accuracy_score(y_test, predicted))
print("MultinomialNB Confunsio Matrix:", metrics.confusion_matrix(y_test, predicted))
