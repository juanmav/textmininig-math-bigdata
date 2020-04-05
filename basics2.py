import pandas as pd
import numpy as np
import nltk
import os
import nltk
import nltk.corpus

from nltk.tokenize import word_tokenize

tokens = word_tokenize(mytext)

from nltk.probability import FreqDist

FreqDist(tokens).most_common(10)

nltk.download("wordnet")

from nltk.stem import WordNetLemmatizer

lemma = WordNetLemmatizer()

lemma.lemmatize("running")

from nltk.stem import PorterStemmer
# LancasterStemmer


stem = PorterStemmer()
stem.stem("running")

from nltk.corpus import stopwords

words = set(stopwords.words("english"))

clean = [x for x in listofwords if x not in words]


nltk.pos_tag([tokens])

for token in tokens:
	nltk.pos_tag([token])
