import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

sentences = [
    "The receptive @ field is a the The the portion of sensory space that can elicit neuronal responses when stimulated. ",
    "Each ganglion cell ! or * optic nerve fiber bears a receptive field, increasing with intensifying light.",
    "When used in this sense, the term adopts a meaning reminiscent of receptive fields in actual biological nervous systems."
]


##for s in sentences:
##    local = s
##    local = re.sub('[^a-zA-Z0-9- ]', '', local)
##    local = local.split()
##    print(local)

tokens = word_tokenize(sentences[0])
frequencies = FreqDist(tokens)

#for key, value in frequencies.items():
#    print(key, '-> ', value)

tuples = list(frequencies.items())
print(tuples)
tuples.sort(key=lambda x: x[0])
print(tuples)

from nltk.stem import WordNetLemmatizer

lemma = WordNetLemmatizer()

print(lemma.lemmatize('apples'))
print(lemma.lemmatize('dies'))

from nltk.stem import PorterStemmer
# LancasterStemmer

stem = PorterStemmer()
print('running', stem.stem("running"))

from nltk.corpus import stopwords

stopwords = set(stopwords.words("english"))

print(stopwords)

result = list(filter(lambda e: e not in stopwords, tokens))
result2 = [x for x in tokens if x not in stopwords]

print("------------------------------")

print(result)
print(result2)

print(nltk.pos_tag(tokens))
