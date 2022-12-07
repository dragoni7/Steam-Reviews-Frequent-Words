import string
import numpy as np
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk import FreqDist

# set the path to project location
path = str(__file__)
path = path.split("steam_review_datamining.py")
path = path[0].replace('\\', '/')

stop_words = set(stopwords.words('english')) # stop words
table = str.maketrans('', '', string.punctuation) # punctuation table

# cleans a series of sentences
def clean(series):
    # tokenize and clean data
    tokens = series['review_text'].apply(word_tokenize)
    # convert to lowercase
    tokens = [list(map(lambda word: word.lower(), sentence)) for sentence in tokens]
    # remove word punctuation
    strippedTokens = [list(map(lambda word: word.translate(table), sentence)) for sentence in tokens]
    # remove special characters
    words = [[word for word in sentence if word.isalpha()] for sentence in strippedTokens]
    # remove produced empty lists
    words = [sentence for sentence in words if sentence != []]
    # remove stop words
    return [[word for word in sentence if not word in stop_words] for sentence in words]

# normalizes a sentence by applying a lemmatizer based on tagged word
def normalize(sentence):
    result = []
    for word, tag in sentence:
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        result.append(WordNetLemmatizer().lemmatize(word, pos))
    return result

# fastest method for flattening a list. Source: https://chrisconlan.com/fastest-way-to-flatten-a-list-in-python/
def flatten_sentence(sentence_list):
    result = []
    for sentence in sentence_list:
        result += sentence
    return result

if __name__ == "__main__":
    # import the data:
    steam_reviews = pd.read_csv(path + 'reviews/dataset.csv', nrows=100000)
    #steam_reviews = steam_reviews[steam_reviews['review_votes'] > 0] # only consider reviews that were recommended
    steam_reviews = steam_reviews.astype({"review_text": str}) # convert review text to string for tokenization
    steam_reviews = steam_reviews[steam_reviews['review_score'].notnull()] # remove null scores
    steam_reviews.drop_duplicates(['review_text', 'review_score'], inplace=True) # remove duplicates

    # sort data by review score and save as .csv
    pos_reviews = steam_reviews[steam_reviews['review_score'] == 1] # score of 1 is positive.
    pos_reviews.to_csv(path + 'reviews/positive_reviews.csv')

    neg_reviews = steam_reviews[steam_reviews['review_score'] == -1] # score of -1 is negative.
    neg_reviews.to_csv(path + 'reviews/negative_reviews.csv')

    posWords = clean(pos_reviews)
    negWords = clean(neg_reviews)

    # use pos_tag to determine context of review words.
    taggedPosWords = [pos_tag(sentence) for sentence in posWords]
    taggedNegWords = [pos_tag(sentence) for sentence in negWords]
    # lemmatize sentences
    lemmatizedPosWords = [normalize(sentence) for sentence in taggedPosWords]
    lemmatizedNegWords = [normalize(sentence) for sentence in taggedNegWords]
    # flatten the sentences into one
    allPosWords = flatten_sentence(lemmatizedPosWords)
    allNegWords = flatten_sentence(lemmatizedNegWords)

    posWordFreq = FreqDist(allPosWords)
    negWordFreq = FreqDist(allNegWords)

    print(posWordFreq.most_common(20))
    print(negWordFreq.most_common(20))
