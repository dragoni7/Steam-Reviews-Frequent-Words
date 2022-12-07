import string
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# set the path to project location
path = str(__file__)
path = path.split("steam_review_datamining.py")
path = path[0].replace('\\', '/')

# import the data:
steam_reviews = pd.read_csv(path + 'reviews/dataset.csv', nrows=10000)
steam_reviews = steam_reviews.astype({"review_text": str}) # convert review text to string for tokenization

# sort data by review score and save as .csv
pos_reviews = steam_reviews[steam_reviews['review_score'] == 1] # score of 1 is positive.
pos_reviews.to_csv(path + 'reviews/positive_reviews.csv', index=False)

neg_reviews = steam_reviews[steam_reviews['review_score'] < 1] # score less than 1 is negative.
neg_reviews.to_csv(path + 'reviews/negative_reviews.csv', index=False)

# tokenize and clean data
posTokens = pos_reviews['review_text'].apply(word_tokenize)
negTokens = neg_reviews['review_text'].apply(word_tokenize)
posTokens = posTokens.explode().dropna()
negTokens = negTokens.explode().dropna()
# convert to lowercase
posTokens = [t.lower() for t in posTokens]
negTokens = [t.lower() for t in negTokens]
# remove word punctuation
table = str.maketrans('', '', string.punctuation)
posStripped = [t.translate(table) for t in posTokens]
negStripped = [t.translate(table) for t in negTokens]
# remove special characters
posWords = [w for w in posStripped if w.isalpha()]
negWords = [w for w in negStripped if w.isalpha()]
# remove stop words
stop_words = set(stopwords.words('english'))
posWords = [w for w in posWords if not w in stop_words]
negWords = [w for w in negWords if not w in stop_words]
