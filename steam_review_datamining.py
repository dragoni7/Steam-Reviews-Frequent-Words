import string
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk import FreqDist

# set the path to project location
path = str(__file__)
path = path.split("steam_review_datamining.py")
path = path[0].replace('\\', '/')

stop_words = set(stopwords.words('english')) # set the stop words
stop_words.update(['game', 'games', 'gaming', 'play', 'playing', 'played']) # these words appear frequently in both negative and positive reviews, therefore they have little significance
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
        if tag.startswith('NN'): # Nouns
            pos = 'n'
        elif tag.startswith('VB'): # Verbs
            pos = 'v'
        else:
            pos = 'a'
        result.append(WordNetLemmatizer().lemmatize(word, pos))
    return result

# fast method for flattening a list. Source: https://chrisconlan.com/fastest-way-to-flatten-a-list-in-python/
def flatten_sentence(sentence_list):
    result = []
    for sentence in sentence_list:
        result += sentence
    return result

# plots a word cloud
def show_cloud(wordCloud):
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordCloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.show()

if __name__ == "__main__":
    # import the data:
    steam_reviews = pd.read_csv(path + 'reviews/dataset.csv')
    steam_reviews = steam_reviews.sample(frac=0.35) # use a percentage of the data for efficiency
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

    # get frequent words
    posWordFreq = FreqDist(allPosWords)
    negWordFreq = FreqDist(allNegWords)
    commonPosWords = [t[0] for t in posWordFreq.most_common(30)]
    commonNegWords = [t[0] for t in negWordFreq.most_common(30)]
    
    # get unique words for positive and negative reviews
    commonPosWords, commonNegWords = list(set(commonPosWords) - set(commonNegWords)), list(set(commonNegWords) - set(commonPosWords))

    # generate a word cloud
    posWordCloud = WordCloud(width = 800, height = 800, background_color="white").generate(" ".join(commonPosWords))
    negWordCloud = WordCloud(width = 800, height = 800, background_color="white").generate(" ".join(commonNegWords))

    show_cloud(posWordCloud)
    show_cloud(negWordCloud)
