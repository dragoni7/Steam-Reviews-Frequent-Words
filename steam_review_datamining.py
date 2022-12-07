import pandas as pd

# set the path to project location
path = str(__file__)
path = path.split("steam_review_datamining.py")
path = path[0].replace('\\', '/')

# import the data:
steam_reviews = pd.read_csv(path + 'reviews/dataset.csv')
#steam_reviews = steam_reviews.iloc[:1000,:]

# sort data by review score and save as .csv
pos_reviews = steam_reviews[steam_reviews['review_score'] == 1] # score of 1 is positive.
print(pos_reviews.head())
pos_reviews.to_csv(path + 'reviews/positive_reviews.csv', index=False)

neg_reviews = steam_reviews[steam_reviews['review_score'] < 1] # score less than 1 is negative.
print(neg_reviews.head())
neg_reviews.to_csv(path + 'reviews/negative_reviews.csv', index=False)
