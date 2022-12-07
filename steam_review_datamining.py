import pandas as pd

# set the path to project location
path = str(__file__)
path = path.split("steam_review_datamining.py")
path = path[0].replace('\\', '/')

# import the data:
steam_reviews = pd.read_csv(path + 'SteamReviewFrequentWords/data/steam.csv')