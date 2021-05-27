
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
sys.path.append(
    '/home/ritvik/Final Year Project/Music Recommendation system/preprocessing')
import pandas as pd
from make_dataset import get_dataset
import json
import random

def get_dataframe():
    return pd.read_csv(
        '/home/ritvik/Final Year Project/Music Recommendation system/preprocessing/music.csv')

def get_similarity():
    data = get_dataframe()
    # creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['combined'])
    # creating a similarity score matrix
    similarity = cosine_similarity(count_matrix)
    return similarity

def get_music_based_on_mood(mood):
    
    with open('/home/ritvik/Final Year Project/Music Recommendation system/genres_with_mood.json', 'r') as fp:
        genres = json.load(fp)
    df = get_dataframe()
    genre = genres[mood.lower()]
    x = df['Genre'].str.contains(genre)
    return df[x].reset_index(drop = True)


def recommend(album):
    
    data = get_dataframe()
    similarity = get_similarity()
    if album not in data['Album'].unique():
        return('Sorry! The Album you requested is not in our database. Please check the spelling or try with some other music')
    else:
        i = data.loc[data['Album'] == album].index[0]

        lst = list(enumerate(similarity[i]))

        lst = sorted(lst, key=lambda x: x[1], reverse=True)
        # excluding first item since it is the requested album itself
        lst = lst[1:11]
        music = dict()

        music['music_titles'] = []
        music['music_artists'] = []
        
        for i in range(len(lst)):
            a = lst[i][0]
            row = data.loc[a]
            music['music_titles'].append(row['Album'])
            music['music_artists'].append(row['Artist'])

        return music

def get_music_dict(mood):
    recommendations = get_music_based_on_mood(mood)
    titles = recommendations['Album']
    
    return recommend(random.choice(titles))
