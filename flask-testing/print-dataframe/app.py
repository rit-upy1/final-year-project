import sys
sys.path.append('/home/ritvik/Final Year Project/fyp')

sys.path.append('/home/ritvik/Final Year Project/video-game-recommender')
sys.path.append('/home/ritvik/Final Year Project/movie-recommender')
sys.path.append('/home/ritvik/Final Year Project/Music Recommendation system')
sys.path.append('/home/ritvik/Final Year Project/aks/src')
import face_mood_analyser
import emotions
import os

from flask import Flask,render_template,request,url_for,redirect,session
from recommend import get_games_dict
from headless_main import get_movie_dict
from recommend_music import get_music_dict
import secrets

app = Flask(__name__)
secret = secrets.token_urlsafe(32)
app.secret_key = secret

@app.route('/', methods=['POST', 'GET'])
def main():
    if request.method == 'POST':
        user = request.form['sam']
        print("usER IS ", user)
    return render_template('home.html')

@app.route('/recommendations')
def recommendations():
    print('redirection complete')
    mood = session.get('mood')
    print(f'mood is {mood}')
    print('redirection complete')
    movies = get_movie_dict(mood)
    music = get_music_dict(mood)    
    games = get_games_dict(mood)

    return render_template('Recommendation_Page.html', movies = movies,games = games,music = music)
    
    
    
@app.route("/opencamera", methods=["post", 'get'])
def opencamera():
    # return {'msg':"ok"}
    # return  emotions.display()
    isempty = True
    while isempty:
        mood = emotions.display()
        if mood == None:
            continue
        else:
            isempty = False
    mood = mood[0]
    print(mood)
    session['mood'] = mood
    print('redirecting')
    
    return redirect(url_for('recommendations'), code=302)


def get_mood():
    webcam.capture_pictures(10)
    return face_mood_analyser.get_mood('/home/ritvik/Final Year Project/Emotion-detection/src/webcamera/image')


app.run(debug=True)
