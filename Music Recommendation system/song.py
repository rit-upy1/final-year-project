from vlc import MediaPlayer
import time
import vlc
import os
import random

def play(song_path):

    # creating a vlc instance
    vlc_instance = vlc.Instance()

    # creating a media player
    player = vlc_instance.media_player_new()

    # creating a media
    media = vlc_instance.media_new(song_path)

    # setting media to the player
    player.set_media(media)
    player.play()
    time.sleep(0.5)

    duration = player.get_length()
    print("Duration: ", duration)
    time.sleep(duration//1000)
    #time.sleep(10)
  
    print(player.will_play())


def play_song_based_on_genre(genre):
    path = os.path.join("genres/", genre)
    files = os.listdir(path)
    song = random.choice(files)
    play(os.path.join(path,song))