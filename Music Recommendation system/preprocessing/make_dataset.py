import pandas as pd
import numpy as np


def get_music_dataframe(link,jan_df_number):
    df = pd.read_html(link, header=0)
    music_df = pd.DataFrame(columns=df[jan_df_number].columns.values)
    for i in range(jan_df_number, jan_df_number + 12):
        current_df = df[i]
        columns = current_df.columns.values
        
        if 'Genre (s)' in columns:
            index = np.where(columns == 'Genre (s)')
            columns[index] = 'Genre'
            current_df.columns = columns
        if 'Genre(s)' in columns:
            index = np.where(columns == 'Genre(s)')
            columns[index] = 'Genre'
            index = np.where(columns == 'Artist(s)')
            columns[index] = 'Artist'
            current_df.columns = columns
        music_df = music_df.append(current_df, ignore_index=True)

    return music_df


def convert_to_lower_string(s):
    if type(s) != str:
        return s
    return s.lower()

def get_dataset():
    #640 rows
    columns = ['Album', 'Artist', 'Genre']
    year_with_starting_number_american = {'2020': 4, '2019': 3, '2018': 3}
    year_with_starting_number_korean = {'2020': 3, '2019': 3, '2018': 3}

    df = pd.DataFrame(columns = columns)
   
    for year in year_with_starting_number_american:
        print(year)
        american_song_dataset = get_music_dataframe(
            f"https://en.wikipedia.org/wiki/{year}_in_American_music",
             year_with_starting_number_american[year])
        american_song_dataset.drop(columns=['Date'], inplace=True)
        korean_dataset = get_music_dataframe(
            f"https://en.wikipedia.org/wiki/{year}_in_South_Korean_music",
            year_with_starting_number_korean[year])
        if ('Ref.' in korean_dataset.columns.values):
            korean_dataset.columns = [
                'Date', 'Album', 'Artist', 'Genre', 'Ref']
        
        korean_dataset.drop(columns=['Date', 'Ref'], inplace=True)
        american_song_dataset = american_song_dataset.append(korean_dataset, ignore_index=True)
        df = df.append(american_song_dataset)
   
    return df

def convert_dataframe_to_csv():
    df = get_dataset()
    df.dropna(inplace=True)
    df['Genre'] = df['Genre'].apply(lambda x: x.lower())
    df['combined'] = df['Album'] + ' ' + df['Artist'] + ' ' + df['Genre'] + ' '
    df.to_csv('music.csv',index=False)
