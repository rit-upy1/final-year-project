from googleapiclient.discovery import build

API_KEY = 'AIzaSyAnVlazTy4mQv18h9KJBPemnqpKFlEb7Nk'
youtube = build('youtube','v3',developerKey = API_KEY)

request = youtube.search().list(
    part='snippet', #required parameter
    safeSearch="none",
    type='video',
    q = 'What is love haddway'
)

response = request.execute()
for i in response:
    print(response, '\n' * 10)
youtube.close()
