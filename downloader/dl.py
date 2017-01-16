import urllib.request
import requests
from time import sleep
import sys

user_name = sys.argv[1]

### Download user ids

request = u"https://halite.io/api/web/user?fields%5B%5D=isRunning&values%5B%5D=1&orderBy=rank&limit=1000&page=0"

r = requests.get(request)
users = r.json()["users"]
user_ids = [user['userID'] for user in users]


### Download games from one user

# user_name = "nmalaguti"

user_id = [u['userID'] for u in users if u['username']==user_name][0]
request = "https://halite.io/api/web/game?userID={}&limit=1000".format(user_id)
r = requests.get(request)
game_ids = [game['replayName'] for game in r.json()]

testfile = urllib.request.URLopener()

n = len(game_ids)
print('DOWNLOADING')
for i,game_id in enumerate(game_ids):
    print('{}/{}'.format(i+1,n))
    request = "https://s3.amazonaws.com/halitereplaybucket/{}".format(game_id)
    testfile.retrieve(request, "{}.gzip".format(game_id))