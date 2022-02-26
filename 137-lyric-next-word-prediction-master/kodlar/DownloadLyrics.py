import requests
from bs4 import BeautifulSoup
from urllib.parse import urlencode


class GeniusAPI:
    _base_header = {'Authorization': 'Bearer ' + 'oX4gD7dUn7o3x757pAECZCJFTmGKBuG7TFIyv-XZHqpA_Q_glNjxzCkWGhEPO3Ac'}
    _base_params = {'text_format': 'plain'}

    def __init__(self, artist):
        self.artist_name = artist
        self.header = self._base_header
        self.params = self._base_params


    def api_response(self, uri):
        response = requests.get(uri, params=self.params, headers=self.header)

        assert(response.status_code == 200)

        return response


    def song_to_lyrics(self, song_id):
        resp = requests.get(song_id)
        soup = BeautifulSoup(resp.text, 'html.parser')
        txt = soup.get_text()

        end_songname = txt.find("Genius Lyrics") - 3
        start_songname = txt.find(u"\u2013", end_songname - 50) + 2

        songname = txt[start_songname:end_songname].replace("'", "")

        assert(len(songname) != 0)

        start_lyrics = txt.find(songname, end_songname) + len(songname)
        end_lyrics = txt.find('More on Genius', start_lyrics)

        if start_lyrics == len(songname) - 1:
            return ""

        return txt[start_lyrics:end_lyrics]


    def artist_songs(self, artist_id):
        uri = 'https://api.genius.com' + artist_id + '/songs'
        page_number = 1
        songs = []
        num_songs = 0

        self.params['sort'] = 'title'
        self.params['per_page'] = 20

        while True:
            self.params['page'] = page_number
            resp = self.api_response(uri).text

            str_search = '"url"'
            pointer_read = 0

            while True:
                url_start = resp.find(str_search, pointer_read) + len(str_search) + 2

                if url_start == (len(str_search) + 1):
                    break
                url_end = resp.find('"', url_start)
                path = resp[url_start:url_end]

                if path.find('artist') == -1:
                    songs.append(path)

                pointer_read = url_end
            page_number += 1
            if len(songs) == num_songs:
                break
            else:
                num_songs = len(songs)

        return songs


    def search_artists(self, query):
        params = {'per_page': 5, 'q': query}
        url = 'https://genius.com/api/search/multi?' + urlencode(params)
        resp = requests.get(url, timeout=10)
        resp = resp.text
        str_search = '"api_path"'
        path_start = resp.find(str_search) + len(str_search) + 2
        path_end = resp.find('"', path_start)

        return resp[path_start:path_end]


    def download_to_txt(self, txt_file):
        artist_id = self.search_artists(self.artist_name)
        song_ids = self.artist_songs(artist_id)
        self.num_songs = len(song_ids)

        lyrics = ""

        for song in song_ids:
            lyrics += self.song_to_lyrics(song)

        with open(txt_file, 'w+') as f:
            ascii_clear = lyrics.encode('ascii', 'ignore')
            f.write(ascii_clear.decode('utf-8'))

        return
