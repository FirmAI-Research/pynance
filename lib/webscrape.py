import pandas as pd
import requests

def get():
    url = 'https://finviz.com/groups.ashx'

    header = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest"
    }

    def request_response():
        r = requests.get(url, headers=header)

        dfs = pd.read_html(r.text)

        data = dfs[0]

        print(data)

