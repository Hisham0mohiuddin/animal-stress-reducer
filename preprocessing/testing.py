import requests
import os
import pandas as pd
from tqdm import tqdm
import time 
from config import XC_API
if __name__ == "__main__":
    crnt_path = os.getcwd()
    # print(crnt_path)
    audio_path  = os.path.join(crnt_path,"data/alarm_call")
    # print(data_path)
    os.makedirs(audio_path,exist_ok = True)
    API_KEY = XC_API
    QUERY = 'en:"great tit" type:"alarm call" len:0-60'
    PER_PAGE = 100
    MAX_FILES = 1

    base_url = "https://xeno-canto.org/api/3/recordings"

    params = {
        "query": QUERY,
        "key": API_KEY,
        "per_page": PER_PAGE,
        "page": 250
    }

    response = requests.get(base_url, params=params)
    response.raise_for_status()
    data = response.json()

    n = int(data["numPages"])
    print(f"Total pages: {n}")

    metadata = []
    download_count = 0

    for page in range(1, n + 1):

        if download_count >= MAX_FILES:
            break

        params["page"] = page
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        for rec in tqdm(data["recordings"]):
            
            if download_count >= MAX_FILES:
                break

            print(rec)

print(f"\nDownloaded {download_count} files")







