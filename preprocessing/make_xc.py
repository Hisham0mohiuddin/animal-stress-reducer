import requests
import os
import pandas as pd
from tqdm import tqdm
import time 
from config import XC_API
if __name__ == "__main__":
    crnt_path = os.getcwd()
    # print(crnt_path)
    audio_path  = os.path.join(crnt_path,"data/song")
    # print(data_path)
    os.makedirs(audio_path,exist_ok = True)
    API_KEY = XC_API
    QUERY = 'en:"great tit" type:"song" q:A len:0-60'
    PER_PAGE = 100
    MAX_FILES = 300

    base_url = "https://xeno-canto.org/api/3/recordings"

    params = {
        "query": QUERY,
        "key": API_KEY,
        "per_page": PER_PAGE,
        "page": 1
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

            file_url = rec["file"]
            file_id = rec["id"]
            filename = os.path.join(audio_path, f"{file_id}.mp3")
            try:
                r = requests.get(file_url)
                with open(filename, "wb") as f:
                    f.write(r.content)

                metadata.append({
                    "id": file_id,
                    "english_name": rec["en"],
                    "country": rec["cnt"],
                    "location": rec["loc"],
                    "lat": rec["lat"],
                    "lon": rec["lon"],
                    "type": rec["type"],
                    "quality": rec["q"],
                    "length": rec["length"],
                    "time": rec["time"],
                    "date": rec["date"],
                    "temp": rec.get("temp", ""),
                    "sample_rate": rec["smp"]
                })

                download_count += 1

            except Exception as e:
                print(f"Failed {file_id}: {e}")
            df = pd.DataFrame(metadata)
            new_data = pd.DataFrame([metadata[-1]]) # Get only the last item added
            csv_path = "data/metadata.csv"
            new_data.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))   

print(f"\nDownloaded {download_count} files")

df = pd.DataFrame(metadata)
df.to_csv("data/metadata.csv", index=False)







