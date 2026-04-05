import requests
import os
import pandas as pd
from tqdm import tqdm
import librosa
import numpy as np
import time
import matplotlib.pyplot as plt
from config import XC_API

if __name__ == "__main__":

    base_dir = os.getcwd()

    audio_dir = os.path.join(base_dir, "data/song")
    # spec_dir = os.path.join(base_dir, "data_spectrogram/song")

    os.makedirs(audio_dir, exist_ok=True)
    # os.makedirs(spec_dir, exist_ok=True)

    csv_path = os.path.join(base_dir, "metadata_song.csv")

    API_KEY = XC_API
    QUERY = 'en:"great tit" type:"song" len:0-60'
    PER_PAGE = 100
    MAX_FILES = 500

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

    total_pages = int(data["numPages"])
    print(f"Total pages: {total_pages}")

    download_count = 0
    metadata = []

    for page in range(1, total_pages + 1):

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

            audio_path = os.path.join(audio_dir, f"{file_id}.mp3")
            # spec_path = os.path.join(spec_dir, f"{file_id}.png")

            try:
                # download audio
                r = requests.get(file_url)
                with open(audio_path, "wb") as f:
                    f.write(r.content)
                remarks = rec.get("rmk", "")
                remarks = remarks.replace("\n", " ").replace("\r", " ")
                remarks = remarks.replace('"', "'")

                row = {
                    "id": rec["id"],
                    "english_name": rec["en"],
                    "country": rec["cnt"],
                    "lat": rec["lat"],
                    "lon": rec["lon"],
                    "quality": rec["q"],
                    "length": rec["length"],
                    "time": rec["time"],
                    "date": rec["date"],
                    "sample_rate": rec["smp"],
                    "remarks": remarks,
                    "sex": rec.get("sex", ""),
                    "animal_seen": rec.get("animal-seen", ""),
                    "temp": rec.get("temp", ""),

                    # label for ML
                    "label": "song"
                }

                df = pd.DataFrame([row])

                df.to_csv(
                    csv_path,
                    mode="a",
                    index=False,
                    header=not os.path.exists(csv_path)
                )

                download_count += 1
                time.sleep(0.5)

            except Exception as e:
                print(f"Failed {file_id}: {e}")

    print(f"\nDownloaded {download_count} files")