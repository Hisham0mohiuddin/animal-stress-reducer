from filtering import standard_equal_l
import numpy as np
import pandas as pd
import os

def make_df():
    '''
    this makes a dataframe of the path of the 3D data of the images of audio stored in npy files
    '''
    data = []
    home = os.getcwd()
    raw_data = os.path.join(home,"data")
    processed_save_path = os.path.join(home,'processed_npy')
    os.makedirs(processed_save_path,exist_ok=True)

    # looping thru to get teh file and make its 3d matrix
    for type in os.listdir(raw_data):
        label_path = os.path.join(raw_data,type)
        if(os.path.isdir(label_path)):
            for file in os.listdir(label_path):
                if file.endswith("mp3"):

                    final_raw_path = os.path.join(label_path, file)

                    try:
                        y_3d, sr = standard_equal_l(final_raw_path)

                        npy_file = f"{type}_{file.replace('.mp3','.npy')}"
                        save_path = os.path.join(processed_save_path, npy_file)

                        np.save(save_path, y_3d.astype(np.float32))

                        data.append({
                            "file_id": file,
                            "label": type,
                            "npy_path": save_path
                        })

                    except Exception as e:
                        print(f"❌ Skipping bad file: {file} | Error: {e}")
                        continue
    df = pd.DataFrame(data)
    df.to_csv('df_index_to_npy.csv', index=False)

if __name__=="__main__":
    make_df()