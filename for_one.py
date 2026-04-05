import pandas as pd
from predicting_what_to_do import BirdMultiModalGenerator 
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def explain_prediction(model, image_array, meta_vector, meta_cols):
    # 1. CLEANING: Convert strings to numbers and handle NaNs
    clean_meta = pd.to_numeric(meta_vector, errors='coerce')
    clean_meta = np.nan_to_num(clean_meta, nan=0.0).astype(np.float32)

    # 2. PREDICT: Using the dictionary keys defined in your model/generator
    input_dict = {
        "audio_input_layer": np.expand_dims(image_array, axis=0),
        "metadata_input": np.expand_dims(clean_meta, axis=0)
    }
    
    prediction = model.predict(input_dict, verbose=0)
    
    # 3. EXPLAINABILITY MATH
    meta_layer = model.get_layer("meta_feature_extractor")
    weights = meta_layer.get_weights()[0] 

    # Contribution = Attribute Value * Average Weight Impact
    contributions = np.abs(clean_meta * weights.mean(axis=1))

    # 4. REPORTING
    analysis = pd.Series(contributions, index=meta_cols).sort_values(ascending=False)
    guess = "Song" if np.argmax(prediction) == 1 else "Alarm Call"
    
    print(f"\n" + "="*30)
    print(f"PREDICTION: {guess}")
    print(f"="*30)
    print("Top Influencing Attributes for this specific file:")
    print(analysis.head(5))

    return analysis

if __name__ == "__main__":
    # 1. Setup Data
    df_meta = pd.read_csv("final_metadata.csv")
    df_paths = pd.read_csv("df_index_to_npy.csv")
    # Using merge is safer to ensure IDs align correctly
    merged_df = pd.concat([df_meta,df_paths.drop("id",axis=1)],axis =1)
    
    model = tf.keras.models.load_model('full_model.keras')
    meta_cols = [c for c in df_meta.columns if c not in ['id', 'label', 'Unnamed: 0']]

    # 2. Setup Generator
    train_gen = BirdMultiModalGenerator(merged_df, meta_cols)

    # --- FIX: THE CORRECT UNPACKING LOGIC ---
    # train_gen[0] returns (inputs_dict, labels_array)
    inputs_dict, test_labels = train_gen[0]

    # Extract the arrays from the dictionary using the keys
    test_imgs = inputs_dict["audio_input_layer"]
    test_metas = inputs_dict["metadata_input"]

    # 3. Analyze the first bird (Index 0)
    # test_imgs[0] is now the actual pixel array, not a string!
    explain_prediction(model, test_imgs[0], test_metas[0], meta_cols)