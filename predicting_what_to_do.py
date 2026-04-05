import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. SETTINGS & PREVENT WARNINGS
pd.set_option('future.no_silent_downcasting', True)

# --- STEP 2: THE DATA GENERATOR ---
class BirdMultiModalGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, meta_columns, batch_size=32, shuffle=True):
        self.df = dataframe.reset_index(drop=True)
        self.meta_columns = meta_columns
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.label_map = {'alarm_call': 0, 'song': 1}
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        batch_idx = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_idx]

        X_img, X_meta, y = [], [], []

        for _, row in batch_df.iterrows():
            # A. Load Image and Pad to 431 width (Expected by your model)
            img = np.load(row['npy_path'])
            if img.shape[1] < 431:
                pad_width = 431 - img.shape[1]
                img = np.pad(img, ((0, 0), (0, pad_width), (0, 0)), mode='constant')
            elif img.shape[1] > 431:
                img = img[:, :431, :]
            
            img = (img * 2.0) - 1.0  # MobileNetV2 Norm
            X_img.append(img)

            # B. Load Metadata
            meta_vector = row[self.meta_columns].infer_objects(copy=False).fillna(0).values.astype(np.float32)
            X_meta.append(meta_vector)

            # C. Robust Label Handling
            label_cols = [c for c in batch_df.columns if 'label' in c]
            target_col = label_cols[0]
            label_val = row[target_col]
            if isinstance(label_val, pd.Series):
                label_val = label_val.iloc[0]
            y.append(self.label_map[label_val])

        # RETURN DICTIONARY for Multi-Input tracing
        return (
            {
                "audio_input_layer": np.array(X_img, dtype=np.float32),
                "metadata_input": np.array(X_meta, dtype=np.float32)
            },
            np.array(y, dtype=np.int32)
        )

# --- STEP 3: THE MODEL BUILDER ---
def build_merged_model(existing_model_path, num_meta_features):
    # Load old spectrograph model
    base_model = tf.keras.models.load_model(existing_model_path)
    
    # Entrance 1: Audio (Must match 128x431)
    audio_input = layers.Input(shape=(128, 431, 3), name="audio_input_layer")
    audio_features = base_model(audio_input)

    # Entrance 2: Metadata (The CSV numbers)
    meta_input = layers.Input(shape=(num_meta_features,), name="metadata_input")
    # NAMED LAYER for Explainability later
    m = layers.Dense(64, activation='relu', name="meta_feature_extractor")(meta_input)
    m = layers.BatchNormalization()(m)
    m = layers.Dense(32, activation='relu')(m)

    # Merger
    combined = layers.Concatenate()([audio_features, m])
    
    # Final Decision Head
    z = layers.Dense(32, activation='relu')(combined)
    z = layers.Dropout(0.2)(z)
    output = layers.Dense(2, activation='softmax', name="final_prediction")(z)

    full_model = models.Model(inputs=[audio_input, meta_input], outputs=output)
    full_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return full_model

# --- STEP 4: EXECUTION ---
if __name__=="__main__":
    # 1. Sync CSVs
    df_meta = pd.read_csv("final_metadata.csv")
    df_paths = pd.read_csv("df_index_to_npy.csv")
    merged_df = pd.concat([df_meta,df_paths.drop("id",axis=1)],axis =1)

    print(merged_df.head())

    meta_cols = [c for c in df_meta.columns if c not in ['id', 'label', 'Unnamed: 0']]

    # 2. Setup Generator & Model
    train_gen = BirdMultiModalGenerator(merged_df, meta_cols)
    model = build_merged_model("bird_model.keras", len(meta_cols))

    # 3. Train
    print("Starting Multi-Modal Training...")
    model.fit(train_gen, epochs=10)

    # 4. EXPLAINABILITY (The "Why")
    print("\n" + "="*40)
    print("ANALYZING METADATA ATTRIBUTE IMPORTANCE")
    print("="*40)
    
    try:
        # Pull weights from our specifically named metadata layer
        target_layer = model.get_layer("meta_feature_extractor")
        weights = target_layer.get_weights()[0]
        
        # Calculate which CSV column had the most 'voice'
        importance = np.mean(np.abs(weights), axis=1)
        feat_importance = pd.Series(importance, index=meta_cols).sort_values(ascending=False)

        print(feat_importance.head(10))
    except Exception as e:
        print(f"Explainability Error: {e}")

    model.save("full_model.keras")