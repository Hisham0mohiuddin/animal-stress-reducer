import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

class BirdDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, batch_size=32, shuffle=True):
        self.df = dataframe.reset_index(drop=True)
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

        X, y = [], []

        for _, row in batch_df.iterrows():
            img = np.load(row['npy_path'])

            # normalize for MobileNetV2
            img = (img * 2.0) - 1.0

            X.append(img)
            y.append(self.label_map[row['label']])

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

if __name__ == "__main__":

    df = pd.read_csv('df_index_to_npy.csv')

    sample = np.load(df.iloc[0]['npy_path'])
    print("Sample shape:", sample.shape)

    INPUT_SHAPE = sample.shape

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )

    train_gen = BirdDataGenerator(train_df, batch_size=32)
    test_gen = BirdDataGenerator(test_df, batch_size=32, shuffle=False)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalMaxPooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Recall()
        ]
    )
    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=10
    )

    model.save("bird_model.keras")

