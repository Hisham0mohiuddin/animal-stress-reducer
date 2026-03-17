import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from classification_models.tfkeras import Classifiers

class BirdDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, batch_size=32, shuffle=True):
        self.df = dataframe
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # Number of batches per epoch
        return int(np.floor(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        # Shuffles the data after every epoch so the model doesn't learn the order
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, index):
        # Get the rows for the current batch
        batch_df = self.df[index * self.batch_size : (index + 1) * self.batch_size]
        
        X = []
        y = []
        
        for _, row in batch_df.iterrows():
            # Load the 3D data from the path in your DataFrame
            img = np.load(row['npy_path'])
            X.append(img)
            
            label_map = {'alarm_call': 0, 'call': 1, 'song':1}
            y.append(label_map[row['label']])
            
        return np.array(X), np.array(y)

if __name__ =="__main__":
    base_model = tf.keras.applications.MobileNetV2(
        input_shape = (128,431,3),
        include_top = False,
        weights = 'imagenet'
    )
    base_model.trainable = True

    model = models.Sequential([
        base_model,
        layers.GlobalMaxPooling2D(),
        layers.Dense(128,activation= 'relu'),
        layers.Dropout(0.3),
        layers.Dense(1,activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss = 'binary_crossentropy',
        metrics= ['accuracy','precision']
    )
    # model.summary()
    df = pd.read_csv('df_index_to_npy.csv')
    train_df , test_df = train_test_split(df,test_size=0.2,random_state=42,stratify=df['label'])

    train_gen = BirdDataGenerator(train_df)
    test_gen = BirdDataGenerator(test_df)

    history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=10  
    )   


