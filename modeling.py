import discord
import os
import csv
import tensorflow as tf
import ast
import numpy as np

def talk(model, char_to_int, int_to_char) -> str:
    evaluated = [char_to_int[s] for s in "be epic"]
    evaluated = tf.expand_dims(evaluated, 0)

    to_join = []
    temp = 1.0
    model.reset_states()
    for i in range(100):
        predict = model(evaluated)
        predictions = tf.squeeze(predict, 0)
        predict = predict/temp
        predict_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        evaluated = tf.expand_dims([predict_id], 0)
        to_join.append(int_to_char[predict_id])
    return ''.join(to_join)



def train(id, text) -> None:
    
    chars = sorted(list(set(text)))
    char_to_int = {ch:i for i, ch in enumerate(chars)}
    int_to_char = {i:ch for i, ch in enumerate(chars)}

    longest_sequence = max([len(s) for s in text])
    train_x = []
    train_y = []

    for x in range(len(text)):
        try:
            train_x.append([char_to_int[q] for q in text[x:x+longest_sequence]])
            train_y.append(char_to_int[text[x+longest_sequence]])
        except:
            train_x.pop()
            continue
    train_x = tf.keras.preprocessing.sequence.pad_sequences(train_x, maxlen=longest_sequence, padding='post')

    train_y = tf.keras.utils.to_categorical(train_y)

    model = tf.keras.Sequential([tf.keras.layers.Embedding(input_dim=len(chars), output_dim=64),
    tf.keras.layers.LSTM(13728),
    tf.keras.layers.Dense(2),            
    ])

    model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy')
    model.summary()
    model.fit(train_x, train_y, epochs=1)
    
    return talk(model,char_to_int,int_to_char)
    