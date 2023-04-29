import discord
import os
import csv
import tensorflow as tf
import ast
import numpy as np
messageDictionary = {}

with open('compsci.csv', newline='', encoding="utf8") as csvfile:
        # parsing
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            if(row[0].startswith("Author")):
                continue
            sep = '-0'
            row[0] = row[0].split(sep, 1)[0]
            try:
                row = eval(row[0])
            except:
                continue # null/malformed value, we don't need these
            author = row[0]
            message = row[2]
            message = message.replace('"', "")
            message = message.replace("[", "")
            message = message.replace("]", "")
            message = message.replace(",", " ")

            messageList = messageDictionary.get(author) if messageDictionary.get(author) is not None else []
            messageList.append(message)

            messageDictionary.update({author:messageList})

def chat(id) -> None: 
    # search db for user
    train(id)

def train(id) -> None:
    
    text = "\n".join(messageList)
    chars = sorted(list(set(text)))
    char_to_int = {ch:i for i, ch in enumerate(chars)}
    int_to_char = {i:ch for i, ch in enumerate(chars)}

    longest_sequence = max([len(s) for s in text])
    train_x = []
    train_y = []

    for x in range(len(text)):
        train_x.append([char_to_int[q] for q in text[x:x+longest_sequence]])
        train_y.append(char_to_int[text[x+max_len]])
    train_x = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=max_len, padding='post')
    train_y = tf.keras.utils.to_categorical(y)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=len(chars), output_dim=64))
    model.add(tf.keras.layers.LSTM(units=128))
    model.add(tf.keras.layers.Dense(units=len(chars), activation='softmax'))

    model.fit(X, y, epochs=3, batch_size=64)


train(147670976551845888)
    