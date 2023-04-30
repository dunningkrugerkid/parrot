import discord
import os
import csv
import tensorflow as tf
import ast
import numpy as np

def train(id, text) -> None:
    MESSAGE_LENGTH = 80
    BATCH = 128
    BUFF = 10000
    
    chars = sorted(list(set(text)))
    char_to_int = tf.keras.layers.StringLookup(vocabulary=list(chars), mask_token=None)
    int_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_int.get_vocabulary(), invert=True, mask_token=None)

    ids = char_to_int(tf.strings.unicode_split(text, 'UTF-8'))
    ids_dataset = tf.data.Dataset.from_tensor_slices(ids)
    sequences = ids_dataset.batch(MESSAGE_LENGTH+1, drop_remainder=True)
    dataset = sequences.map(split_input_target)

    dataset = (dataset.shuffle(BUFF).batch(BATCH, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))
    size = len(char_to_int.get_vocabulary())

    inp = tf.keras.layers.Input(shape=(size,))
    l1 = tf.keras.layers.Embedding(size, 256)(inp)
    l2 = tf.keras.layers.GRU(1024)(l1)
    l3 = tf.keras.layers.Dense(size)(l2)
    model = tf.keras.Model(inputs=inp, outputs=l3)

    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam')
    model.summary()
    model.fit(dataset, epochs=1)

    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
    
    sample = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sample = tf.squeeze(sample, axis=-1).numpy()
    return str(tf.strings.reduce_join(int_to_char(sample), axis=-1).numpy())

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text