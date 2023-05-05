import discord
import os
import csv
import tensorflow as tf
import ast
import numpy as np
import model as md
import traceback
import singlestep
import pathlib

def train(id, text, name) -> None:
    MESSAGE_LENGTH = 80
    BATCH = 128
    BUFF = 10000
    try:
        if os.name == 'nt':
            path = '..\\models\\'  +name+".ckpt"
        else:
            path = '../models/' + name + ".ckpt"
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=path,
                                                 save_weights_only=True,
                                                 verbose=1)
        
        chars = sorted(list(set(text)))
        char_to_int = tf.keras.layers.StringLookup(vocabulary=list(chars), mask_token=None)
        int_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_int.get_vocabulary(), invert=True, mask_token=None)

        ids = char_to_int(tf.strings.unicode_split(text, 'UTF-8'))
        ids_dataset = tf.data.Dataset.from_tensor_slices(ids)
        sequences = ids_dataset.batch(MESSAGE_LENGTH+1, drop_remainder=True)
        dataset = sequences.map(split_input_target)

        dataset = (dataset.shuffle(BUFF).batch(BATCH, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))
        size = len(char_to_int.get_vocabulary())

        model = md.ParrotGen(
        vocab_size=len(chars),
        embedding_dim=256,
        rnn_units=1024)

        model.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),optimizer='adam')    

        try:
            model.load_weights(path)

        except:
            model.fit(dataset, epochs=30, callbacks=[cp_callback])

        for input_example_batch, target_example_batch in dataset.take(1):
            example_batch_predictions = model(input_example_batch)
        
        sample = tf.random.categorical(example_batch_predictions[0], num_samples=1)
        sample = tf.squeeze(sample, axis=-1).numpy()

        states = None
        next_char = tf.constant(['squawk!\n\n'])
        result = [next_char]
        one_step_model = singlestep.OneStep(model, int_to_char, char_to_int)
        for n in range(100):
            next_char, states = one_step_model.generate_one_step(next_char, states=states)
            result.append(next_char)

        
        if not os.path.exists(path):
            os.makedirs(path)
            model.save_weights(path.format(epoch=30))
        return tf.strings.join(result)[0].numpy().decode("utf-8")
        
    except Exception as e:
        return traceback.format_exc()[:1999]

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text