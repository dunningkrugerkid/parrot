import discord
import os
import csv
import tensorflow as tf
import ast

client = discord.Client()
messageDictionary = {}

@client.event
async def on_ready() -> None:
 
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
    
    train()
    
@client.event
async def on_message(message) -> None:
    if message.content.startswith("parrot!"):
        if len(message.mentions) > 0:
            user = message.mentions[0]
        else:
            user = message.author
        # search csv for user.name. need format of csv from sven   
        try:
            chat(user.id)
        except:
            await message.get_channel().send('user not present in database')

async def chat(id) -> None: 
    # search db for user
    train(id)
    

async def train(id) -> None:
    if messageDictionary.get(id) != None:
        vocab = list(sorted(set(messageDictionary.get(id))))
    else:
        return

    ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
    
    chars_from_ids = tf.keras.layers.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

    for ids in ids_from_chars.take(10):
        print(chars_from_ids(ids).numpy().decode('utf-8'))
client.run("token")
        
