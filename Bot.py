import discord
import os
import csv
import tensorflow as tf
import ast
from dotenv import load_dotenv
from modeling import *

load_dotenv()
intents = discord.Intents.default() 
client = discord.Client(intents = intents)
messageDictionary = {}
TOKEN = os.getenv("DISCORD_TOKEN")

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
    print("load successful!")
    
@client.event
async def on_message(message) -> None:
    if message.content.startswith("parrot!"):
        if len(message.mentions) > 0:
            user = message.mentions[0]
        else:
            user = message.author
        # search csv for user.name. need format of csv from sven 
        try:
            id = user.id
            text = "\n".join(messageDictionary.get(id))
            await message.channel.send(train(id, text))
        except:
            await message.channel.send('user not present in database')

client.run(TOKEN)
        
