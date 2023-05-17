import discord
import os
import csv
import tensorflow as tf
import ast
from dotenv import load_dotenv
from generate import *

load_dotenv()
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
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
            if not (message.startswith("`") and message.endswith("`")):
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
        id = user.id
        
        if messageDictionary.get(id) is not None:
            print(message.content.split())
            text = "\n".join(messageDictionary.get(id))
            seed = "" + " ".join([x for x in message.content.split() if (not x.startswith("<@") and not(x.startswith("parrot!")))])
            async with message.channel.typing():
                await message.channel.send(train(id, text, user.name, seed))
        else:
            await message.channel.send(user.name + ' not present in database')

    elif message.content == "parrot train everyone!":
        for member in message.guild.members:
            id = member.id
            if messageDictionary.get(id) is not None:
                text = "\n".join(messageDictionary.get(id))
                seed = "" + " ".join([x for x in message.content.split() if (not x.startswith("<@") and not(x.startswith("parrot!")))])
                async with message.channel.typing():
                    train(id, text, member.name, seed)
                    await message.channel.send("completed training for " + member.name)


client.run(TOKEN)