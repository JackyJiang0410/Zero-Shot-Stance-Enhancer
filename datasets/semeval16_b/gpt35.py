import os
import openai

from RAG import *
from tqdm import tqdm
from openai import OpenAI

client = OpenAI()

prompt_stance = """Target: {target}. Sentence: {text}. 
Please expand the abbreviations, slang, and hashtags in the [Sentence] into complete phrases and sentences to restate the text.
Based on information related to the topic from Wikipedia is :
{information}
Please extract and summarize background information that is valuable to the text. If it is irrelevant information, please add some.
Please reply in a few sentences."""

import pandas as pd
from tqdm import tqdm
import time


def get_reply(text, target="Hillary Clinton", information=''):
    prompt = prompt_stance.replace("{Sentence}", text).replace("{target}", target).replace("{information}", information)
    count = 0
    while count < 3:
        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system",
                     "content": "You're an expert in the field of stance testing and are good at adding background knowledge based on sentences."},
                    {"role": "user", "content": prompt}
                ]
            )
            break
        except:
            time.sleep(10)
            count += 1

    return completion.choices[0].message.content


db = VectorDatabase()
db.load_texts('HC.pkl')
db.load_faiss('HC.faiss')

tqdm.pandas(desc="Computing embeddings")
df = pd.read_csv("semeval16.csv", encoding='utf-8')

df = df[df['target'] == 'Hillary_Clinton']
df = df.reset_index()
for i in tqdm(range(len(df))):
    text = df.loc[i, 'text']
    information = db.query(text, 200)
    background = get_reply(text, "Hillary Clinton", information)
    df.loc[i, 'background'] = background

output_path = "./covid0.csv"
df.to_csv('Hillary_Clinton.csv', index=False, encoding='utf-8')
