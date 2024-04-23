import os
import openai
from RAG import *
from tqdm import tqdm
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import time

client = OpenAI()

prompt_stance = """Target: {target}. Sentence: {text}. 
Please expand the abbreviations, slang, and hashtags in the [Sentence] into complete phrases and sentences to restate the text.
Based on information related to the topic from Wikipedia is :
{information}
Please extract and summarize background information that is valuable to the text. If it is irrelevant information, please add some.
Please reply in a few sentences."""


def get_reply(text, target="Hillary Clinton", information=''):
    prompt = prompt_stance.replace("{Sentence}", text).replace("{target}", target).replace("{information}", information)
    count = 0
    while count < 3:
        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system",
                     "content": "You're an expert in the field of stance testing and are good at adding background "
                                "knowledge based on sentences."},
                    {"role": "user", "content": prompt}
                ]
            )
            break
        except:
            time.sleep(10)
            count += 1

    return completion.choices[0].message.content


db = VectorDatabase()
db.load_texts('SCHOOL.pkl')
db.load_faiss('SCHOOL.faiss')

tqdm.pandas(desc="Computing embeddings")
test = pd.read_csv("./raw_test_all_onecol.csv", encoding='utf-8')
train = pd.read_csv("./raw_train_all_onecol.csv", encoding='utf-8')
df = pd.concat([test, train])
df = df.reset_index()

df = df[df['Target 1'] == 'school closures']
df = df.reset_index()
for i in tqdm(range(len(df))):
    text = df.loc[i, 'Tweet']
    information = db.query(text, 200)
    background = get_reply(text, "school closures", information)
    df.loc[i, 'background'] = background

output_path = "./covid0.csv"
df.to_csv('school_closures.csv', index=False, encoding='utf-8')
