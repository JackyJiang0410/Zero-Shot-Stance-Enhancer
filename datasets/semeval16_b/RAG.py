import time

import numpy
import numpy as np
import faiss
import openai
# from openai.embeddings_utils import get_embedding#, cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
import re
import tiktoken
import pickle
import os
from openai import OpenAI
client = OpenAI()

class VectorDatabase:
    def __init__(self, dimension=1536):
        """
        VectorDatabase is used to store and retrieve text embeddings efficiently using FAISS.

        Parameters:
            dimension (int): The dimensionality of the embeddings.
        """
        self.dimension = dimension
        # self.index = faiss.IndexFlatL2(dimension)
        self.index = faiss.IndexFlatIP(dimension)  # L2 norm for FAISS index
        self.texts = []  # unit
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.max_length = 30 # max patch length

    def reset(self):
        self.index = faiss.IndexFlatIP(self.dimension)

    def save_faiss(self, path):
        faiss.write_index(self.index, path)

    def load_faiss(self, path):
        self.index = faiss.read_index(path)

    def save_texts(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.texts, f)
    def load_texts(self, path):
        with open(path, 'rb') as f:
            self.texts = pickle.load(f)

    def _get_embedding(self, text):
        """
            Gets the embedding of a text by querying the OpenAI Embeddings API.
            Retries up to 3 times in case of a failure.

            Parameters:
            text (str): The text to get the embedding for.

            Returns:
            embedding (np.array): The embedding of the text.
        """
        count = 0
        while count < 3:
            try:
                embedding = client.embeddings.create(input = [text], model='text-embedding-ada-002').data[0].embedding
                break

            except:
                time.sleep(3)
                count += 1
        embedding = np.array(embedding)
        embedding = embedding.reshape(1, -1)

        return embedding



    def merge_strings_until_max_length(self, strings, max_length, index):
        """
        Merges strings from a list until the tokenization reaches the max_length limit.
        It ensures that the merged string does not exceed the maximum token length.

        Parameters:
            strings (list): A list of strings to merge.
            max_length (int): The maximum token length allowed for the merged string.
            index (list): The indices of the strings to merge.

        Returns:
            merged_string (str): The merged string.
            restored_sentences (list): List of sentences in their original order.
        """
        merged_list = []
        for i in range(len(strings)):
            if len(self.tokenization(''.join(merged_list))) + len(self.tokenization(strings[i])) > max_length:
                break
            merged_list.append(strings[i])

        # ensure semantic consistency
        index = index[:len(merged_list)]
        index_2 = [sorted(index).index(x) for x in index]
        original_order = sorted(enumerate(merged_list), key=lambda x: index_2[x[0]])
        restored_sentences = [sentence for i, sentence in original_order]
        return ''.join(restored_sentences),restored_sentences


    def tokenization(self, text):
        return self.encoding.encode(text)

    def add_text(self, text, split=True):
        """compute embeddings into FAISS"""
        if split:
            segments = self.split_text(text, self.max_length)
            for sentence in segments:
                embedding = self._get_embedding(sentence)
                if self.index.is_trained:
                    self.index.add(embedding)
                self.texts.append(sentence)
        else:
            embedding = self._get_embedding(text)
            if self.index.is_trained:
                self.index.add(embedding)
            self.texts.append(text)

    def query(self, query_text, max_length, return_list=False):
        """Retrival"""
        query_embedding = self._get_embedding(query_text)
        D, I = self.index.search(query_embedding, 99999)
        # [(self.texts[i], D[0][j]) for j, i in enumerate(I[0])]
        relative_sentense =  [self.texts[i] for i in I[0] if i >= 0]
        index = [i for i in I[0] if i >= 0]
        text, text_list = self.merge_strings_until_max_length(strings=relative_sentense, max_length=max_length, index=index)
        if return_list:
            return text_list
        return text

    def query_num(self, query_text, num):
        """Retrival num"""
        query_embedding = self._get_embedding(query_text)
        D, I = self.index.search(query_embedding, 99999)
        # [(self.texts[i], D[0][j]) for j, i in enumerate(I[0])]
        relative_sentense =  [self.texts[i] for i in I[0] if i >= 0]
        if len(relative_sentense) < num:
            return ''.join(relative_sentense)
        relative_sentense = relative_sentense[:num]
        return ''.join(relative_sentense)


    def find_sentences(self, text):
        sentence_endings = r'([。？！.?!\n]+)'
        sentences = re.split(sentence_endings, text)
        sentences = [sentences[i] + sentences[i+1] for i in range(0, len(sentences)-1, 2)]
        sentences = [i.strip() for i in sentences]
        sentences = [i for i in sentences if i != '']
        return sentences

    def split_text(self, text, max_length):
        sentences = self.find_sentences(text)
        segments = []
        current_segment = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(self.tokenization(sentence))
            if sentence_length > max_length:
                if current_segment:
                    segments.append(''.join(current_segment))
                    current_segment = []
                    current_length = 0
                segments.append(sentence)
            elif current_length + sentence_length <= max_length:
                current_segment.append(sentence)
                current_length += sentence_length
            else:
                segments.append(''.join(current_segment))
                current_segment = [sentence]
                current_length = sentence_length

        if current_segment:
            segments.append(''.join(current_segment))

        return segments

if __name__ == '__main__':


    db = VectorDatabase()
    import json
    with open('1. semeval17_topic_knowledge.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    text = data[2]['related_topics'][0]['content']
    db.add_text(text)
    db.save_texts('FM.pkl')
    db.save_faiss('FM.faiss')

    db = VectorDatabase()
    import json
    with open('1. semeval17_topic_knowledge.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    text = data[3]['related_topics'][0]['content']
    db.add_text(text)
    db.save_texts('HC.pkl')
    db.save_faiss('HC.faiss')

    db = VectorDatabase()
    import json
    with open('1. semeval17_topic_knowledge.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    text = data[4]['related_topics'][0]['content']
    db.add_text(text)
    db.save_texts('LA.pkl')
    db.save_faiss('LA.faiss')

