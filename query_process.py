import json
import torch
from utils.lfqa_utils import *
from joblib import Parallel, delayed, parallel_backend
import os
import math

device = torch.device('cpu') # Change this if you want

TRAIN_DATA_FILE_PATH = r'/data/ELI5_train_10_doc.json'
RESULT_FOLDER_PATH = r'../results'
TRAIN_DATA_BATCH_NUM = 25
CONTEXT_LENGTH = 10


def retrieve(question, qs_model, qs_tokenizer, device, limit_doc=30):
    question_embd = get_embds_qs(qs_model, qs_tokenizer, question, device=device)
    documents_wiki = query_embd(question_embd, limit_doc=limit_doc)
    return [doc[-1] for doc in documents_wiki]

def create_dict(original_data, qs_model, qs_tokenizer):
    question = original_data['question']
    retrieve_result = retrieve(question, qs_model, qs_tokenizer, device, limit_doc=CONTEXT_LENGTH)
    # need to safe type cast retrieve_result to be able to seralize result to JSON
    if retrieve_result is None:
        retrieve_result = ''
    return {
        'question_id': original_data['question_id'],
        'question': question,
        'answers': original_data['answers'],
        'ctxs': retrieve_result
    }

def load_data(path = TRAIN_DATA_FILE_PATH):
    with open(path) as f:
        return json.load(f)

def save_data(documents, index, path=RESULT_FOLDER_PATH):
    if os.path.exists(path) == False:
        os.makedirs(path)
    with open(os.path.join(path, f'MARCO_processed_{10000*(index+1)}.json'), 'w') as f:
        json.dump(documents, f)

def main():
    qs_model, qs_tokenizer = load_model_qs(device=device)
    train_data = load_data()
    batch_size = math.ceil(len(train_data) / TRAIN_DATA_BATCH_NUM)

    for i in range(0, TRAIN_DATA_BATCH_NUM):
        with parallel_backend('threading', n_jobs=-1):
            documents = Parallel()(delayed(create_dict)(train_datapoint, qs_model, qs_tokenizer) for train_datapoint in train_data[batch_size*i:batch_size*(i+1)])
            save_data(documents, i)

if __name__ == "__main__":
    with open('test.json', 'r') as f:
        train_data = json.load(f)
        qs_model, qs_tokenizer = load_model_qs(device=device)
        retrieve_result = retrieve(train_data['question'], qs_model, qs_tokenizer, device, limit_doc=CONTEXT_LENGTH)
        print(retrieve_result)


