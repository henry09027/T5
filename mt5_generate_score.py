import json
import click
import numpy as np
import pandas as pd
import re

from typing import Dict, List, Union
from tqdm import tqdm
from torch.nn.functional import softmax
from opencc import OpenCC
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from rich.console import Console
from torch.utils.data import Dataset, DataLoader

NO_EXTERNAL = '外規編號'
NO_INTERNAL = '應匹配的內規編號'
EXTERNAL_COLNAME = '外規內容'
INTERNAL_COLNAME_1 = '應匹配的內規1內容'
INTERNAL_COLNAME_2 = '應匹配的內規2內容'
INTERNAL_COLNAME_3 = '應匹配的內規3內容'
INTERNAL_RULE = '內文'

console=Console(record=True)
cc = OpenCC('tw2s')

class InferenceCollateFunction():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def collate(self, list_of_data_dic: List[dict]):
        external_laws = [d['external'] for d in list_of_data_dic]
        internal_regulations = [d['internal']['text'] for d in list_of_data_dic]
        internal_codes = [d['internal']['code'] for d in list_of_data_dic]
        zipped = zip(external_laws, internal_regulations)
        query = [seq_to_seq(tup[0], tup[1]) for tup in zipped]
        batch = self.tokenizer(query, padding=True, truncation=True, return_tensors='pt')

        return batch, internal_codes

def seq_to_seq(external: str, internal:str):
    return "".join(['STSB sentence1: ', cc.convert(external), ' sentence2: ', cc.convert(internal)])

@click.command()
@click.option('--internal', '-i', type=str, default='../data/data0716/internal_dictionary.json')
@click.option('--label', '-l', type=str, default='../data/data0716/label-test.json')
@click.option('--start', '-s', type=int, default=0)
@click.option('--end', '-e', type=int, default=100)
@click.option('--batch_size', '-b', type=int, default=32)
@click.option('--model_name', '-m', type=str, default='finetuned_models/google_base_0810/model_files')
@click.option('--output_dir', '-o', type=str, default='model_generations/google_base_0828_generations/')
@click.option('--device', '-d', type=str, default='cuda:0')


def main(internal: str, label: str, start: int, end: int, batch_size: int, model_name: str, output_dir: str, device: str):

    
    with open(internal, 'r', encoding = 'utf-8') as i:
        internal = json.load(i) #raw_internal is a list of dictionaries
        i.close()
    with open(label, 'r', encoding = 'utf-8') as l:
        label = json.load(l) #raw_internal is a list of dictionaries
        l.close()
    
    label = label[start:end]
    internal_dictionary = {dic['text']: dic['code'] for dic in internal}
    tokenizer = MT5Tokenizer.from_pretrained(model_name, local_files_only=True)
    model = MT5ForConditionalGeneration.from_pretrained(model_name, local_files_only=True)
    model.to(device)
    model.eval()
    
    collate_fn = InferenceCollateFunction(tokenizer)
    NEGATIVE_LABEL = "_1"
    POSITIVE_LABEL = "_0"
    label_inds = tokenizer.convert_tokens_to_ids(
        [POSITIVE_LABEL, NEGATIVE_LABEL])
    
    count = start
    for data in tqdm(label):

        external_law = data[EXTERNAL_COLNAME]
        answer_1_index = internal_dictionary[data[INTERNAL_COLNAME_1]] if type(data[INTERNAL_COLNAME_1]) == str else -1
        answer_2_index = internal_dictionary[data[INTERNAL_COLNAME_2]] if type(data[INTERNAL_COLNAME_2]) == str else -1
        answer_3_index = internal_dictionary[data[INTERNAL_COLNAME_3]] if type(data[INTERNAL_COLNAME_3]) == str else -1
        
        dataset = InferenceDataset(external_law, internal)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn.collate)
        predictions = []
        for batch, internal_codes in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model.generate(**batch, output_scores=True, return_dict_in_generate=True, num_beams=1)
            scores = out.scores[0]
            scores = scores[:, label_inds]
            positive_ind = 0
            negative_ind = 1
            positive_scores = scores[:, positive_ind].cpu()
            negative_scores = scores[:, negative_ind].cpu()
            positive_probas = softmax(positive_scores, dim=0).tolist()
            negative_probas = softmax(negative_scores, dim=0).tolist()
            probas = [pos-neg for pos, neg in zip(positive_probas, negative_probas)]
            predictions += [i for i in zip(internal_codes, probas)]

        predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
        json_file = {'answer_1': answer_1_index, 'answer_2': answer_2_index, 'answer_3': answer_3_index, 'prediction': predictions}
        with open(output_dir + str(count) + '.json', 'w') as f:
            json.dump(json_file, f, indent=4)
        count += 1

    
if __name__ == '__main__':
    main()
