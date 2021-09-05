# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 04:22:49 2021

@author: henry
"""

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

def process_json(internal, label, internal_target, label_target):
    cc = OpenCC('tw2s')
    #internal_list = []
    #internal_index_list = []
    #for i, dic in enumerate(internal):
        #internal_index_list.append(i)
        #internal_list.append(str(dic['內文']))
    #pair_dict = dict(zip(internal_list, internal_index_list))
    #pair_dict['NaN'], pair_dict['nan'] = -1, -1
    #hypothesises = [str(dictionary[internal_target]) for dictionary in internal]
    #premises = [str(dictionary[label_target]) for dictionary in label]
    premises = [str(dic[label_target]) for dic in label]
    hypothesises = [str(string) for string in internal]
    answer_1 = [dic['應匹配的內規1內容'] for dic in label]
    answer_2 = [dic['應匹配的內規2內容'] for dic in label]
    answer_3 = [dic['應匹配的內規3內容'] for dic in label]
    #answer_1 = [map(pair_dict, raw_answer_1)]
    #answer_2 = [map(pair_dict, raw_answer_2)]
    #answer_3 = [map(pair_dict, raw_answer_3)]

    queries = []
    for index, premise in enumerate(premises):
        for hypothesis in hypothesises:
             temp = [process_nli(premise=cc.convert(premise), hypothesis=cc.convert(hypothesis)), premise, hypothesis, answer_1[index], answer_2[index], answer_3[index]]
             queries.append(temp)
        
    return np.array(queries)

def process_nli(premise: str, hypothesis: str):
    """ process to required xnli format with task prefix """
    premise = fix_punc(premise)
    hypothesis = fix_punc(hypothesis)
    premise = premise[:195] if len(premise) > 195 else premise
    hypothesis = hypothesis[:195] if len(hypothesis) > 195 else hypothesis
    return "".join(['stsb sentence1: ', premise, ' sentence2: ', hypothesis])

def fix_punc(sentense):
    
    return re.sub(r"[\s+\.\!\/_,$%^*(+\"\']+|\[+——！，。？、~@#￥%……&*（）|─ |┼ |├ |┤ |│ │ ├ ─ □◎┤ ", "", sentense)

class YourDataSetClass(Dataset):

    def __init__(
        self, queries, tokenizer, source_len
    ):
        
        self.tokenizer = tokenizer
        self.queries = queries
        self.source_len = source_len


    def __len__(self):
        """returns the length of dataframe"""

        return len(self.queries)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
       
        model_query = self.queries[:,0][index]
        premise = self.queries[:,1][index]
        hypothesis = self.queries[:,2][index]        
        answer_1 = self.queries[:,3][index]
        answer_2 = self.queries[:,4][index]
        answer_3 = self.queries[:,5][index]
        return [model_query, premise, hypothesis, answer_1, answer_2,answer_3]
    
def collate_fn(data):
    """:data: a list for a batch of samples. [[string, tensor], ..., [string, tensor]]
    """
    transposed_data = list(zip(*data))
    query, premise, hypothesis, answer_1, answer_2, answer_3= transposed_data[0], transposed_data[1], transposed_data[2], transposed_data[3], transposed_data[4], transposed_data[5]
    return (query, premise, hypothesis, answer_1, answer_2, answer_3)

def get_score(internal, label, model_params):
    
    tokenizer = MT5Tokenizer.from_pretrained(model_params["MODEL"], local_files_only=True)
    model = MT5ForConditionalGeneration.from_pretrained(model_params["MODEL"], local_files_only=True)
    model.to(model_params["DEVICE"])
    model.eval()
    
    NEGATIVE_LABEL = "_1"
    POSITIVE_LABEL = "_0"
    label_inds = tokenizer.convert_tokens_to_ids(
        [POSITIVE_LABEL, NEGATIVE_LABEL])
    
    queries = process_json(internal=internal, label=label, internal_target='內文', label_target='外規內容')
    data_set = YourDataSetClass(queries, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"])
    loader_params ={
        'batch_size': model_params['BATCH_SIZE'],
        'shuffle': False,
        'num_workers': 0
        }
    data_loader = DataLoader(data_set, **loader_params, collate_fn=collate_fn)
    source_len = model_params["MAX_SOURCE_TEXT_LENGTH"]
    result = pd.DataFrame(columns=['label', 'internal', 'score', 'answer_1', 'answer_2','answer_3'])
    
    for _, data in enumerate(data_loader, 0):
        
        tokenized_query = tokenizer.batch_encode_plus(
            data[0], 
            max_length=source_len,
            pad_to_max_length=True,
            truncation=True,
            return_tensors="pt", 
            padding=True)
        inputs = tokenized_query.to(model_params["DEVICE"])
        hypothesis = np.array(data[2])
        premise = np.array(data[1])
        answer_1 = np.array(data[3])
        answer_2 = np.array(data[4])
        answer_3 = np.array(data[5])
        out = model.generate(**inputs, output_scores=True, return_dict_in_generate=True, num_beams=1)
        scores = out.scores[0]
        # cut down scores to our task labels
        scores = scores[:, label_inds]
        positive_ind = 0
        # this gives a zero-shot classification style output across labels
        current_token = tokenizer.convert_ids_to_tokens(label_inds[0])
        positive_scores = scores[:, positive_ind].cpu()
        positive_probas = np.array(softmax(positive_scores, dim=0))
        temp = pd.DataFrame({'label':premise,'internal':hypothesis,'current_token': current_token,'score': positive_probas,'answer_1':answer_1, 'answer_2':answer_2, 'answer_3':answer_3})
        result = pd.concat([result,temp])
        
    result["score"] = result["score"].astype(float)
    return result

class InferenceDataset(Dataset):
    def __init__(self, external_law: str, internal_regulations: List[dict]):
        self.external_law = external_law
        self.internal_regulations = internal_regulations

    def __len__(self):
        return len(self.internal_regulations)

    def __getitem__(self, i):
        data_dic = {'external': self.external_law, 'internal': self.internal_regulations[i]}
        return data_dic


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

def temp():

    label = label[start:end]
    model_1 = "finetuned_models/alan_turing_0808/"
    model_2 = "finetuned_models/google_base_0810/"
    
    output_dir_2 = 'model_generations/google_base_generations_0810/'
    output_dir_1 = 'model_generations/alan_turing_generations_0808/'
    model_params={
    "MODEL": model_2+'model_files/',
    "BATCH_SIZE": 32,          # training batch size
    "MAX_SOURCE_TEXT_LENGTH":400,  # max length of source text
    "SEED": 42,                 # set seed for reproducibility 
    "DEVICE": device
    }
    count = start
    for batch_label in tqdm(label):
        try:
            df = get_score(internal=internal, label=[batch_label], model_params=model_params)
            output_filename=output_dir_2+'google_base_'+str(count)+'_scores.csv'
            #output_json = output_dir+'alan_turing_'+str(start)+'_to_'+str(end)+'_scores.json'
            df.to_csv(output_filename, encoding='utf-8')
        except RuntimeError:
            console.log(f"No. {count} label-test CUDA out of memory occured")
        count+=1
    #result_json = df.to_json(orient='records')
    #with open(output_json, 'w', encoding='utf-8') as f:
    #    json.dump(result_json, f, ensure_ascii=False, indent=4)

    
if __name__ == '__main__':
    main()
