import json
import random 
import jieba
import click

from BM25_model import BM25_Model
from tqdm import tqdm
from rich.console import Console



@click.command()
@click.option('--internal', '-i', type=str, default='../data/data0716/old_internal.json')
@click.option('--label', '-l', type=str, default='../data/data0716/label-train.json')
@click.option('--output_dir', '-o', type=str, default='../data/data0716/finetune_data/')


def main(internal: str, label: str, output_dir: str):
    
    console = Console(record=True)

    with open(internal, 'r', encoding='utf-8') as f:
        old_internal = json.load(f)
        f.close()
    
    with open(label, 'r', encoding='utf-8') as f:
        label_train = json.load(f)
        f.close()
        
    label_train = label_train
    old_internal_list = [dic['內文'] for dic in old_internal]
    tokenized_old_internal = [list(jieba.cut(str(internal))) for internal in old_internal_list]
    bm25_model = BM25_Model(tokenized_old_internal)
    pos_sample = []
    neg_sample = []
    for dic in tqdm(label_train):
        
        tokenized_query = list(jieba.cut(dic['外規內容']))
        document_score = bm25_model.get_documents_score(tokenized_query)
        internal_scores = dict(zip(old_internal_list, document_score))
        top_negatives = sorted(internal_scores, key=internal_scores.get, reverse=False)[:500]
        temp_neg = []
        
        if dic["應匹配的內規1內容"] == dic["應匹配的內規1內容"]:
            pos_sample.append({'external': dic['外規內容'], 'internal': dic['應匹配的內規1內容'], 'label':'_0'})        
        else:
            temp_neg.append({'external': dic['外規內容'], 'internal': random.choice(top_negatives), 'label':'_1'})


        if dic["應匹配的內規2內容"] == dic["應匹配的內規2內容"]:
            pos_sample.append({'external': dic['外規內容'], 'internal': dic["應匹配的內規2內容"], 'label':'_0'})
        else:
            temp_neg.append({'external': dic['外規內容'], 'internal': random.choice(top_negatives), 'label':'_1'})
            
        if dic["應匹配的內規3內容"] == dic["應匹配的內規3內容"]:
            pos_sample.append({'external': dic['外規內容'], 'internal': dic["應匹配的內規3內容"], 'label':'_0'})
        else:
            temp_neg.append({'external': dic['外規內容'], 'internal': random.choice(top_negatives), 'label':'_1'})   
        neg_sample.append(random.choice(temp_neg)) if temp_neg else 0

    filtered_neg_sample = random.sample(neg_sample, int(len(pos_sample)*1.5))
    console.print(f"Size of Positive Sample: {len(pos_sample)}")
    console.print(f"Size of Raw Negative Sample: {len(neg_sample)}")
    console.print(f"Size of Negative Sample:{len(filtered_neg_sample)}")
    console.log(f"Size of Positive Sample: {len(pos_sample)}")
    console.log(f"Size of Raw Negative Sample: {len(neg_sample)}")
    console.log(f"Size of Negative Sample:{len(filtered_neg_sample)}")
    console.save_text(output_dir+'logs.txt')
    
    sample = pos_sample+filtered_neg_sample
    random.shuffle(sample)
    
    output_path = output_dir+'train.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample,f,ensure_ascii=False,indent=4)
        
    
if __name__ == '__main__':
    main()
