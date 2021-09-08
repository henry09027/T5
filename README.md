# T5

## Introduction

T5: Text-to-Text-Transfer-Transformer model proposes reframing all NLP tasks into a unified text-to-text-format where the input and output are always text strings. This formatting makes one T5 model fit for multiple tasks. As can be seen in the featured animation that it takes in text input from left for various NLP tasks and outputs the text for that respective task. The T5 model, pre-trained on C4, achieves state-of-the-art results on many NLP benchmarks while being flexible enough to be fine-tuned to a variety of important downstream tasks. As out task is semantic textual similarity on traditional Chinese dataset, we used the mT5 models, which are the mutiligual version of the T5 model coving 101 languages.

## Prerequiste
* tensorflow pip installed
* hugginface transformers pip installed
* OpenCC pip installed

## DataSet (List of dictionaries)

* Label Set \
External queries each with 0-3 internal pairs. \
Sample:
```
{
    "外規內容": "第b6至b8段提供評估資產對企業是否有其他用途之指引。",
    "應匹配的內規1內容": NaN,
    "應匹配的內規2內容": NaN,
    "應匹配的內規3內容": NaN
},
{
    "外規內容": "檢舉案件之受理及調查過程,有利益衝突之人,應予迴避。",
    "應匹配的內規1內容": "指派檢舉受理人員或專責單位。",
    "應匹配的內規2內容": NaN,
    "應匹配的內規3內容": "檢舉案件受理、處理過程、處理結果及相關文件製作之紀錄與保存。"
},
{
    "外規內容": "分離帳戶保險商品費用:係凡符合國際財務報導準則第四號保險合約定義之分離帳戶保險商品之各項費用總和皆屬之。",
    "應匹配的內規1內容": "分離帳戶保險商品費用。",
    "應匹配的內規2內容": NaN,
    "應匹配的內規3內容": NaN
}
```
* Internal Set \
A list of internal corpus. \
Sample:
```
{
    "法規名稱": "1050323融資循環_捷智_V1@20170410 (1)",
    "內文": "股東權益"
},
{
    "法規名稱": "1050323融資循環_捷智_V1@20170410 (1-1)",
    "內文": "目的："
}
```
## Model Workflow
![alt text](https://github.com/henry09027/T5/blob/main/photo/Screen%20Shot%202021-09-07%20at%201.56.47%20PM.png)

## Finetune

The finetune stage is a supervised classification task with text-to-text query. 90 percent of the Label Set were translated from traditional to simplified Chinese with OpenCC, and made text-to-text format e.g. (STSB sentence1: 金融资产之转移 sentence2: 金融资产之转移). The sentence pairs with entailment relationships were labeled "_0" and the contradict ones labeled "_1". Note that the 外規內容 without a 應匹配的內規內容 will be paired with one of the BM25 calculated least similar sentence for the 內規 library before labeled "_1"

## Inference

After the mT5 model being finetuned with our classification dataset, we test the model performance on the testing set (10 percent of the Label Set). Each 外歸內容 will be paired with each and every 內規 in the 內規 library before forming text-to-text format queries. Note that outputs of text-to-text model such as our mT5 model are tokens that have the highest softmax scores in each calculations. As we need the similarity scores of each query for document ranking, we apply the softmax function onto the score of the "_0" and "_1" token. We defined the similarity scores as the "_0" probabilities deducted by the "_1" probabilities. With these similarity scores, we can apply threshold and top k filters and calculate the accuracy and f1-scores.

## Method

Firstly, the BM25_model.py and the generate_train_data.py were for generating train.json data for finetune from 90 percent of the Label Set. Therefored, need not to be called here as I already put the trained.json in ../data/data0716/my_finetune_data. 

The remaining files were pretty straightforward. mt5_finetuning.py should be called first to finetuning our pretrained models. The finetuned models will be stored under the directory ./finetuned_models/. Then call mt5_generate_score.py that tests the label-test.json data on the finetuned model. The result will be stored under the directory ./model_generations/ as the format shown in the model generations section below. Finally, calculate_matrics.py will filter threshold and top-k on all the predictions from ./model_generations/ and output the final f1-score table.  

## Model Generations
Model generation sample (all 內規 and answer pairs have been converted to indecies according to the internal dictionary to simplify the f1 calculation process):

![alt text](https://github.com/henry09027/T5/blob/main/photo/Screen%20Shot%202021-09-08%20at%209.58.12%20PM.png)

## Results

![alt text](https://github.com/henry09027/T5/blob/main/photo/image.png)

## Version Control

A series of debugging was conducted aiming to improve to peculiar low f1-score on our mT5 models. The earliest adjustments happened to the f1-score calculation matrics, as a more accurate Label Dataset with one to three paired 內規 answers as supposed to only one paired 內規 answer was released. The following was purely to varify whether the bad performance only happened on training data. It turned out that the f1-score was also only around 0.04 on training data. This proved that it the model wasn't overfitting the training data. The next two attempts were increasing the training epoch in the risk of model overfitting on training data happening. As shown in the results table, this act pushed the classification accuracy to 0.998 for the alan-turing model at validation stage. However, the f1-scores of the models remained undesireable. The next attempt was to avoid imbalanced training data. We have more training sentence pairs with contradicted relationship than the ones with entailed relationship. I got rid of some of the contradicted sentence pairs to make the contradiction-entailment ratio 1:1. On top of this, the negative samples for the training 外規內容 without paired answers was changed from randomly picked from the internal library to one of the BM25 calculated least 100  similar 內規. The final version I tried had a f1-score of 0.05 which is still below our expectations.

## Last Words

Our finetune work got the best result on the mT5-large model trained by the alan-turing institute (https://huggingface.co/alan-turing-institute/mt5-large-finetuned-mnli-xtreme-xnli). Therefore, our primary focus will be on this pretrained model. With that being said, the greatest f1-score we were able to reach was around 0.05 at k being 20, compared to 0.43 of the BM25 model. The undesirable low f1-score seemed to be only happening on the mT5 model but not the T5 model. Similar work was shown in the “Document Ranking with a Pretrained Sequence-to-Sequence Model” paper (https://aclanthology.org/2020.findings-emnlp.63.pdf), in which the T5 model showed better performance compared to a BM25 model and BM25+BERT-Large model (BM25 bag of word retrieval followed by a BERT reranker) on document ranking tasks.
