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

Firstly, call the BM25_generate_scores.py function, that will build the model with tokenized internal corpus and will pick out top 100 most similar label-internal pairs with their similarity scores calculated by the model. Secondly, the BM25_filter_and_index.py function does [min_threshold, max_threshold] X [min_k, max_k] model predictions filtering based on their similarity scores before converting predictions and answers to index for simplicity. Lastly, the accuracy, presicion, recall and f1-score of the individual threshold and top k filtered predictions can be calculated and saved to a csv table by the BM25_calculate_matrics.py function.  

## Model Generations
Sample:

![alt text](https://github.com/henry09027/BM25/blob/main/photo/model_predictions.png)

Sample final f1-score table:

![alt text](https://github.com/henry09027/BM25/blob/main/photo/result_table.png)

## Result

| Model         | f1@ threshold: 1 | f1@ threshold: 5 | f1@ threshold: 10| f1@ threshold: 20|
| ------------- |:----------------:|:----------------:|:----------------:|:----------------:|
| BM25          |    0.1994 (7.5)  |    0.3300 (5)    |    0.3940 (5)    |    0.4368 (5)    |

*f1 scores on the data0716/label-test.json dataset.
