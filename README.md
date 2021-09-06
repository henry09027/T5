# T5

## Introduction

BM25 is a ranking function used by search engines to estimate the relevance of documents to a given search query. This README provides an overview of how the BM25 model is used to complete a semantic textual similarity(STS) task on Chinese-based law corupus.

## Prerequiste
* tensorflow pip installed
* ckiptagger pip installed
* ckiptagger model saved under current directory in a folder named "data/" 
  * Model data can be downloaded and extracted to the desired path by one of the included API.
```
# Downloads to ./data.zip (2GB) and extracts to ./data/
# data_utils.download_data_url("./") # iis-ckip
data_utils.download_data_gdown("./") # gdrive-ckip
```

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
![alt text](https://github.com/henry09027/BM25/blob/main/photo/workflow_pic.png)

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
