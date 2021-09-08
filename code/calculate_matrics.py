import os
import click
import copy
import json
import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Union, Dict, List

def filter_predictions(prediction: List[Dict[str, Union[Dict[int, int], int]]], k: int, threshold: float) -> List[Dict[str, Union[List[int], int]]]:
    for dic in prediction:
       # filter threshold
        for key_ in dic['prediction'].copy():
            if float(dic['prediction'][key_])<float(threshold):
                del dic['prediction'][key_]
        #filter top k -> list of keys(internal)
        dic['prediction'] = sorted(dic['prediction'], key=dic['prediction'].get,reverse=True)[:int(k)]
    return prediction

def calculate_metrics(outcome: List[Dict[str, Union[List[int], int]]]) -> float:
    tp, fn, tn, fp = 0, 0, 0, 0
    for row in outcome:
        ans_1 = row[ANS_1] # int > -1 if there is true answer, -1 if there is no answer
        ans_2 = row[ANS_2]
        ans_3 = row[ANS_3]
        predicts = row[PRED] # List[int] if there topK prediction has sth > threshold, [] if no prediction
        if predicts: #there is prediction
            if ans_1 != -1 or ans_2 != -1 or ans_3 != -1: #there is at least one answer
                if ans_1 in predicts or ans_2 in predicts or ans_3 in predicts:
                    tp = tp+1 #prediction hits either of the answer
                else:
                    fp = fp+1 #none of the answers were predicted
            else:
                fp = fp+1 # no ans, but there is predictions, should be FP
        else:
            if ans_1 == -1 and ans_2 == -1 and ans_3 == -1:
                tn = tn + 1 # no answer no predictions
            elif ans_1 != -1 or ans_2 != -1 or ans_3 != -1:
                fn = fn + 1 # there are one or more answers but no prediction
    assert len(outcome) == tp + fn + tn + fp
    accuracy  = (tp + tn) / (tp + fn + tn + fp)
    print(tp, fn, tn, fp)
    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall = 0
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = 0
    try:
        f1_score  = 2 * recall * precision / (recall + precision)
    except ZeroDivisionError:
        f1_score = 0

    accuracy, recall, precision, f1_score = map(lambda x: round(x, 4), [accuracy, recall, precision, f1_score])
    return accuracy, recall, precision, f1_score

ANS_1  = 'answer_1'
ANS_2  = 'answer_2'
ANS_3  = 'answer_3'
PRED = 'prediction'

@click.command()
@click.option('--prediction_directory', '-p', type=str, default='model_generations/alan_turing_generations/')


def main(prediction_directory: str):
    
    files = os.listdir(prediction_directory)
    files_ = []
    for file_name in files:
        if not file_name.startswith('.'):
            files_.append(file_name)
    predictions = []
    for filename in tqdm(files_):
        try:
            with open(prediction_directory+filename, 'r') as f:
                temp=json.load(f)
                f.close()
            predictions.append(temp)
        except UnicodeDecodeError:
            0
    for dic in predictions:
        temp = {}
        for prediction in dic['prediction']:
            temp.update({prediction[0]: prediction[1]})
        dic['prediction'] = temp
    threshold_range = np.linspace(0, 0.6, 12)
    k_range = [1, 5, 10, 20]
    result = []
    for threshold in tqdm(threshold_range):
        for k in k_range:
            temp_prediction = copy.deepcopy(predictions)
            filtered_predictions = filter_predictions(prediction=temp_prediction, k=k, threshold=threshold)
            accuracy, recall, precision, f1_score = calculate_metrics(filtered_predictions)
            result.append({
                'threshold': threshold,
                'k': k,
                'accuracy': accuracy,
                'recall': recall,
                'precision': precision,
                'f1_score': f1_score
            })
    output_filename = prediction_directory+'f1_score.csv'
    result_df=pd.DataFrame(result)
    result_df.to_csv(output_filename, encoding='utf-8')

if __name__ == '__main__':
    main()
