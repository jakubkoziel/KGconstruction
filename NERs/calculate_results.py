from utils_NER import evaluate_NER_F1_exact
from data_utlis import DocREDLoader, PredictedNERLoader
import pandas as pd
import os
from itertools import product

dr_loader = DocREDLoader('..')
ner_loader = PredictedNERLoader()

results_pretrained_models = pd.DataFrame(
    columns=['dataset', 'split', 'model', 'prediction_level', 'TP', 'FP', 'FN', 'F1', 'Precision', 'Recall'])

datasets = ['docred', 'redocred']
splits = ['dev', 'test']
prediction_levels = ['sentence', 'document']
models = ['en_core_web_sm', 'en_core_web_lg', 'en_core_web_trf']

for combination in product(datasets, splits, models, prediction_levels):
    docred_type, split, model_name, prediction_level = combination

    dataset_true = dr_loader.load_docs(docred_type=docred_type, split=split)
    dataset_predicted = ner_loader.load_docs(docred_type=docred_type, split=split, model_name=model_name,
                                             prediction_level=prediction_level)

    scores_list = evaluate_NER_F1_exact(dataset_true, dataset_predicted, mode='position')

    results_pretrained_models.loc[len(results_pretrained_models)] = combination + scores_list

results_pretrained_models.to_csv('results_pretrained_models.csv', index=False)
