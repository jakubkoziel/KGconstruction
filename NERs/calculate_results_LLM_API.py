from utils_NER import evaluate_NER_F1_exact
from data_utlis import DocREDLoader, PredictedNERLoader
import pandas as pd
import os
from LLMs_via_API.construct_datasets_from_responses import get_basic_ner_experiment_conducted, \
    get_refinement_ner_experiment_conducted

dr_loader = DocREDLoader('..')
ner_loader = PredictedNERLoader()

results_LLMs_API = pd.DataFrame(
    columns=['dataset', 'split', 'model', 'prediction_level', 'TP', 'FP', 'FN', 'F1', 'Precision', 'Recall'])

basic_ner_experiments_conducted = get_basic_ner_experiment_conducted()
refinement_ner_experiments_conducted = get_refinement_ner_experiment_conducted()
ner_experiments_conducted = basic_ner_experiments_conducted + refinement_ner_experiments_conducted
for experiment in ner_experiments_conducted:
    dataset_true = dr_loader.load_docs(docred_type=experiment['dataset'], split=experiment['split'])
    dataset_predicted = ner_loader.load_docs(docred_type=experiment['dataset'],
                                             split=experiment['split'],
                                             model_name=os.path.join(experiment['experiment'], experiment['model']),
                                             prediction_level=str(experiment['narrow_docs_to']))
    if experiment['narrow_docs_to'] is not None:
        dataset_true = dataset_true[:experiment['narrow_docs_to']]
    scores_list = evaluate_NER_F1_exact(dataset_true, dataset_predicted, mode='position')

    results_LLMs_API.loc[len(results_LLMs_API)] = (experiment['dataset'], experiment['split'],
                                                   os.path.join(experiment['experiment'], experiment['model']),
                                                   str(experiment['narrow_docs_to'])) + scores_list

results_LLMs_API.to_csv('results_pretrained_LLMs_API.csv', index=False)
