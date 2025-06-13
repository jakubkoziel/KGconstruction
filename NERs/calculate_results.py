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

models_spacy = ['en_core_web_sm', 'en_core_web_lg', 'en_core_web_trf']
models_transformers = ["xlm-roberta-large-finetuned-conll03-english", "Babelscape/wikineural-multilingual-ner",
                       "dslim/bert-base-NER",
                       "dslim/bert-large-NER",
                       "dslim/bert-base-NER-uncased"]
models_flair_gliner = ["flair", "urchade/gliner_large-v2.1"]
models_fine_tuned = ['wikineural-multilingual-ner-fine-tuned']
models = models_flair_gliner + models_transformers + models_spacy + models_fine_tuned

for combination in product(datasets, splits, models, prediction_levels):
    docred_type, split, model_name, prediction_level = combination

    dataset_true = dr_loader.load_docs(docred_type=docred_type, split=split)
    if not ner_loader.check_if_exists(docred_type=docred_type, split=split, model_name=model_name,
                                      prediction_level=prediction_level):
        print(f"Predictions for {docred_type} - {split} - {model_name} - {prediction_level} do not exist.")
        continue
    dataset_predicted = ner_loader.load_docs(docred_type=docred_type, split=split, model_name=model_name,
                                             prediction_level=prediction_level)

    scores_list = evaluate_NER_F1_exact(dataset_true, dataset_predicted, mode='position')

    results_pretrained_models.loc[len(results_pretrained_models)] = combination + scores_list

results_pretrained_models.to_csv('results_pretrained_models.csv', index=False)
