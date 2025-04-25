import time
from data_utlis import DocREDLoader, PredictedNERLoader
from itertools import product
import sys


def predict(docred_type, split, model_name, prediction_level):
    dr_loader = DocREDLoader('..')
    ner_loader = PredictedNERLoader()

    if ner_loader.check_if_exists(docred_type=docred_type, split=split,
                                  prediction_level=prediction_level, model_name=model_name):
        print(f'Predictions already exist for {docred_type}, {split}, {model_name}, {prediction_level}')
        return

    dataset = dr_loader.load_docs(docred_type, split)

    print(f'Running for {docred_type}, {split}, {model_name}, {prediction_level}')
    start = time.time()

    if model_name in ('en_core_web_sm', 'en_core_web_lg', 'en_core_web_trf'):
        from spacy_predict_NER_pretrained_models import predict_spacy
        dataset_predicted = predict_spacy(dataset=dataset, model_name=model_name, prediction_level=prediction_level)
    elif model_name in (
            "xlm-roberta-large-finetuned-conll03-english", "Babelscape/wikineural-multilingual-ner",
            "dslim/bert-base-NER",
            "dslim/bert-large-NER",
            "dslim/bert-base-NER-uncased"):
        from transformers_predict_NER_pretrained_models import predict_transformers
        dataset_predicted = predict_transformers(dataset=dataset, model_name=model_name,
                                                 prediction_level=prediction_level)
    else:
        raise Exception('Such model not supported')

    stop = time.time()
    print(f'Predictions took {(stop - start) / 60} minutes.')

    ner_loader.save_docs(docs=dataset_predicted, docred_type=docred_type, split=split,
                         prediction_level=prediction_level, model_name=model_name)


datasets = ['docred', 'redocred']
splits = ['dev', 'test']
prediction_levels = ['sentence', 'document']

env_name = sys.executable
print(f"Environment path: {env_name}")
# KGconstruction_environments\spacy env expected
models = ['en_core_web_sm', 'en_core_web_lg', 'en_core_web_trf']

predict(docred_type='docred', split='dev', model_name="xlm-roberta-large-finetuned-conll03-english",
        prediction_level='sentence')
# for combination in product(datasets, splits, models, prediction_levels):
#     docred_type, split, model_name, prediction_level = combination
#
#     predict(docred_type=docred_type, split=split, model_name=model_name, prediction_level=prediction_level)
