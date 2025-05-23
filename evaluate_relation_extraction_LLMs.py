import os
from data_utlis import DocREDLoader, PredictedNERLoader, LLM_API_Response_Loader
from dreeam.evaluation import official_evaluate
import pandas as pd


def find_document_by_title(title, documents):
    for i in range(len(documents)):
        if documents[i]['title'] == title:
            return documents[i]
    raise Exception(f'Document with title {title} not found in documents')


def remap_entity_to_true(entity_id_from_prediction, doc_true, doc_evaluated):
    # print(doc_evaluated['vertexSet'][entity_id_start])
    entity_to_map = doc_evaluated['vertexSet'][entity_id_from_prediction][0]
    found_entity_index = -1
    for i in range(len(doc_true['vertexSet'])):
        for j in range(len(doc_true['vertexSet'][i])):
            # doc_true['vertexSet'][i][j]['name'] == entity_to_map['name'] and \
            if doc_true['vertexSet'][i][j]['pos'] == entity_to_map['pos'] and \
                    doc_true['vertexSet'][i][j]['sent_id'] == entity_to_map['sent_id']:
                found_entity_index = i
                break
        if found_entity_index != -1:
            break

    return found_entity_index


def adjust_relations(predicted_triplets, truth, predicted_docs, scores):
    updated_triplets = []
    for single_result in predicted_triplets:
        if 'h_idx' not in single_result:
            single_result['h_idx'] = single_result['h']
        if 't_idx' not in single_result:
            single_result['t_idx'] = single_result['t']
        doc_true = find_document_by_title(single_result['title'], truth)
        doc_predicted = find_document_by_title(single_result['title'], predicted_docs)
        new_h = remap_entity_to_true(single_result['h_idx'], doc_true, doc_predicted)
        new_t = remap_entity_to_true(single_result['t_idx'], doc_true, doc_predicted)
        if single_result['r'] == "Na":
            raise Exception('Na relation can be found in the results, adjust the code.')

        if scores:
            tmp = {
                'h_idx': new_h,
                't_idx': new_t,
                'r': single_result['r'] if new_t != -1 and new_h != -1 else 'P_unpredictable',
                'title': single_result['title'],
                'evidence': single_result['evidence'],
                'score': single_result['score'],
            }
        else:

            tmp = {
                'h_idx': new_h,
                't_idx': new_t,
                'r': single_result['r'] if new_t != -1 and new_h != -1 else 'P_unpredictable',
                'title': single_result['title'],
                'evidence': single_result['evidence']
            }
        updated_triplets.append(tmp)

    return updated_triplets


def narrow_truth_to_only_evaluated(truth, predicted_docs_subset):
    truth_subset = []
    for doc in predicted_docs_subset:
        doc_true = find_document_by_title(doc['title'], truth)
        truth_subset.append(doc_true)

    return truth_subset


def single_experiment(docred_type, split, ner_model_name, prediction_level, model, experiment, experiment_type='re_'):
    dr_loader = DocREDLoader()
    truth = dr_loader.load_docs(docred_type=docred_type, split=split)

    ner_loader = PredictedNERLoader('NERs')
    predicted_docs = ner_loader.load_docs(docred_type=docred_type, split=split, model_name=ner_model_name,
                                          prediction_level=str(None))

    response_loader = LLM_API_Response_Loader('LLMs_via_API')
    predicted_triplets = response_loader.triplets_predictions(docs_starting=predicted_docs,
                                                              experiment_type=experiment_type + ner_model_name,
                                                              dataset=docred_type, split=split,
                                                              experiment=experiment, model=model,
                                                              narrow_docs_to=prediction_level)
    predicted_triplets = adjust_relations(predicted_triplets, truth, predicted_docs, scores=False)

    truth = narrow_truth_to_only_evaluated(truth, predicted_docs[:prediction_level])
    eval_res = official_evaluate(predicted_triplets=predicted_triplets, truth=truth)
    df = pd.DataFrame(eval_res, columns=['precision', 'recall', 'f1-score'])
    df['metric_type'] = ['re', 'evi', 're_ignore_train_annotated (used officially)', 're_ignore_train']
    df['docred_type'] = docred_type
    df['split'] = split
    df['ner_model_name'] = ner_model_name
    df['prediction_level'] = prediction_level
    df['experiment_type'] = experiment_type + ner_model_name
    df['experiment'] = experiment
    df['model'] = model

    print(
        f'Done - {docred_type} {split} {ner_model_name} {prediction_level} {model} {experiment} {prediction_level} {experiment}')
    return df


def get_LLM_results():
    results = []
    prediction_level = 20

    for ner_model_name in ['entities_separately']:
        for docred_type in ['docred', 'redocred']:
            for split in ['dev', 'test']:
                if (docred_type == 'docred' and split == 'test') or (docred_type == 'redocred' and split == 'dev'):
                    continue
                # if docred_type != 'docred' or split != 'dev':
                #     continue

                for model in ['deepseek-chat', 'deepseek-reasoner', 'gpt-4o-mini']:  # 'deepseek-chat', #'gpt-4o-mini'
                    for experiment in ['v1', 'v2', 'v3', 'v4', 'v5', 'v6',
                                       'v7']:  # []: # 'v1', 'v2', 'v3', 'v4', 'v5', 'v6',
                        # se = single_experiment(docred_type=docred_type, split=split, model_name='original_NER',
                        #                        prediction_level='original_NER')
                        se = single_experiment(docred_type=docred_type, split=split,
                                               ner_model_name=ner_model_name, prediction_level=prediction_level,
                                               model=model, experiment=experiment)
                        results.append(se)

    results = pd.concat(results, ignore_index=True)
    results.to_csv('results_LLMs_API_sadfasdfa.csv', index=False)


def get_LLM_results_on_full_set():
    results = []
    prediction_level = None

    for ner_model_name in ['entities_separately']:
        for docred_type in ['docred', 'redocred']:  # 'docred',
            for split in ['dev', 'test']:
                if (docred_type == 'docred' and split == 'test') or (docred_type == 'redocred' and split == 'dev'):
                    continue
                # if docred_type != 'docred' or split != 'dev':
                #     continue

                for model in ['deepseek-chat', 'deepseek-reasoner']:
                    for experiment in ['v7']:
                        se = single_experiment(docred_type=docred_type, split=split,
                                               ner_model_name=ner_model_name, prediction_level=prediction_level,
                                               model=model, experiment=experiment, experiment_type='re_')
                        results.append(se)

                for model in ['deepseek-chat', 'deepseek-reasoner']:
                    for experiment in ['v7']:
                        se = single_experiment(docred_type=docred_type, split=split,
                                               ner_model_name=ner_model_name, prediction_level=prediction_level,
                                               model=model, experiment=experiment, experiment_type='verifier_re_')
                        results.append(se)

    results = pd.concat(results, ignore_index=True)
    results.to_csv('results_LLMs_API_v7_both_full_sets.csv', index=False)


if __name__ == '__main__':
    # get_LLM_results()
    get_LLM_results_on_full_set()
