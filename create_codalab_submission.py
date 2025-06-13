import os
from data_utlis import DocREDLoader, PredictedNERLoader
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


def single_experiment(docred_type, split, model_name, prediction_level):
    try:
        dr_loader = DocREDLoader()
        truth = dr_loader.load_docs(docred_type=docred_type, split=split)
        if model_name == 'original_NER':
            predicted_triplets = dr_loader.get_dreeam_RE_results(docred_type='docred', split='dev')
        else:
            ner_loader = PredictedNERLoader('NERs')
            predicted_docs = ner_loader.load_docs(docred_type=docred_type, split=split, model_name=model_name,
                                                  prediction_level=prediction_level)
            predicted_triplets = ner_loader.get_dreeam_RE_results(docred_type=docred_type, split=split,
                                                                  model_name=model_name,
                                                                  prediction_level=prediction_level)
            predicted_copy = predicted_triplets.copy()
            predicted_triplets = adjust_relations(predicted_triplets, truth, predicted_docs, scores=True)
            changed = 0
            unchanged = 0
            for r in range(len(predicted_triplets)):
                if predicted_triplets[r]['h_idx'] != predicted_copy[r]['h_idx'] or \
                        predicted_triplets[r]['t_idx'] != predicted_copy[r]['t_idx']:
                    changed += 1
                    print(f'Changed: {predicted_copy[r]["h_idx"]} -> {predicted_triplets[r]["h_idx"]}, ', end='; ')
                    print(f'Changed: {predicted_copy[r]["t_idx"]} -> {predicted_triplets[r]["t_idx"]}, ')
                else:

                    unchanged += 1
            print(f'Changed: {changed}, Unchanged: {unchanged}')

            with open(f'codalab_submission.json', 'w') as f:
                import json
                json.dump(predicted_triplets, f)

        return None

    except FileNotFoundError as e:

        print(f'Skipping, not ready; File not found: {e}')
        raise e


def get_dreeam_results():
    results = []

    for docred_type in ['docred']:
        for split in ['test']:

            for model in [ 'wikineural-multilingual-ner-fine-tuned',
                          ]:
                se = single_experiment(docred_type=docred_type, split=split, model_name=model,
                                       prediction_level='sentence')
                results.append(se)


if __name__ == '__main__':
    get_dreeam_results()
