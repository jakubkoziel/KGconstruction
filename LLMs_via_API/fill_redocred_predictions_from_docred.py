import json

from data_utlis import DocREDLoader, LLM_API_Response_Loader, PredictedNERLoader
from construct_datasets_from_responses import get_basic_ner_experiment_conducted, \
    get_refinement_ner_experiment_conducted
import os

"""This file assumes that you have already constructed a dataset from the responses of experiment you want to copy and that it was done for all documents."""


def _fill_responses(experiment):
    dr_loader = DocREDLoader('..')
    response_loader = LLM_API_Response_Loader()
    predicted_NER_loader = PredictedNERLoader(os.path.join('..', 'NERs'))

    # predicted = response_loader.docs_from_NER_predictions(experiment_type=experiment['experiment_type'],
    #                                                       dataset=experiment['dataset'], split=experiment['split'],
    #                                                       experiment=experiment['experiment'],
    #                                                       model=experiment['model'], docs_starting=docs,
    #                                                       narrow_docs_to=experiment['narrow_docs_to'])
    dataset_from_experiment = predicted_NER_loader.load_docs(docred_type=experiment['dataset'],
                                                             split=experiment['split'],
                                                             model_name=os.path.join(experiment['experiment'],
                                                                                     experiment['model']),
                                                             prediction_level=str(experiment['narrow_docs_to']))

    for split in ['dev', 'test']:
        redocred = dr_loader.load_docs(docred_type='redocred', split=split)
        matched = 0
        filled = 0
        for i in range(len(redocred)):
            if response_loader.check_if_document_done(experiment_type=experiment['experiment_type'],
                                                      dataset='redocred', split=split,
                                                      experiment=experiment['experiment'],
                                                      model=experiment['model'], doc_id=i):
                # print(f"Document {i} already done, skipping. - {experiment}")
                matched += 1
                continue

            for j in range(len(dataset_from_experiment)):
                if redocred[i]['title'] == dataset_from_experiment[j]['title'] and redocred[i]['sents'] == \
                        dataset_from_experiment[j]['sents']:
                    matched += 1
                    filled += 1
                    # print(f'Assigning response {j} as response {i}: redocred({i}) <- docred_dev({j})')

                    unpacked_entities = [e[0] for e in dataset_from_experiment[j]['vertexSet']]
                    imitate_model_response = '$$$' + json.dumps(unpacked_entities, ensure_ascii=False) + '$$$'
                    response_loader.save_response(experiment_type=experiment['experiment_type'], dataset='redocred',
                                                  split=split,
                                                  experiment=experiment['experiment'], model=experiment['model'],
                                                  document_id=i,
                                                  response=imitate_model_response)

        print(f"{split} - Matched: {matched}/{len(redocred)}; Filled: {filled}")


def run_response_replication():
    experiments = [
        {
            'experiment_type': 'ner',
            'dataset': 'docred',
            'split': 'dev',
            'experiment': 'v2',
            'model': 'deepseek-reasoner',
            'narrow_docs_to': None},
        {
            'experiment_type': 'ner',
            'dataset': 'docred',
            'split': 'dev',
            'experiment': 'v4',
            'model': 'deepseek-chat',
            'narrow_docs_to': None},
        {
            'experiment_type': 'ner',
            'dataset': 'docred',
            'split': 'dev',
            'experiment': 'v2_verifier',
            'model': 'deepseek-reasoner',
            'narrow_docs_to': None},
        {
            'experiment_type': 'ner',
            'dataset': 'docred',
            'split': 'dev',
            'experiment': 'v4_verifier',
            'model': 'deepseek-chat',
            'narrow_docs_to': None}
    ]

    are_done = get_basic_ner_experiment_conducted() + get_refinement_ner_experiment_conducted()

    for experiment in experiments:
        if experiment not in are_done:
            print(f"Experiment is not done yet as it doesn't have dataset constructed.: {experiment}")
            continue
        else:
            print(f"Base Experiment is done, filling responses for ReDocRED - {experiment}")
            _fill_responses(experiment)


if __name__ == '__main__':
    run_response_replication()
