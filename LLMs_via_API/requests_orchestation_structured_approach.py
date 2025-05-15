import json
from LLM_API_request_handler import RequestHandler
from credentials import credentials
import multiprocessing
import time
import NER_predefined_messages
import RE_predefined_messages
from data_utlis import DocREDLoader, LLM_API_Response_Loader, PredictedNERLoader
from data_utlis import PredictedNERLoader
import os
import re


def _ner_basic_messages(model, experiment, doc_sents):
    if experiment in ('v3', 'v4', 'v5'):
        system_msg = NER_predefined_messages.experiment_prompts['system_' + experiment](docred_type='redocred',
                                                                                        split='train')
    elif experiment in ('v1', 'v2'):
        system_msg = NER_predefined_messages.experiment_prompts['system_' + experiment]
    else:
        raise Exception('Experiment not defined')

    system_role = 'system' if model != 'o1-mini' else 'user'

    messages = [
        {"role": system_role, "content": system_msg},
        {"role": "user", "content": f"Text to analyze: {json.dumps(doc_sents, ensure_ascii=False)}"}
    ]

    return messages


def _ner_refine_messages(model, experiment_type, dataset, split, experiment, doc_sents, doc_title, document_id):
    experiment_base = experiment.split('_refined_')[0]
    experiment_refine = experiment.split('_refined_')[1]
    messages_base = _ner_basic_messages(model=model, experiment=experiment_base,
                                        doc_sents=doc_sents)
    # Load previous response
    llm_api_loader = LLM_API_Response_Loader()
    previous_response = llm_api_loader.read_single_response(experiment_type=experiment_type, dataset=dataset,
                                                            split=split, experiment=experiment_base,
                                                            model=model, document_id=document_id)

    # Load fine-tuned model response
    predicted_ner_loader = PredictedNERLoader(os.path.join('..', 'NERs'))
    other_model_predictions = predicted_ner_loader.load_docs(docred_type=dataset, split=split,
                                                             model_name='wikineural-multilingual-ner-fine-tuned',
                                                             prediction_level='sentence')
    compare_response = other_model_predictions[document_id]
    if compare_response['title'] != doc_title:
        for candidate in other_model_predictions:
            if candidate['title'] == doc_title:
                compare_response = candidate
                break
    if compare_response['title'] != doc_title:
        raise Exception('Title not found in other model predictions')
    compare_response = [entity[0] for entity in compare_response['vertexSet']]

    messages_continuation = [
        {"role": "assistant", "content": '$$$' + json.dumps(previous_response, ensure_ascii=False) + '$$$'},
        {'role': 'user',
         'content': NER_predefined_messages.experiment_prompts['refine_' + experiment_refine](
             other_prediction=json.dumps(compare_response, ensure_ascii=False))
         }]
    return messages_base + messages_continuation


def _ner_verifier_messages(model, dataset, split, experiment, doc_sents, doc_title, document_id):
    experiment_base = experiment.split('_')[0]
    messages_base = _ner_basic_messages(model=model, experiment=experiment_base,
                                        doc_sents=doc_sents)
    # Load fine-tuned model response
    predicted_ner_loader = PredictedNERLoader(os.path.join('..', 'NERs'))
    other_model_predictions = predicted_ner_loader.load_docs(docred_type=dataset, split=split,
                                                             model_name='wikineural-multilingual-ner-fine-tuned',
                                                             prediction_level='sentence')
    compare_response = other_model_predictions[document_id]
    if compare_response['title'] != doc_title:
        for candidate in other_model_predictions:
            if candidate['title'] == doc_title:
                compare_response = candidate
                break
    if compare_response['title'] != doc_title:
        raise Exception('Title not found in other model predictions')
    compare_response = [entity[0] for entity in compare_response['vertexSet']]

    messages_continuation = [
        {"role": "assistant", "content": '$$$' + json.dumps(compare_response, ensure_ascii=False) + '$$$'},
        {'role': 'user',
         'content': NER_predefined_messages.experiment_prompts['verifier']()
         }]
    return messages_base + messages_continuation


def _re_basic_messages(model, experiment, doc_sents, doc_vertexSet):
    system_msg = RE_predefined_messages.experiment_prompts['system_' + experiment]

    system_role = 'system' if model != 'o1-mini' else 'user'

    for i in range(len(doc_vertexSet)):
        entity = doc_vertexSet[i][0]
        doc_sents[entity['sent_id']][entity['pos'][0]] = '<<' + str(i) + '>>' + \
                                                         doc_sents[entity['sent_id']][entity['pos'][0]]
        doc_sents[entity['sent_id']][entity['pos'][1] - 1] = doc_sents[entity['sent_id']][
                                                                 entity['pos'][1] - 1] + '<<' + str(i) + '>>'

    concatenated_sents = []
    for i in range(len(doc_sents)):
        concatenated_sents.append(' '.join(doc_sents[i]))
    concatenated_sents = ' '.join(concatenated_sents)

    messages = [
        {"role": system_role, "content": system_msg},
        {"role": "user", "content": f"Text to analyze: {json.dumps(concatenated_sents, ensure_ascii=False)}"}
    ]

    return messages


def construct_messages(model, experiment_type, dataset, split, experiment, doc_sents, doc_title, document_id,
                       doc_vertexSet=None):
    if experiment_type == 'ner':
        if 'verifier' in experiment:
            return _ner_verifier_messages(model=model, dataset=dataset, split=split,
                                          experiment=experiment,
                                          doc_sents=doc_sents, doc_title=doc_title, document_id=document_id)
        elif '_refined_' not in experiment:
            return _ner_basic_messages(model=model, experiment=experiment,
                                       doc_sents=doc_sents)

        else:
            return _ner_refine_messages(model=model, experiment_type=experiment_type, dataset=dataset,
                                        split=split, experiment=experiment,
                                        doc_sents=doc_sents, doc_title=doc_title, document_id=document_id)

    elif experiment_type.startswith('re'):
        return _re_basic_messages(experiment=experiment, model=model,
                                  doc_sents=doc_sents, doc_vertexSet=doc_vertexSet)


def revert_relation_mapping(response):
    with open('../data/DocRED/rel_info.json', 'r', encoding='utf-8') as file:
        relations = json.load(file)
    relations = {v: k for k, v in relations.items()}

    response_remapped = []
    for rel in response:
        if rel["r"] in relations:
            new_rel = rel.copy()
            new_rel["r"] = relations[rel["r"]]
            response_remapped.append(new_rel)
        else:
            print(f"WARNING: Relation {rel['r']} not found in relation_dict")
    return response_remapped


def single_process_for_requests(process_id, model, docs, document_subset, experiment_type, dataset, split,
                                experiment):
    if model == "gpt-4o-mini" or model == 'o1-mini':
        request_handler = RequestHandler(api_key=credentials['OpenAI']['api_key'])
    elif model == "deepseek-chat" or model == "deepseek-reasoner":
        request_handler = RequestHandler(api_key=credentials['deepseek']['api_key'],
                                         base_url=credentials['deepseek']['base_url'])
    else:
        raise Exception('Asking this model not assumed')

    error_for_simple_ner = []
    response_loader = LLM_API_Response_Loader()

    for i in document_subset:
        if response_loader.check_if_document_done(experiment_type=experiment_type, dataset=dataset, split=split,
                                                  experiment=experiment, model=model,
                                                  doc_id=i):
            print(f'Document {i} already done. Skipping this.')
            continue

        print(f'Process {process_id} starting to analyze document {i}')
        messages = construct_messages(model=model, experiment_type=experiment_type, dataset=dataset, split=split,
                                      experiment=experiment,
                                      doc_sents=docs[i]['sents'], doc_title=docs[i]['title'], document_id=i,
                                      doc_vertexSet=docs[i]['vertexSet'])
        # if model == "deepseek-chat":
        #     print(messages)
        try:
            response = request_handler.send_request(model=model, messages=messages, process_id=process_id)

            if experiment_type.startswith('re') and (
                    experiment == 'v4' or experiment == 'v3' or experiment == 'v6' or experiment == 'v7'):
                # print(response)
                response = json.loads(re.search(r'\$\$\$(.*)\$\$\$', response, re.DOTALL).group(1))
                response = revert_relation_mapping(response)
                response = f'$$${json.dumps(response, ensure_ascii=False)}$$$'
                # print(response)

            response_loader.save_response(experiment_type=experiment_type, dataset=dataset, split=split,
                                          experiment=experiment, model=model, document_id=i,
                                          response=response)

        except Exception as e:
            error_for_simple_ner.append(str(i))
            print('simple error:', e, i)

    response_loader.save_document_ids_with_errors(process_id=process_id, experiment_type=experiment_type,
                                                  dataset=dataset, split=split,
                                                  experiment=experiment, model=model, data_to_save=error_for_simple_ner)


def main_NER():
    # Settings
    experiment_type = 'ner'
    dataset = 'redocred'  # 'docred'
    split = 'dev'
    models = ['deepseek-reasoner']  # ['deepseek-reasoner']  #   # , 'gpt-4o-mini', 'deepseek-reasoner']
    experiments = ['v2']  # ['v4_refined_v1', 'v4_refined_v2']  # [v1, v2, ...]
    num_processes = 10
    dr_loader = DocREDLoader('..')
    docs = dr_loader.load_docs(docred_type=dataset, split=split)
    number_of_docs = len(docs)

    # Logic
    timestamp = int(time.time())
    processes = []

    processes_subsets = [[] for _ in range(num_processes)]
    for i in range(number_of_docs):
        processes_subsets[i % num_processes].append(i)

    for experiment in experiments:
        for model in models:
            for i in range(
                    num_processes):
                process = multiprocessing.Process(target=single_process_for_requests,
                                                  args=(
                                                      i, model, docs,
                                                      processes_subsets[i],
                                                      experiment_type, dataset, split, experiment))
                processes.append(process)
                process.start()
                time.sleep(2)

            for process in processes:
                process.join()

            print('Processes have finished', experiment, model, (int(time.time()) - timestamp) / 60, 'minutes')


def main():
    # Settings
    ner_model_name = 'entities_separately'
    experiment_type = 're_' + ner_model_name
    dataset = 'redocred'  # 'docred'
    split = 'test'
    models = ['deepseek-chat', 'gpt-4o-mini',
              'deepseek-reasoner']  # ['deepseek-reasoner']  #   # , 'gpt-4o-mini', 'deepseek-reasoner']
    experiments = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6',
                   'v7']  # ['v1', 'v2', 'v3', 'v4', 'v5']  # ['v4_refined_v1', 'v4_refined_v2']  # [v1, v2, ...]
    num_processes = 10  # 10
    dr_loader = PredictedNERLoader('../NERs')

    docs = dr_loader.load_docs(docred_type=dataset, split=split, model_name=ner_model_name,
                               prediction_level=str(None))
    # docs = dr_loader.load_docs(docred_type=dataset, split=split, model_name='wikineural-multilingual-ner-fine-tuned',
    #                            prediction_level='sentence')
    number_of_docs = 20  # len(docs)

    # Logic
    timestamp = int(time.time())
    processes = []

    processes_subsets = [[] for _ in range(num_processes)]
    for i in range(number_of_docs):
        processes_subsets[i % num_processes].append(i)

    for experiment in experiments:
        for model in models:
            for i in range(
                    num_processes):
                process = multiprocessing.Process(target=single_process_for_requests,
                                                  args=(
                                                      i, model, docs,
                                                      processes_subsets[i],
                                                      experiment_type, dataset, split, experiment))
                processes.append(process)
                process.start()
                time.sleep(2)

            for process in processes:
                process.join()

            print('Processes have finished', experiment, model, (int(time.time()) - timestamp) / 60, 'minutes')


if __name__ == '__main__':
    start = time.time()
    main()
    stop = time.time()
    print(f'Execution took {((stop - start) / 60):.2f} minutes.')
