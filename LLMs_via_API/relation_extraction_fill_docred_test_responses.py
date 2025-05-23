from data_utlis import DocREDLoader, LLM_API_Response_Loader
import json

redocred_test = DocREDLoader('..').load_docs(docred_type='redocred', split='test')
docred_dev = DocREDLoader('..').load_docs(docred_type='docred', split='dev')

response_loader = LLM_API_Response_Loader()

experiment_type = 'verifier_re_entities_separately'  # 're_entities_separately'
models = ['deepseek-reasoner', 'deepseek-chat']
experiment = 'v7'

for model in models:
    for i in range(len(redocred_test)):
        if response_loader.check_if_document_done(experiment_type=experiment_type, dataset='redocred', split='test',
                                                  experiment=experiment, model=model, doc_id=i):
            print(f"Document {i} already done, skipping. - {experiment}")
            continue
        found = False
        for j in range(len(docred_dev)):
            if redocred_test[i]['title'] == docred_dev[j]['title']:
                found = True
                break

        if not found:
            raise Exception(f"Document {i} not found in docred_dev: {redocred_test[i]['title']}")

        response = response_loader.read_single_response(experiment_type=experiment_type, dataset='docred', split='dev',
                                                        experiment=experiment, model=model, document_id=j)
        response_loader.save_response(experiment_type=experiment_type, dataset='redocred', split='test',
                                      experiment=experiment, model=model, document_id=i,
                                      response=f'$$${json.dumps(response, ensure_ascii=False)}$$$')
        print(
            f"Document {i} filled with response from docred_dev {j} - {experiment}, docred title: {docred_dev[j]['title']},"
            f" redocred title: {redocred_test[i]['title']}")
