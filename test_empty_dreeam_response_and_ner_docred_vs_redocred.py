from data_utlis import DocREDLoader, PredictedNERLoader

dr_loader = DocREDLoader()
ner_loader = PredictedNERLoader('NERs')

docred_dev = dr_loader.load_docs(docred_type='docred', split='dev')
dreeam_response_entities_separately = ner_loader.get_dreeam_RE_results(docred_type='docred', split='dev',
                                                                       model_name='entities_separately',
                                                                       prediction_level=str(None))
docs_missing_from_response = []
for i in range(len(docred_dev)):
    found = False
    for j in range(len(dreeam_response_entities_separately)):
        if docred_dev[i]['title'] == dreeam_response_entities_separately[j]['title']:
            found = True
            break
    if not found:
        docs_missing_from_response.append((i, docred_dev[i]['title']))
print(f"Missing documents from response: {len(docs_missing_from_response)}")
print(docs_missing_from_response)

redocred_test = dr_loader.load_docs(docred_type='redocred', split='test')

redocred_doc_not_found = []
for i in range(len(redocred_test)):
    found = False
    for j in range(len(docred_dev)):
        if redocred_test[i]['title'] == docred_dev[j]['title']:
            found = True
            break
    if not found:
        redocred_doc_not_found.append((i, redocred_test[i]['title']))
    else:
        docred_vset = set()
        redocred_vset = set()
        for eg in docred_dev[j]['vertexSet']:
            for e in eg:
                docred_vset.add((e['sent_id'], e['pos'][0], e['pos'][1]))
        for eg in redocred_test[i]['vertexSet']:
            for e in eg:
                redocred_vset.add((e['sent_id'], e['pos'][0], e['pos'][1]))

        if docred_vset != redocred_vset:
            print(f"Different vertex sets for {j, docred_dev[j]['title']}, {i, redocred_test[i]['title']}")
            print(f"DocRED: {docred_vset}")
            print(f"RedocRED: {redocred_vset}")
            raise Exception('Something wrong with the vertex sets?')

    print('Checked document: ', i, end=' ')
print('---')
print(redocred_doc_not_found)
