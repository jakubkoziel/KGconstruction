import os
from dreeam.run import main
from data_utlis import DocREDLoader, PredictedNERLoader


def separate_entities():
    dr_loader = DocREDLoader('..')
    ner_loader = PredictedNERLoader('../NERs')
    for split in ['dev', 'test']:
        for docred_type in ['docred', 'redocred']:
            if docred_type == 'docred' and split == 'test':
                continue
            dev_true = dr_loader.load_docs(docred_type=docred_type, split=split)
            dev_true_separately = []
            for i in range(len(dev_true)):
                doc = {}
                for k in dev_true[i].keys():
                    if k == 'vertexSet':
                        vertexSet = []
                        for eg in dev_true[i][k]:
                            for e in eg:
                                vertexSet.append([e])
                        doc[k] = vertexSet
                    else:
                        doc[k] = dev_true[i][k]
                dev_true_separately.append(doc)
            ner_loader.save_docs(docs=dev_true_separately, docred_type=docred_type, split=split,
                                 model_name='entities_separately',
                                 prediction_level=str(None))


def reject_faulty_entities():
    ner_loader = PredictedNERLoader('../NERs')

    for docred_type in ['docred', 'redocred']:
        for split in ['dev', 'test']:
            if docred_type == 'docred' and split == 'test':
                continue
            for model in [os.path.join('v2', 'deepseek-reasoner'), os.path.join('v2_verifier', 'deepseek-reasoner'),
                          os.path.join('v4', 'deepseek-chat'), os.path.join('v4_verifier', 'deepseek-chat')]:
                set = ner_loader.load_docs(docred_type=docred_type, split=split, model_name=model,
                                           prediction_level=str(None))

                new_set = []
                bad_entities = 0
                for i in range(len(set)):
                    doc = {}
                    for k in set[i].keys():
                        if k == 'vertexSet':
                            vertexSet = []
                            for eg in set[i][k]:
                                for e in eg:
                                    try:
                                        name = set[i]['sents'][e['sent_id']][e['pos'][0]: e['pos'][1]]
                                        name_start = set[i]['sents'][e['sent_id']][e['pos'][0]]
                                        name_end = set[i]['sents'][e['sent_id']][e['pos'][1] - 1]
                                        vertexSet.append([e])
                                    except Exception as e:
                                        bad_entities += 1
                                        print(f'{docred_type} {split} {model} No. of bad entities:', bad_entities)
                            doc[k] = vertexSet
                        else:
                            doc[k] = set[i][k]
                    new_set.append(doc)

                ner_loader.save_docs(new_set, docred_type=docred_type, split=split, model_name=model + '-repaired',
                                     prediction_level=str(None))


def define_selected_experiments(prefix_dr_loader='..', prefix_ner_loader=f'..{os.path.sep}NERs'):
    dr_loader = DocREDLoader(prefix_dr_loader)
    ner_loader = PredictedNERLoader(prefix_ner_loader)

    experiments = []
    for docred_type in ['docred', 'redocred']:
        for split in ['dev', 'test']:
            if docred_type == 'docred' and split == 'test':
                continue
            experiments.append(dr_loader.return_path_to_read_from(docred_type=docred_type, split=split))

            for model in ["xlm-roberta-large-finetuned-conll03-english", 'wikineural-multilingual-ner-fine-tuned',
                          ]:
                experiments.append(
                    ner_loader.return_path_to_read_from(docred_type=docred_type, split=split, model_name=model,
                                                        prediction_level='sentence'))
            for model in [os.path.join('v2', 'deepseek-reasoner-repaired'),
                          os.path.join('v2_verifier', 'deepseek-reasoner-repaired'),
                          os.path.join('v4', 'deepseek-chat-repaired'),
                          os.path.join('v4_verifier', 'deepseek-chat-repaired'),
                          'entities_separately']:
                experiments.append(
                    ner_loader.return_path_to_read_from(docred_type=docred_type, split=split, model_name=model,
                                                        prediction_level=str(None)))

    return experiments


if __name__ == '__main__':
    # TODO separate_entities()
    # TODO reject_faulty_entities()

    for e in define_selected_experiments():
        print('\n\n\n', e)

        result_path = e.strip('.').strip(os.path.sep).strip('.json')
        if not os.path.exists(os.path.join('official_model_checkpoints', 'roberta_student', result_path)):
            print(
                f"{os.path.join('official_model_checkpoints', 'roberta_student', result_path)} does not exist; making dir")
            os.makedirs(os.path.join('official_model_checkpoints', 'roberta_student', result_path))
        if not os.path.exists(
                os.path.join('official_model_checkpoints', 'roberta_student', result_path, "results.json")):

            main([f'--data_dir={f"{os.path.sep}".join(e.split(os.path.sep)[:-1])}', f'--transformer_type=roberta',
                  f'--model_name_or_path=roberta-large',
                  f'--load_path=official_model_checkpoints/roberta_student', f'--eval_mode=single',
                  f'--test_file={e.split(os.path.sep)[-1]}', f'--test_batch_size=4', f'--num_labels=4',
                  f'--evi_thresh=0.2',
                  f'--num_class=97', f'--pred_file={os.path.join(result_path, "results.json")}'])
        else:
            print(e, 'Already exists, skipping...')
