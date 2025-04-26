from data_utlis import DocREDLoader
import json
import os


def transform_docs_info_transformers_tagged_format(dataset):
    dataset_processed = []
    entities_problematic = 0
    no_problem = 0

    for doc in dataset:
        ner_tags = []
        for s in doc['sents']:
            ner_tags.append(['O'] * len(s))

        for entity_group in doc['vertexSet']:
            for entity in entity_group:
                is_problematic = False

                for i in range(entity['pos'][0], entity['pos'][1]):
                    if ner_tags[entity['sent_id']][i] != 'O':
                        is_problematic = True
                        break
                if is_problematic:
                    entities_problematic += 1
                    break

                for i in range(entity['pos'][0], entity['pos'][1]):
                    if i == entity['pos'][0]:
                        ner_tags[entity['sent_id']][i] = 'B-' + entity['type']
                    else:
                        ner_tags[entity['sent_id']][i] = 'I-' + entity['type']

                no_problem += 1

        for sent_id in range(len(doc['sents'])):
            dataset_processed.append({
                'tokens': doc['sents'][sent_id],
                'ner_tags': ner_tags[sent_id],
            })
    print(
        f"Entities problematic: {entities_problematic}, Entities no problem: {no_problem}, Ratio: {entities_problematic / (no_problem) * 100} %")
    return dataset_processed


def save_transformed_format(docs, docred_type, split):
    file_path = f'tagged_{docred_type}_{split}.json'
    # with open(path_to_save, 'w') as file:
    #     for doc in docs:
    #         file.write(json.dumps(doc) + '\n')
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(docs, f, indent=4, ensure_ascii=False)


dr_loader = DocREDLoader(os.path.join('..', '..'))
train_dataset = dr_loader.load_docs(docred_type='redocred', split='train')
train_dataset_transformed = transform_docs_info_transformers_tagged_format(train_dataset)
save_transformed_format(train_dataset_transformed, 'redocred', 'train')

dev_dataset = dr_loader.load_docs(docred_type='redocred', split='dev')
dev_dataset_transformed = transform_docs_info_transformers_tagged_format(dev_dataset)
save_transformed_format(dev_dataset_transformed, 'redocred', 'dev')
