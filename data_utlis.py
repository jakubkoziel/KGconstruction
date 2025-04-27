import json
import os
import re


class DataLoader:

    def __init__(self, path_prefix=None):
        self.path_prefix = path_prefix

    def _load_json(self, file_path):
        if self.path_prefix is not None:
            file_path = os.path.join(self.path_prefix, file_path)

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    # def load_docs(self, docred_type: str, split: str):
    #     raise NotImplementedError("This method should be overridden by subclasses")

    def _save_json(self, docs, file_path):
        if self.path_prefix is not None:
            file_path = os.path.join(self.path_prefix, file_path)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(docs, f, indent=4, ensure_ascii=False)

    def _path_exists(self, file_path):
        if self.path_prefix is not None:
            file_path = os.path.join(self.path_prefix, file_path)
        return os.path.exists(file_path)

    def _save_txt(self, txt, file_path):
        if self.path_prefix is not None:
            file_path = os.path.join(self.path_prefix, file_path)
        f = open(file_path, 'w', encoding="utf-8")
        f.write(txt)
        f.close()

    # def _read_txt(self, file_path):
    #     if self.path_prefix is not None:
    #         file_path = os.path.join(self.path_prefix, file_path)
    #     with open(file_path, 'r', encoding='utf-8') as f:
    #         data = f.read()
    #     return data

    def _list_directory(self, path):
        if self.path_prefix is not None:
            path = os.path.join(self.path_prefix, path)
        return os.listdir(path)


class DocREDLoader(DataLoader):

    def load_docs(self, docred_type: str, split: str):

        if docred_type == 'docred':
            folder = 'DocRED'
            if split == 'train':
                f_name = 'train_annotated.json'
            elif split == 'train_distant':
                f_name = 'train_distant.json'
            elif split == 'dev':
                f_name = 'dev.json'
            elif split == 'test':
                f_name = 'test.json'
            else:
                raise ValueError(f"Unknown split: {split}")
        elif docred_type == 'redocred':
            folder = 'Re-DocRED'
            if split == 'train':
                f_name = 'train_revised.json'
            elif split == 'dev':
                f_name = 'dev_revised.json'
            elif split == 'test':
                f_name = 'test_revised.json'
            else:
                raise ValueError(f"Unknown split: {split}")
        else:
            raise ValueError(f"Unknown docred_type: {docred_type}")

        path = os.path.join('data', folder, f_name)

        return self._load_json(path)

    def load_rel_id2name(self):
        path = os.path.join('data', 'DocRED', 'rel_info.json')
        return self._load_json(path)

    def load_rel2id(self):
        path = os.path.join('data', 'DocRED', 'rel2id.json')
        return self._load_json(path)


class PredictedNERLoader(DataLoader):
    def save_docs(self, docs, docred_type: str, split: str, model_name: str, prediction_level: str):
        model_name.replace('/', '--')
        path = os.path.join('data', model_name, prediction_level, docred_type)
        f_name = f'{split}.json'

        if not os.path.exists(os.path.join(self.path_prefix, path)):
            os.makedirs(os.path.join(self.path_prefix, path))

        self._save_json(docs, os.path.join(path, f_name))

    def load_docs(self, docred_type: str, split: str, model_name: str, prediction_level: str):
        model_name.replace('/', '--')
        path = os.path.join('data', model_name, prediction_level, docred_type, f'{split}.json')

        return self._load_json(path)

    def check_if_exists(self, docred_type: str, split: str, model_name: str, prediction_level: str):
        model_name.replace('/', '--')
        path = os.path.join('data', model_name, prediction_level, docred_type, f'{split}.json')

        return self._path_exists(path)


class LLM_API_Response_Loader(DataLoader):
    def save_response(self, experiment_type, dataset, split, experiment, model, document_id, response):
        json_path = os.path.join(experiment_type, dataset, split, experiment, model)
        txt_path = os.path.join(experiment_type, dataset, split, experiment, model + '-errors')
        txt_path_responses = os.path.join(experiment_type, dataset, split, experiment, model + '-responses')

        for p in [json_path, txt_path, txt_path_responses]:
            if not os.path.exists(p):
                os.makedirs(p)

        # Save raw response
        file_path = os.path.join(txt_path_responses, f"doc_{document_id}.txt")
        self._save_txt(response, file_path)

        try:
            result = json.loads(re.search(r'\$\$\$(.*)\$\$\$', response, re.DOTALL).group(1))
            file_path = os.path.join(json_path, f"doc_{document_id}.json")
            self._save_json(result, file_path)
        except Exception as e:
            file_path = os.path.join(txt_path, f"doc_{document_id}.txt")
            self._save_txt(response, file_path)

    def read_single_response(self, experiment_type, dataset, split, experiment, model, document_id):
        json_path = os.path.join(experiment_type, dataset, split, experiment, model)

        # Save raw response
        file_path = os.path.join(json_path, f"doc_{document_id}.json")
        return self._load_json(file_path)

    def check_if_document_done(self, experiment_type, dataset, split, experiment, model, doc_id):
        path = os.path.join(experiment_type, dataset, split, experiment, model, 'doc_' + str(doc_id) + '.json')

        return self._path_exists(path)

    def save_document_ids_with_errors(self, process_id, experiment_type, dataset, split, experiment, model,
                                      data_to_save):
        txt_path = os.path.join(experiment_type, dataset, split, experiment, model + '-errors-ids')
        if not os.path.exists(txt_path):
            os.makedirs(txt_path)
        file_path = os.path.join(txt_path,
                                 f"process_{process_id}_errors.txt")
        self._save_txt(','.join(data_to_save), file_path)

    def docs_from_NER_predictions(self, docs_starting, experiment_type, dataset, split, experiment, model,
                                  narrow_docs_to):
        path_to_predictions = os.path.join(experiment_type, dataset, split, experiment, model)

        output_docs = []
        if narrow_docs_to is not None:
            no_docs_to_read = narrow_docs_to
        else:
            no_docs_to_read = len(self._list_directory(path_to_predictions))

        for i in range(no_docs_to_read):
            predicted = self._load_json(os.path.join(path_to_predictions, f'doc_{i}.json'))
            nested_ents = []
            for entity in predicted:
                nested_ents.append([entity])
            document_predicted = {}
            for k in docs_starting[i].keys():
                if k != 'vertexSet':
                    document_predicted = document_predicted | {k: docs_starting[i][k]}
                else:
                    document_predicted = document_predicted | {k: nested_ents}
            output_docs.append(document_predicted)

        return output_docs
