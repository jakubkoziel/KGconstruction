import json
import os


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

        if not os.path.exists(path):
            os.makedirs(path)

        self._save_json(docs, os.path.join(path, f_name))

    def load_docs(self, docred_type: str, split: str, model_name: str, prediction_level: str):
        model_name.replace('/', '--')
        path = os.path.join('data', model_name, prediction_level, docred_type, f'{split}.json')

        return self._load_json(path)

    def check_if_exists(self, docred_type: str, split: str, model_name: str, prediction_level: str):
        model_name.replace('/', '--')
        path = os.path.join('data', model_name, prediction_level, docred_type, f'{split}.json')

        return self._path_exists(path)
