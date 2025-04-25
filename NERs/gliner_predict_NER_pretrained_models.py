from gliner import GLiNER
from utils_NER import insert_predictions_to_document
from tqdm import tqdm


def _ner_with_gliner_per_sentence(model, document_sents):
    predictions = []

    for i in range(len(document_sents)):
        sentence = document_sents[i]

        text = ' '.join(sentence)
        char_to_word_start = [0]
        for w in text.split(' '):
            char_to_word_start.append(len(w))

        for i_c in range(len(char_to_word_start)):
            if i_c != 0:
                char_to_word_start[i_c] += char_to_word_start[i_c - 1] + 1

        start_mapping = {}
        end_mapping = {}
        for i_c in range(len(char_to_word_start)):
            if i_c != len(char_to_word_start) - 1:
                start_mapping[char_to_word_start[i_c]] = i_c
            if i_c != 0:
                end_mapping[char_to_word_start[i_c] - 1] = i_c - 1

        labels = ["organization", "location", "time", "person", "number"]
        entities = model.predict_entities(text, labels)

        for entity in entities:

            try:
                single_vertex = {
                    'name': entity["text"],
                    'type': entity["label"],
                    'pos': [start_mapping[entity['start']], end_mapping[entity['end']] + 1],
                    'sent_id': i
                }

                predictions.append(single_vertex)

            except Exception as e:
                print('error entity mismatch', e)

    return predictions


def predict_gliner(dataset, prediction_level):
    nlp = GLiNER.from_pretrained("urchade/gliner_large-v2.1")

    dataset_predicted = []
    for i in tqdm(range(len(dataset))):
        doc = dataset[i]
        if prediction_level == 'sentence':
            predictions = _ner_with_gliner_per_sentence(model=nlp, document_sents=doc['sents'])
        else:
            raise Exception('Such level not supported')

        document_predicted = insert_predictions_to_document(doc, predictions)

        dataset_predicted.append(document_predicted)

    return dataset_predicted
