from flair.data import Sentence
from flair.models import SequenceTagger
import re
from tqdm import tqdm
from utils_NER import insert_predictions_to_document, get_all_sentences_with_position


def _ner_with_flair_per_sentence(model, document_sents):
    predictions = []

    for i in range(len(document_sents)):
        sentence = document_sents[i]

        sentence = Sentence(' '.join(sentence))
        model.predict(sentence)

        for entity in sentence.get_spans('ner'):
            span_id = re.findall('^Span\[\d+:\d+\]', entity.unlabeled_identifier)[0]
            span_id_start = re.findall('\d+', span_id)[0]
            span_id_end = re.findall('\d+', span_id)[1]

            single_vertex = {
                'name': entity.text,
                'type': entity.get_label().value,
                'pos': [int(span_id_start), int(span_id_end)],
                'sent_id': i
            }
            predictions.append(single_vertex)

    return predictions


def _ner_with_flair_per_document(model, document_sents):
    predictions = []
    all_sentences, sent_positions = get_all_sentences_with_position(document_sents)

    sentence = Sentence(' '.join(all_sentences))
    model.predict(sentence)

    for entity in sentence.get_spans('ner'):

        # print(entity.get_label())
        # print(entity.unlabeled_identifier)
        span_id = re.findall('^Span\[\d+:\d+\]', entity.unlabeled_identifier)[0]
        span_id_start = re.findall('\d+', span_id)[0]
        span_id_end = re.findall('\d+', span_id)[1]
        for i in range(len(sent_positions) - 1):
            if sent_positions[i + 1] > int(span_id_start):
                break

        single_vertex = {
            'name': entity.text,
            'type': entity.get_label().value,
            'pos': [int(span_id_start) - sent_positions[i], int(span_id_end) - sent_positions[i]],
            'sent_id': i
        }

        predictions.append(single_vertex)

    return predictions


def predict_flair(dataset, prediction_level):
    nlp = SequenceTagger.load("flair/ner-english-ontonotes-fast")

    dataset_predicted = []
    for i in tqdm(range(len(dataset))):
        doc = dataset[i]
        if prediction_level == 'sentence':
            predictions = _ner_with_flair_per_sentence(model=nlp, document_sents=doc['sents'])

        elif prediction_level == 'document':
            predictions = _ner_with_flair_per_document(model=nlp, document_sents=doc['sents'])
        else:
            raise Exception('Such level not supported')

        document_predicted = insert_predictions_to_document(doc, predictions)

        dataset_predicted.append(document_predicted)

    return dataset_predicted
