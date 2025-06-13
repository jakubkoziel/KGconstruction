import spacy
import cupy
import torch
from spacy.tokens import Doc
from utils_NER import insert_predictions_to_document, get_all_sentences_with_position

from tqdm import tqdm


def predict_spacy(dataset, model_name, prediction_level):
    print("CuPy version:", cupy.__version__)
    print("Available GPUs:", cupy.cuda.runtime.getDeviceCount())

    print("Is CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("GPU devices:", torch.cuda.device_count())

    spacy.require_gpu()

    gpu_available = spacy.require_gpu()
    if gpu_available:
        print("spaCy is using the GPU!")
    else:
        print("spaCy is using the CPU.")

    nlp = spacy.load(model_name)
    dataset_predicted = []
    for i in tqdm(range(len(dataset))):
        doc = dataset[i]
        if prediction_level == 'sentence':
            predictions = _ner_with_spacy_per_sentence(spacy_model=nlp, model_name=model_name, text=doc['sents'])

        elif prediction_level == 'document':
            predictions = _ner_with_spacy_per_document(spacy_model=nlp, model_name=model_name, text=doc['sents'])
        else:
            raise Exception('Such level not supported')

        document_predicted = insert_predictions_to_document(doc, predictions)

        dataset_predicted.append(document_predicted)

    return dataset_predicted


def _ner_with_spacy_per_sentence(spacy_model, model_name, text):
    """Text is whole document."""
    predictions = []

    for i in range(len(text)):
        sentence = text[i]
        if model_name in ('en_core_web_sm', 'en_core_web_lg'):
            doc = Doc(spacy_model.vocab, words=sentence)
            doc = spacy_model.get_pipe("ner")(doc)
        elif model_name == 'en_core_web_trf':
            doc = spacy_model(' '.join(sentence))
        else:
            raise Exception('Wrong model type')

        for ent in doc.ents:
            single_vertex = {
                'name': ent.text,
                'type': ent.label_,
                'pos': [ent.start, ent.end],
                'sent_id': i
            }
            predictions.append(single_vertex)

    return predictions


def _ner_with_spacy_per_document(spacy_model, model_name, text):
    """Text is whole document."""
    predictions = []
    all_sentences, sent_positions = get_all_sentences_with_position(text)

    # doc = Doc(model.vocab, words=all)
    # doc = model.get_pipe("ner")(doc)
    if model_name in ('en_core_web_sm', 'en_core_web_lg'):
        doc = Doc(spacy_model.vocab, words=all_sentences)
        doc = spacy_model.get_pipe("ner")(doc)
    elif model_name == 'en_core_web_trf':
        doc = spacy_model(' '.join(all_sentences))
    else:
        raise Exception('Wrong model type')

    for ent in doc.ents:

        for i in range(len(sent_positions) - 1):
            if sent_positions[i + 1] > ent.start:
                break

        single_vertex = {
            'name': ent.text,
            'type': ent.label_,
            'pos': [ent.start - sent_positions[i], ent.end - sent_positions[i]],
            'sent_id': i
        }
        predictions.append(single_vertex)

    return predictions
