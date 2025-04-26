import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from tqdm import tqdm
from utils_NER import insert_predictions_to_document, get_all_sentences_with_position


def _predict_on_tokens(tokenizer, model, tokens, device):
    inputs = tokenizer(
        tokens,
        is_split_into_words=True,  # Indicates the input is pre-tokenized
        return_tensors="pt",  # Return as PyTorch tensors
        padding=True,  # Pad inputs for uniform length
        truncation=True,  # Truncate inputs if needed
        return_offsets_mapping=True  # Get mapping of tokens to original input
    )
    offset_mapping = inputs.pop("offset_mapping")  # Remove offset mapping (not used by the model)
    word_ids = inputs.word_ids()  # Map token positions back to the original words

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # inputs = {key: value.to(device) for key, value in inputs.items()}

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs.to(device))

    # Extract logits and predicted class indices
    logits = outputs.logits
    predicted_class_indices = torch.argmax(logits, dim=2)

    # Map class indices to labels
    labels = model.config.id2label
    predicted_labels = [labels[idx.item()] for idx in predicted_class_indices[0]]

    # Align predictions with original tokens
    aligned_labels = []
    previous_word_id = None
    for word_id, label in zip(word_ids, predicted_labels):
        if word_id is None or word_id == previous_word_id:
            # Skip special tokens ([CLS], [SEP]) or subword repetitions
            continue
        aligned_labels.append(label)
        previous_word_id = word_id

    # Combine tokens and their predicted labels
    results = list(zip(tokens, aligned_labels))

    # Print the final predictions
    # print("Predicted Named Entities:")
    # for token, label in results:
    #     print(f"{token}: {label}")

    return results


def _ner_with_transformer_per_document(document_sents, model, tokenizer, device):
    predictions = []
    all_sentences, sent_positions = get_all_sentences_with_position(document_sents)

    results = _predict_on_tokens(model=model, tokenizer=tokenizer, tokens=all_sentences, device=device)
    # print(results)
    name, start, e_type = None, None, None
    for j in range(len(results)):
        if results[j][1].startswith('B-'):
            start = j
            e_type = results[j][1].split('-')[1]
            name = results[j][0]
        elif results[j][1].startswith('I-') and start is not None:
            name += ' ' + results[j][0]
        elif results[j][1].startswith('I-') and start is None:
            start = j
            e_type = results[j][1].split('-')[1]
            name = results[j][0]

        elif results[j][1].startswith('O') and start is not None:
            for i in range(len(sent_positions) - 1):
                if sent_positions[i + 1] > start:
                    break
            single_vertex = {
                'name': name,
                'type': e_type,
                'pos': [start - sent_positions[i], j - sent_positions[i]],
                'sent_id': i
            }
            predictions.append(single_vertex)
            start = None

    return predictions


def _ner_with_transformer_per_sentence(document_sents, model, tokenizer, device):
    predictions = []

    for i in range(len(document_sents)):
        sentence = document_sents[i]
        results = _predict_on_tokens(model=model, tokenizer=tokenizer, tokens=sentence, device=device)
        # print(results)
        name, start, e_type = None, None, None
        for j in range(len(results)):
            if results[j][1].startswith('B-'):
                start = j
                e_type = results[j][1].split('-')[1]
                name = results[j][0]
            elif results[j][1].startswith('I-') and start is not None:
                name += ' ' + results[j][0]
            elif results[j][1].startswith('I-') and start is None:
                start = j
                e_type = results[j][1].split('-')[1]
                name = results[j][0]

            elif results[j][1].startswith('O') and start is not None:
                single_vertex = {
                    'name': name,
                    'type': e_type,
                    'pos': [start, j],
                    'sent_id': i
                }
                predictions.append(single_vertex)
                start = None

    return predictions


def predict_transformers(dataset, model_name, prediction_level):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if model_name == 'wikineural-multilingual-ner-fine-tuned':
        model_name_tmp = r'D:\masters_fine-tune\test-ner\checkpoint-8500'
        tokenizer = AutoTokenizer.from_pretrained(model_name_tmp)
        model = AutoModelForTokenClassification.from_pretrained(model_name_tmp)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
    model.to(device)

    dataset_predicted = []

    for i in tqdm(range(len(dataset))):
        doc = dataset[i]
        if prediction_level == 'sentence':
            predictions = _ner_with_transformer_per_sentence(model=model, tokenizer=tokenizer,
                                                             document_sents=doc['sents'], device=device)

        elif prediction_level == 'document':
            predictions = _ner_with_transformer_per_document(model=model, tokenizer=tokenizer,
                                                             document_sents=doc['sents'], device=device)
        else:
            raise Exception('Such level not supported')

        document_predicted = insert_predictions_to_document(doc, predictions)

        dataset_predicted.append(document_predicted)

    return dataset_predicted
