def _extract_entities(docs, mode):
    entities = []
    for i in range(len(docs)):
        doc = docs[i]
        vs = doc['vertexSet']
        for entity_list in vs:
            for entity in entity_list:
                if mode == 'position_type':
                    entities.append(
                        (doc['title'], (entity['pos'][0], entity['pos'][1]), entity['sent_id'], entity['type'],))
                else:
                    entities.append((doc['title'], (entity['pos'][0], entity['pos'][1]), entity['sent_id'],))

    return set(entities)


def _count_TP_FP_position(entities_true, entities_predicted):
    TP = 0
    FP = 0

    for ep in entities_predicted:
        if ep in entities_true:
            TP += 1
        else:
            FP += 1

    return TP, FP


def _count_TP_FP_position_type(entities_true, entities_predicted):
    TP = 0
    FP = 0

    for ep in entities_predicted:

        if ep in entities_true:
            TP += 1
        else:
            FP += 1

    return TP, FP


def evaluate_NER_F1_exact(docs_true, docs_predicted, mode):
    if mode != 'position' and mode != 'position_and_type':
        raise NotImplementedError(f'Mode {mode} is not implemented')

    entities_true = _extract_entities(docs_true, mode)
    entities_predicted = _extract_entities(docs_predicted, mode)

    if mode == 'position':
        TP, FP = _count_TP_FP_position(entities_true, entities_predicted)
    else:
        TP, FP = _count_TP_FP_position_type(entities_true, entities_predicted)

    FN = len(entities_true) - TP
    F1 = 2 * TP / (2 * TP + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    print(f'TP: {TP}; FP: {FP}; FN: {FN}; F1score: {F1}')

    return TP, FP, FN, F1, precision, recall


def get_all_sentences_with_position(text):
    sent_positions = [0]

    all_sentences = []
    for sent in text:
        all_sentences += sent
        sent_positions.append(len(sent))

    for cumulative in range(1, len(sent_positions)):
        sent_positions[cumulative] += sent_positions[cumulative - 1]

    return all_sentences, sent_positions


def insert_predictions_to_document(doc, predictions):
    vertex_set = []
    for p in predictions:
        vertex_set.append([p])

    document_predicted = {}
    for k in doc.keys():
        if k != 'vertexSet':
            document_predicted = document_predicted | {k: doc[k]}
        else:
            document_predicted = document_predicted | {k: vertex_set}

    return document_predicted
