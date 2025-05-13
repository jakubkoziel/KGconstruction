from data_utlis import PredictedNERLoader, DocREDLoader
from NERs.utils_NER import evaluate_NER_F1_exact

ner_loader = PredictedNERLoader('NERs')
dr_loader = DocREDLoader()

dev_true = dr_loader.load_docs(docred_type='docred', split='dev')
dev_compare = ner_loader.load_docs(docred_type='docred', split='dev', model_name='v2/deepseek-reasoner-repaired',
                                   prediction_level=str(None))


def is_reconstruction_correct(e, doc):
    e_name_reconstructed = None
    try:
        sentence = doc['sents'][e['sent_id']]
        e_name_reconstructed = sentence[e['pos'][0]: e['pos'][1]]
        for token in e_name_reconstructed:
            if token not in e['name']:
                return False, e_name_reconstructed
    except Exception as e:
        print('here')  # Handling errors never worked as the problem was on sentence level
        return False, None
    return True, e_name_reconstructed


count_of_problems = 0
count_of_all = 0
repaired = 0
unrepaired = 0
count_of_errors = 0

dataset_corrected = []

for i in range(len(dev_compare)):
    new_vertexSet = []
    for eg in dev_compare[i]['vertexSet']:
        for e in eg:
            count_of_all += 1
            # print(e)
            # print(e['name'])
            problem = False
            try:
                correct, e_name_reconstructed = is_reconstruction_correct(e, dev_compare[i])

                if not correct:
                    count_of_problems += 1
                    print(
                        f'Document {i}, name_in_response: {e["name"]}, name_in_sentence: {e_name_reconstructed}')

                    length_of_sent = len(dev_compare[i]['sents'][e['sent_id']])
                    length_of_span = e['pos'][1] - e['pos'][0]
                    successful = False
                    # print('aaaaaaa')
                    for hmm in range(length_of_sent - length_of_span):
                        # print(hmm)
                        e['pos'][0] = hmm
                        e['pos'][1] = hmm + length_of_span
                        correct, e_name_reconstructed = is_reconstruction_correct(e, dev_compare[i])
                        if correct:
                            successful = True
                            break
                    if successful:
                        new_vertexSet.append([e])
                        repaired += 1
                        print('Worked')
                        print(
                            f'Document {i}, name_in_response: {e["name"]}, name_in_sentence: {e_name_reconstructed}')
                    else:
                        unrepaired += 1
                else:
                    new_vertexSet.append([e])

            except Exception as error:
                # print(e, error)
                count_of_errors += 1

    tmp = {}
    for k in dev_compare[i].keys():
        if k == 'vertexSet':
            tmp[k] = new_vertexSet
        else:
            tmp[k] = dev_compare[i][k]
    dataset_corrected.append(tmp)

print(count_of_all, count_of_problems, count_of_errors, unrepaired, repaired)
ner_loader.save_docs(docs=dataset_corrected, docred_type='docred', split='dev',
                     model_name='v2-improved/deepseek-reasoner',
                     prediction_level=str(None))
scores_list1 = evaluate_NER_F1_exact(dev_true, dev_compare, mode='position')
scores_list2 = evaluate_NER_F1_exact(dev_true, dataset_corrected, mode='position')
print('a')
# if e['name'] != ' '.join(e_name_reconstructed):
#     print(f'Document {i}, name_in_response: {e["name"]}, name_in_sentence: {" ".join(e_name_reconstructed)}')
#         break
#     break
# break
