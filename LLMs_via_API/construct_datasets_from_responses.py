import os
from data_utlis import DocREDLoader, LLM_API_Response_Loader, PredictedNERLoader


def get_basic_ner_experiment_conducted():
    basic_ner_experiments_conducted = []
    for model in ['deepseek-chat', 'gpt-4o-mini', 'deepseek-reasoner']:
        for experiment in ['v1', 'v2', 'v3', 'v4', 'v5']:
            basic_ner_experiments_conducted.append({
                'experiment_type': 'ner',
                'dataset': 'docred',
                'split': 'dev',
                'experiment': experiment,
                'model': model,
                'narrow_docs_to': 20})

    basic_ner_experiments_conducted.append(
        {
            'experiment_type': 'ner',
            'dataset': 'docred',
            'split': 'dev',
            'experiment': 'v2',
            'model': 'deepseek-reasoner',
            'narrow_docs_to': None})
    basic_ner_experiments_conducted.append(
        {
            'experiment_type': 'ner',
            'dataset': 'docred',
            'split': 'dev',
            'experiment': 'v4',
            'model': 'deepseek-chat',
            'narrow_docs_to': None}
    )

    return basic_ner_experiments_conducted


def get_refinement_ner_experiment_conducted():
    refinement_ner_experiments_conducted = []
    refinement_ner_experiments_conducted.append({
        'experiment_type': 'ner',
        'dataset': 'docred',
        'split': 'dev',
        'experiment': 'v2_refined_v1',
        'model': 'deepseek-reasoner',
        'narrow_docs_to': 20})
    refinement_ner_experiments_conducted.append({
        'experiment_type': 'ner',
        'dataset': 'docred',
        'split': 'dev',
        'experiment': 'v2_refined_v2',
        'model': 'deepseek-reasoner',
        'narrow_docs_to': 20})
    refinement_ner_experiments_conducted.append({
        'experiment_type': 'ner',
        'dataset': 'docred',
        'split': 'dev',
        'experiment': 'v4_refined_v1',
        'model': 'deepseek-chat',
        'narrow_docs_to': 20})
    refinement_ner_experiments_conducted.append({
        'experiment_type': 'ner',
        'dataset': 'docred',
        'split': 'dev',
        'experiment': 'v4_refined_v2',
        'model': 'deepseek-chat',
        'narrow_docs_to': 20})
    refinement_ner_experiments_conducted.append({
        'experiment_type': 'ner',
        'dataset': 'docred',
        'split': 'dev',
        'experiment': 'v4_verifier',
        'model': 'deepseek-chat',
        'narrow_docs_to': 20})
    refinement_ner_experiments_conducted.append({
        'experiment_type': 'ner',
        'dataset': 'docred',
        'split': 'dev',
        'experiment': 'v2_verifier',
        'model': 'deepseek-reasoner',
        'narrow_docs_to': 20})
    refinement_ner_experiments_conducted.append({
        'experiment_type': 'ner',
        'dataset': 'docred',
        'split': 'dev',
        'experiment': 'v2_verifier',
        'model': 'deepseek-reasoner',
        'narrow_docs_to': None})

    return refinement_ner_experiments_conducted


def save_ner_datasets_constructed_from_responses(what_to_save):
    if what_to_save == 'basic':
        ner_experiments_conducted = get_basic_ner_experiment_conducted()
    elif what_to_save == 'refinement':
        ner_experiments_conducted = get_refinement_ner_experiment_conducted()
    else:
        raise ValueError(f"Unknown what: {what_to_save}")

    dr_loader = DocREDLoader('..')
    response_loader = LLM_API_Response_Loader()
    predicted_NER_loader = PredictedNERLoader(os.path.join('..', 'NERs'))

    for experiment in ner_experiments_conducted:
        docs = dr_loader.load_docs(docred_type=experiment['dataset'], split=experiment['split'])
        predicted = response_loader.docs_from_NER_predictions(experiment_type=experiment['experiment_type'],
                                                              dataset=experiment['dataset'], split=experiment['split'],
                                                              experiment=experiment['experiment'],
                                                              model=experiment['model'], docs_starting=docs,
                                                              narrow_docs_to=experiment['narrow_docs_to'])
        print(predicted)
        predicted_NER_loader.save_docs(docs=predicted, docred_type=experiment['dataset'],
                                       split=experiment['split'],
                                       model_name=os.path.join(experiment['experiment'], experiment['model']),
                                       prediction_level=str(experiment['narrow_docs_to']))


if __name__ == '__main__':
    save_ner_datasets_constructed_from_responses(what_to_save='refinement')

    pass
