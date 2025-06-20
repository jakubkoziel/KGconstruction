# KGconstruction

This repository contains the source code and experiments related to the Master's diploma thesis:

**"Knowledge graph construction based on text data"**  
Author: Jakub Kozieł  
Supervisor: Anna Wróblewska, Ph.D.  
Advisor: Daniel Dan, Ph.D.  
Faculty of Mathematics and Information Science  
Warsaw University of Technology  

Conducted in collaboration with [APA – Austria Press Agency](https://apa.at/).

## NER with pretrained models

### Repository structure (which files can be used directly)

- NERs/understand_entity_groups - shows entities are not grouped by name or type
- NERs/predict_NER_pretrained - used to run pretrained NER models no OpenAI or deepseek. You need to use
  correct environment for each model:
    - (Spacy models) KGconstruction_environments\spacy -> check KG_constructions_environments_spacy_requirements.txt
    - (Huggingface transformers) (D:\Masters\Masters_thesis\masters_llama_transformers) -> check
      transformers_requirements.txt
    - (Flair & Gliner) (D:\Masters\Masters_thesis\masters_flair_ner) -> check flair_gliner_requirements.txt
    - (Fine-tuned wikineural) -> KGconstruction_environments\transformers to run this you need to first follow the steps
      from NERs/fine-tuning/README.md; After that you need to specify path to fine-tuned model in
      transformers_predict_NER_pretrained_models.py
- NERs/calculate_results - used to calculate results for NERs
- NERs/fine-tuning - used to finetune models -> check readme in the folder for more details

### Environment for spacy models (KGconstruction_environments\spacy)

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install -c conda-forge cupy
conda install -c conda-forge spacy
python -m spacy download en_core_web_trf (and other models as well)
conda install pandas

## NER with LLMs available via API (deepseek-chat, deepseek-reasoner, gpt-4o-mini)

Run using (D:\Masters\Masters_thesis\masters_deepseek) environment (look at LLM_requirements.txt)
First of all, if you wish to perform refinement based on other model predicitions, you need to provide them. In my case
I use /NERs/data/wikineural-multilingual-ner-fine-tuned
which are predictions from fine-tuned wikineural model.

- LLMs_via_API/requests_orchestration_structured_approach -> main file for running experiments using models specified
  above.
    - You can use any other models available via OpenAI API, however, if you wish to use the same models, create
      credentials.py file in parent folder of this project and provide you API credentials
    - You need to specify settings for the experiment you wish to run in # Settings section of main() function
- construct_datasets_from_responses -> used to turn single responses from LLMs into the dataset of unified schema (the
  same as we got in NERs with pretrained models). Predictions will be saved in NERs/data/(experiment_path) You need to
  add conducted experiments there.
- calculate_results_LLMs_API -> calculates results for datasets constructed with construct_datasets_from_responses.py
  and saves them into results_pretrained_LLMs_API.csv
- fill_redocred_predictions_from_docred -> my approach was to calculate first results on docred dev set and then fill
  responses for redocred using this script. Only 2 documents are missing with such approach and need to be filled by
  querying API.


## RE extraction DREEAM.

https://github.com/YoumiMa/dreeam code & model checkpoints
to run dreeam use (D:\Masters\Masters_thesis\masters_env) requirements.txt in dreeam dir

Check dreeam\run_overlay.py to conduct dreeam experiments. Required changes to do on code level of unzipped dreeam
repository are commited. No further changes should be required.

After this evaluate_relation_extraction.py script can be used to evaluate results of dreeam on various NERs of previous
experiments. There are 2 TODOs in runoverlay. One prepares dataset with entities separated and the second prepares
dataset returned by LLMs via API which disregards faulty entities.

RE_results_to_overleaf with manual corrections prepares dreeam results for thesis.

## RE with LLMs

main_NER() in requests_orchestration_structured_approach.py is for NER
main() in requests_orchestration_structured_approach.py is for RE
configuration similar to NERs

RE_descriptions contains reliation_description_docred
from https://github.com/THUDM/AutoRE/blob/main/AutoRE/data/relations_desc/relation_description_redocred.json with my
adjustments
RE_descirpitons contains property_dict which is mine extedsion of relation_description_docred with additional properties
2 jupyters there contain some steps required for reproduction which do not include manual corrections

in LLMs_via_API/minimal subset examples we solve https://en.wikipedia.org/wiki/Set_cover_problem with greedy
estimation - this gives us which docs to use to cover as many relations as possible - relations weighted per occurrences
during selection



