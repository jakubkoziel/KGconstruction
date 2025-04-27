# KGconstruction

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
First of all, if you wish to perform refinement based on other model predicitions, you need to provide them. In my case I use /NERs/data/wikineural-multilingual-ner-fine-tuned
which are predictions from fine-tuned wikineural model.

- LLMs_via_API/requests_orchestration_structured_approach -> main file for running experiments using models specified above. 
  - You can use any other models available via OpenAI API, however, if you wish to use the same models, create credentials.py file in parent folder of this project and provide you API credentials
  - You need to specify settings for the experiment you wish to run in # Settings section of main() function
- construct_datasets_from_responses -> used to turn single responses from LLMs into the dataset of unified schema (the same as we got in NERs with pretrained models). Predictions will be saved in NERs/data/(experiment_path)
- calculate_results_LLMs_API -> calculates results for datasets constructed with construct_datasets_from_responses.py and saves them into results_pretrained_LLMs_API.csv

