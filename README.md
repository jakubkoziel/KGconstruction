# KGconstruction

## NER with pretrained models

### Repository structure (which files can be used directly)

- NERs/understand_entity_groups - shows entities are not grouped by name or type
- NERs/predict_NER_pretrained - used to run pretrained NER models no training, no OpenAI or deepseek. You need to use
  correct environment for each model:
    - (Spacy models) KGconstruction_environments\spacy -> check KG_constructions_environments_spacy_requirements.txt
    - (Huggingface transformers) (D:\Masters\Masters_thesis\masters_llama_transformers) -> check
      transformers_requirements.txt
    - (Flair & Gliner) (D:\Masters\Masters_thesis\masters_flair_ner) -> check flair_gliner_requirements.txt
- NERs/calculate_results - used to calculate results for NERs
- NERs/fine-tuning - used to finetune models -> check readme in the folder for more details
- 

### Environment for spacy models (KGconstruction_environments\spacy)

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install -c conda-forge cupy
conda install -c conda-forge spacy
python -m spacy download en_core_web_trf (and other models as well)
conda install pandas 