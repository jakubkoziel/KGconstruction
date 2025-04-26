# Fine-tuning models

## First step is to prepare data according to transformer requirements.

Use: prepare_set_for_run_ner.py
BIO tagging doesn't support nested entities which were found in the dataset. Some of overlapping/nested entities were of
different entity type.
Redocred train stats: Entities problematic: 520, Entities no problem: 78782, Ratio: 0.6600492498286411 %
Redocred dev stats: Entities problematic: 56, Entities no problem: 13117, Ratio: 0.4269268887702981 %
As number of them is not significant we decide to omit such entities choosing only 1 from overlapping span.

## Environment

Environment for fine-tuning models (D:\Masters\KGconstruction_environments\transformers) Python3.11
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install datasets>=1.8.0 accelerate>=0.12.0 seqeval evaluate
pip install git+https://github.com/huggingface/transformers -> installed from source "4.52.0.dev0"
(KGconstruction_transformers_requirements.txt)

## Notes

FacebookAI/xlm-roberta-large-finetuned-conll03-english achieved best results but has much more parameters compared to
Babelscape/wikineural-multilingual-ner which was second.
It's 560M vs 177M parameters and training XLM on GTX 1660 Ti was not possible due to only 6GB memory available.
We decide to finetune wikineural-multilingual-ner on our dataset.

## Commands

Sample commands to use for run_ner or run_ner_no_trainer (can be tailored to needs) I have used first option. 3 epochs,
batch size 8
The files for finetuning are
from https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/README.md

```
python run_ner.py --model_name_or_path Babelscape/wikineural-multilingual-ner --train_file tagged_redocred_train.json --validation_file tagged_redocred_dev.json --output_dir /masters_fine-tune/test-ner --do_train --do_eval --ignore_mismatched_sizes

python run_ner_no_trainer.py --model_name_or_path Babelscape/wikineural-multilingual-ner --train_file tagged_redocred_train.json --validation_file tagged_redocred_dev.json --task_name ner --max_length 128 --per_device_train_batch_size 2 --learning_rate 2e-5 --num_train_epochs 3 --output_dir /tmp/ner/ --ignore_mismatched_sizes
```

## Reevaluation and eval metrics (with visualization)

Use eval_all_checkpoints to recalculate them on validation set and save.
Use visualize_eval_results to generate dataframe and plot. (matplotlib to install for visualization)