"""Not used in the end."""
# # Install required packages
# # pip install transformers datasets accelerate torch evaluate seqeval
#
# import torch
# from transformers import (
#     AutoTokenizer,
#     AutoModelForTokenClassification,
#     TrainingArguments,
#     Trainer,
#     DataCollatorForTokenClassification
# )
# from datasets import Dataset, load_metric
# import numpy as np
#
#
# # Step 1: Prepare your dataset
# # Replace this with your dataset loading logic
# # Expected format: List of sentences with tokens and NER tags
# # Example format:
# # dataset = [
# #     {
# #         "tokens": ["John", "lives", "in", "New", "York"],
# #         "ner_tags": [B-PER, O, O, B-LOC, I-LOC]
# #     },
# #     ...
# # ]
#
# def load_custom_dataset():
#     """Implement this function to load your custom dataset"""
#     raise NotImplementedError("Implement your dataset loading")
#
#
# # Load your dataset
# raw_datasets = load_custom_dataset()
#
# # {"BLANK": 0, "ORG": 1, "LOC": 2, "TIME": 3, "PER": 4, "MISC": 5, "NUM": 6}
# # Step 2: Define label mappings
# label_list = [
#     "O",  # Outside
#     "B-PER",  # Begin Person
#     "I-PER",  # Inside Person
#     "B-ORG",  # Begin Organization
#     "I-ORG",  # Inside Organization
#     "B-LOC",  # Begin Location
#     "I-LOC",  # Inside Location
#     "B-TIME",  # Begin Time
#     "I-TIME",  # Inside Time
#     "B-NUM",  # Begin Number
#     "I-NUM",  # Inside Number
#     "B-MISC",  # Begin Miscellaneous
#     "I-MISC"  # Inside Miscellaneous
# ]
#
# label2id = {label: i for i, label in enumerate(label_list)}
# id2label = {i: label for i, label in enumerate(label_list)}
#
# # Step 3: Load model and tokenizer
# model_checkpoint = "xlm-roberta-large-finetuned-conll03-english"
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# model = AutoModelForTokenClassification.from_pretrained(
#     model_checkpoint,
#     id2label=id2label,
#     label2id=label2id,
#     ignore_mismatched_sizes=True
# )
#
#
# # Step 4: Tokenize and align labels
# def tokenize_and_align_labels(examples):
#     tokenized_inputs = tokenizer(
#         examples["tokens"],
#         truncation=True,
#         is_split_into_words=True,
#         max_length=512,
#         padding="max_length"
#     )
#
#     labels = []
#     for i, label in enumerate(examples["ner_tags"]):
#         word_ids = tokenized_inputs.word_ids(batch_index=i)
#         previous_word_idx = None
#         label_ids = []
#         for word_idx in word_ids:
#             if word_idx is None:
#                 label_ids.append(-100)
#             elif word_idx != previous_word_idx:
#                 label_ids.append(label[word_idx])
#             else:
#                 label_ids.append(-100)
#             previous_word_idx = word_idx
#         labels.append(label_ids)
#
#     tokenized_inputs["labels"] = labels
#     return tokenized_inputs
#
#
# # Process dataset
# tokenized_datasets = raw_datasets.map(
#     tokenize_and_align_labels,
#     batched=True,
#     remove_columns=raw_datasets["train"].column_names
# )
#
# # Step 5: Set up training
# data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
# metric = load_metric("seqeval")
#
#
# def compute_metrics(p):
#     predictions, labels = p
#     predictions = np.argmax(predictions, axis=2)
#
#     true_predictions = [
#         [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
#         for prediction, label in zip(predictions, labels)
#     ]
#     true_labels = [
#         [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
#         for prediction, label in zip(predictions, labels)
#     ]
#
#     results = metric.compute(predictions=true_predictions, references=true_labels)
#     return {
#         "precision": results["overall_precision"],
#         "recall": results["overall_recall"],
#         "f1": results["overall_f1"],
#         "accuracy": results["overall_accuracy"],
#     }
#
#
# training_args = TrainingArguments(
#     output_dir="./results",
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     logging_dir="./logs",
#     logging_steps=100,
#     save_strategy="epoch",
#     load_best_model_at_end=True,
#     metric_for_best_model="f1",
#     greater_is_better=True,
#     fp16=True,
# )
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["validation"],
#     data_collator=data_collator,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics
# )
#
# # Step 6: Train the model
# trainer.train()
#
# # Step 7: Save the model
# model.save_pretrained("./fine-tuned-ner-model")
# tokenizer.save_pretrained("./fine-tuned-ner-model")
#
#
# # Step 8: Example inference
# def predict_ner(text):
#     inputs = tokenizer(
#         text.split(),
#         is_split_into_words=True,
#         return_tensors="pt",
#         truncation=True,
#         padding=True
#     )
#
#     with torch.no_grad():
#         outputs = model(**inputs)
#
#     predictions = torch.argmax(outputs.logits, dim=2)
#     tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
#     labels = [id2label[prediction] for prediction in predictions[0].numpy()]
#
#     # Align labels with original words
#     aligned_labels = []
#     previous_word_id = None
#     for word_id, label in zip(inputs.word_ids(0), labels):
#         if word_id is None:
#             continue
#         if word_id != previous_word_id:
#             aligned_labels.append(label)
#             previous_word_id = word_id
#
#     return list(zip(text.split(), aligned_labels))
#
#
# # Example usage
# print(predict_ner("John works at Google in New York"))