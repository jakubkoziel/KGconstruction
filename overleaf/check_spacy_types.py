import spacy

# Load the spaCy model (replace with your preferred model)
nlp = spacy.load("en_core_web_sm")

# Check if the pipeline has a NER component
if "ner" in nlp.pipe_names:
    # Get all unique entity labels from the NER component
    ner_labels = nlp.get_pipe("ner").labels
    unique_ner_labels = set(ner_labels)

    print("Available NER types:")
    for label in sorted(unique_ner_labels):
        print(f"- {label}")
else:
    print("This model doesn't have a NER component.")