import spacy
from spacy.training.example import Example
from spacy.util import minibatch, compounding
import random

# Load a pre-trained Portuguese SpaCy model and add a new NER pipeline
nlp = spacy.blank("pt")
ner = nlp.add_pipe("ner")

# Add labels to the NER pipeline
ner.add_label("PRODUCT_EAN")
ner.add_label("PRODUCT_NAME")
ner.add_label("PRODUCT_PRICE")

file_path = "escpos_text.txt"

# Lê o conteúdo do arquivo de texto
with open(file_path, 'r') as file:
    full_text = file.read()

# Estrutura TRAIN_DATA usando o conteúdo lido do arquivo
TRAIN_DATA = [
    (full_text, {"entities": [(3, 29, "PRODUCT_NAME"), (45, 50, "PRODUCT_PRICE")]}),
    (full_text, {"entities": [(4, 22, "PRODUCT_NAME"), (44, 49, "PRODUCT_PRICE")]}),
    (full_text, {"entities": [(5, 24, "PRODUCT_NAME"), (45, 50, "PRODUCT_PRICE")]}),
    (full_text, {"entities": [(6, 28, "PRODUCT_NAME"), (50, 55, "PRODUCT_PRICE")]}),
    (full_text, {"entities": [(7, 30, "PRODUCT_NAME"), (52, 56, "PRODUCT_PRICE")]}),
]

# Start training
optimizer = nlp.begin_training()
for i in range(50):  # Adjust the number of iterations based on your dataset
    random.shuffle(TRAIN_DATA)
    losses = {}
    batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
    for batch in batches:
        for text, annotations in batch:
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], drop=0.5, sgd=optimizer, losses=losses)
    print(f"Losses at iteration {i}: {losses}")

# Save the model
nlp.to_disk("product_ner_model_pt")