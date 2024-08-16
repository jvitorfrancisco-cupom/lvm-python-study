import spacy
import pathlib

# Testing the model with a new text
file_path1 = "example1.txt"
file_path2 = "example2.txt"
text1 = pathlib.Path(file_path2).read_text(encoding="utf-8")
nlp = spacy.load("product_ner_model_pt")
doc = nlp.make_doc(text1)
for ent in doc.ents:
    print(ent.text, ent.label_)