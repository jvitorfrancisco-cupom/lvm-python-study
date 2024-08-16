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
# ner.add_label("PAYMENT_METHOD")
# ner.add_label("PAYMENT_VALUE")
# ner.add_label("ACCESS_KEY")

TRAIN_DATA = [
   ("02 7891079000229 MAC. INST. NISSIN LAME 4 UN X 2,89 11,56", {"entities": [(3, 29, "PRODUCT_NAME"), (45, 50, "PRODUCT_PRICE")]}),
   ("03 7891234567890 ARROZ TIO JOÃO 5KG 1 UN X 19,90 19,90", {"entities": [(3, 21, "PRODUCT_NAME"), (43, 48, "PRODUCT_PRICE")]}),
   ("04 7899876543210 FEIJÃO CARIOCA 1KG 2 UN X 7,50 15,00", {"entities": [(3, 22, "PRODUCT_NAME"), (43, 48, "PRODUCT_PRICE")]}),
   ("05 7896543210987 LEITE COND. MOÇA 395G 3 UN X 4,99 14,97", {"entities": [(3, 25, "PRODUCT_NAME"), (47, 52, "PRODUCT_PRICE")]}),
   ("06 7891111111111 ÓLEO DE SOJA LIZA 900ML 1 UN X 4,99 4,99", {"entities": [(3, 26, "PRODUCT_NAME"), (48, 52, "PRODUCT_PRICE")]}),
   ("07 7892222222222 AÇÚCAR REFINADO UNIÃO 1KG 2 UN X 2,89 5,78", {"entities": [(3, 30, "PRODUCT_NAME"), (50, 55, "PRODUCT_PRICE")]}),
   ("08 7893333333333 SABÃO EM PÓ OMO 800G 1 UN X 9,99 9,99", {"entities": [(3, 23, "PRODUCT_NAME"), (45, 50, "PRODUCT_PRICE")]}),
   ("09 7894444444444 CERVEJA BRAHMA LATA 350ML 6 UN X 2,49 14,94", {"entities": [(3, 28, "PRODUCT_NAME"), (50, 55, "PRODUCT_PRICE")]}),
   ("10 7895555555555 CAFÉ PILÃO TRADICIONAL 500G 1 UN X 8,99 8,99", {"entities": [(3, 30, "PRODUCT_NAME"), (52, 57, "PRODUCT_PRICE")]}),
   ("11 7896666666666 MACARRÃO ESPAGUETE BARILLA 500G 2 UN X 5,49 10,98", {"entities": [(3, 34, "PRODUCT_NAME"), (56, 61, "PRODUCT_PRICE")]}),
   ("12 7897777777777 CREME DENTAL COLGATE TOTAL 90G 1 UN X 3,99 3,99", {"entities": [(3, 33, "PRODUCT_NAME"), (54, 59, "PRODUCT_PRICE")]}),
   ("13 7898888888888 AMACIANTE CONFORT 2L 1 UN X 12,99 12,99", {"entities": [(3, 28, "PRODUCT_NAME"), (49, 54, "PRODUCT_PRICE")]}),
   ("14 7899999999999 DETERGENTE YPÊ 500ML 3 UN X 1,79 5,37", {"entities": [(3, 25, "PRODUCT_NAME"), (45, 50, "PRODUCT_PRICE")]}),
   ("15 7890000000000 SABONETE DOVE 90G 4 UN X 1,99 7,96", {"entities": [(3, 22, "PRODUCT_NAME"), (44, 49, "PRODUCT_PRICE")]}),
   ("16 7890111111111 MANTEIGA PRESIDENT 200G 1 UN X 15,99 15,99", {"entities": [(3, 29, "PRODUCT_NAME"), (52, 57, "PRODUCT_PRICE")]}),
]

    # ("VALOR TOTAL R$ 22,38"),
    # ("Total:6,77"),
    # ("Total:6,77"),

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

