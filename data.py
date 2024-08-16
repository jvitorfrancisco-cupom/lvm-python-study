# Training data (example)
TRAIN_DATA = [
    ("7897664171718-SABONETE ALBANY 1X 2.29 2,29", {
        "entities": [(0, 13, "PRODUCT_EAN"),(14, 28, "PRODUCT_NAME"), (34, 39, "PRODUCT_PRICE")]
    }),
    ("7897664171718-ALGODAO 85G - 3X 0.50 1.50", {
        "entities": [(0, 13, "PRODUCT_EAN"),(14, 20, "PRODUCT_NAME"), (37, 41, "PRODUCT_PRICE")]
    }),
    ("7891575329039-SORVETE 1X 1.49 1,49", {
        "entities": [(0, 13, "PRODUCT_EAN"),(14, 17, "PRODUCT_NAME"), (27, 31, "PRODUCT_PRICE")]
    }),
    ("7891000100100-CERVEJA LATA 350ML 2X 2.99 5,98", {
        "entities": [(0, 13, "PRODUCT_EAN"),(14, 26, "PRODUCT_NAME"), (36, 41, "PRODUCT_PRICE")]
    }),
    ("7891010101010-LEITE UHT 1L 1X 3.49 3,49", {
        "entities": [(0, 13, "PRODUCT_EAN"),(14, 19, "PRODUCT_NAME"), (29, 33, "PRODUCT_PRICE")]
    }),
    ("7891020202020-ARROZ PARBOILIZADO 5KG 1X 15.99 15,99", {
        "entities": [(0, 13, "PRODUCT_EAN"),(14, 31, "PRODUCT_NAME"), (41, 46, "PRODUCT_PRICE")]
    }),
    ("7891030303030-CAFE EM PÓ 500G 2X 4.99 9,98", {
        "entities": [(0, 13, "PRODUCT_EAN"),(14, 23, "PRODUCT_NAME"), (33, 38, "PRODUCT_PRICE")]
    }),
    # -----------------------------------------------------------------------
    ("02 7891079000229 MAC. INST. NISSIN LAME 4 UN X 2,89 11,56", {
        "entities": [(4, 16, "PRODUCT_EAN"), (18, 40, "PRODUCT_NAME"), (47, 51, "PRODUCT_PRICE")]
    }),
    ("03 7891234567890 ARROZ TIO JOÃO 5KG 1X 19,99 19,99", {
        "entities": [(4, 16, "PRODUCT_EAN"), (18, 35, "PRODUCT_NAME"), (39, 44, "PRODUCT_PRICE")]
    }),
    ("04 7899876543210 FEIJÃO CARIOCA 1KG 2X 7,49 14,98", {
        "entities": [(4, 16, "PRODUCT_EAN"), (18, 35, "PRODUCT_NAME"), (39, 43, "PRODUCT_PRICE")]
    }),
    ("05 7896543210987 CAFÉ PILÃO 500G 3X 8,99 26,97", {
        "entities": [(4, 16, "PRODUCT_EAN"), (18, 32, "PRODUCT_NAME"), (37, 40, "PRODUCT_PRICE")]
    }),
    ("06 7891111111111 AÇÚCAR UNIÃO 1KG 2X 3,29 6,58", {
        "entities": [(4, 16, "PRODUCT_EAN"), (18, 33, "PRODUCT_NAME"), (38, 41, "PRODUCT_PRICE")]
    }),
    ("07 7892222222222 ÓLEO SOJA LIZA 900ML 1X 5,49 5,49", {
        "entities": [(4, 16, "PRODUCT_EAN"), (18, 37, "PRODUCT_NAME"), (37, 44, "PRODUCT_PRICE")]
    }),
    ("08 7893333333333 MACARRÃO PENE 500G 4X 2,49 9,96", {
        "entities": [(4, 16, "PRODUCT_EAN"), (18, 34, "PRODUCT_NAME"), (38, 42, "PRODUCT_PRICE")]
    }),
    ("09 7894444444444 LEITE NINHO INTEGRAL 1L 3X 6,79 20,37", {
        "entities": [(4, 16, "PRODUCT_EAN"), (18, 39, "PRODUCT_NAME"), (43, 47, "PRODUCT_PRICE")]
    }),
    ("10 7895555555555 DESODORANTE DOVE 150ML 1X 12,99 12,99", {
        "entities": [(4, 16, "PRODUCT_EAN"), (18, 38, "PRODUCT_NAME"), (42, 47, "PRODUCT_PRICE")]
    }),
    ("11 7896666666666 CREME DENTAL COLGATE 90G 2X 4,99 9,98", {
        "entities": [(4, 16, "PRODUCT_EAN"), (18, 40, "PRODUCT_NAME"), (44, 48, "PRODUCT_PRICE")]
    }),
    ("12 7897777777777 DETERGENTE YPÊ 500ML 3X 1,89 5,67", {
        "entities": [(4, 16, "PRODUCT_EAN"), (18, 36, "PRODUCT_NAME"), (40, 44, "PRODUCT_PRICE")]
    }),
    # -----------------------------------------------------------------------
    # ("Dinheiro 20,00", {"entities": [(0, 8, "PAYMENT_METHOD"), (9, 14, "PAYMENT_VALUE")]}),
    # ("Dinheiro 5,20", {"entities": [(0, 8, "PAYMENT_METHOD"), (9, 13, "PAYMENT_VALUE")]}),
    # ("Dinheiro 107,99", {"entities": [(0, 8, "PAYMENT_METHOD"), (9, 15, "PAYMENT_VALUE")]}),
    # ("Dinheiro 0,99", {"entities": [(0, 8, "PAYMENT_METHOD"), (9, 13, "PAYMENT_VALUE")]}),
    # ("Cartão de Crédito 5,00", {"entities": [(0, 17, "PAYMENT_METHOD"), (18, 22, "PAYMENT_VALUE")]}),
    # ("Cartão de Crédito 62,01", {"entities": [(0, 17, "PAYMENT_METHOD"), (18, 23, "PAYMENT_VALUE")]}),
    # ("Cartão de Crédito 134,71", {"entities": [(0, 17, "PAYMENT_METHOD"), (18, 24, "PAYMENT_VALUE")]}),
    # -----------------------------------------------------------------------
    # ("4319 0701 2345 6700 0174 5500 1000 0001 2345 6789 0018", {"entities": [(0, 54, "ACCESS_KEY")]}),
    # ("3518 0308 7654 3200 0127 5500 1000 0006 7895 4321 0009", {"entities": [(0, 54, "ACCESS_KEY")]}),
    # ("4217 0609 8765 4300 0132 5500 1000 0004 5678 9432 0015", {"entities": [(0, 54, "ACCESS_KEY")]}),
    # ("3119 0704 3210 9870 0142 5500 1000 0002 3456 7890 0028", {"entities": [(0, 54, "ACCESS_KEY")]}),
    # ("2918 0402 3456 7800 0156 5500 1000 0008 9765 4321 0032", {"entities": [(0, 54, "ACCESS_KEY")]}),
    # # -----------------------------------------------------------------------
    # ("43190701234567000174550010000001234567890018", {"entities": [(0, 44, "ACCESS_KEY")]}),
    # ("35180308765432000127550010000006789543210009", {"entities": [(0, 44, "ACCESS_KEY")]}),
    # ("42170609876543000132550010000004567894320015", {"entities": [(0, 44, "ACCESS_KEY")]}),
    # ("31190704321098700142550010000002345678900028", {"entities": [(0, 44, "ACCESS_KEY")]}),
    # ("29180402345678000156550010000008976543210032", {"entities": [(0, 44, "ACCESS_KEY")]})
]