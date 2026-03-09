import re

def extract_lab_values(text):

    values = {}

    hb = re.search(r"Hemoglobin[: ]+(\d+)", text)
    sugar = re.search(r"Blood Sugar[: ]+(\d+)", text)
    chol = re.search(r"Cholesterol[: ]+(\d+)", text)

    if hb:
        values["Hemoglobin"] = int(hb.group(1))

    if sugar:
        values["Blood Sugar"] = int(sugar.group(1))

    if chol:
        values["Cholesterol"] = int(chol.group(1))

    return values