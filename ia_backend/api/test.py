with open("models.py", "r", encoding="utf-16") as file:
    contenuto = file.read()

with open("models.py", "w", encoding="utf-8") as file:
    file.write(contenuto)