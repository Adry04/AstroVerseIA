with open("models.py", "rb") as file:
    content = file.read()
    if b'\x00' in content:
        print("Il file contiene caratteri null.")
    else:
        print("Il file è pulito.")
