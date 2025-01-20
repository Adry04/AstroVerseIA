import pandas as pd
import random

macro_topics = [
    "Tecnologia e Ingegneria", "Finanza e Investimenti", "Salute e Benessere",
    "Cultura e Società", "Relazioni e Famiglie", "Hobby e Interessi Personali",
    "Scienze e Natura", "Arte e Creatività"
]

recommended_topics = [
    "Economia", "Matematica", "Fisica", "Scienze della terra", "Biologia",
    "Geografia", "Storia", "Giornalismo", "Politica", "Sport", "Programmazione",
    "Intelligenza Artificiale", "Web Design", "Cybersecurity", "Cloud Computing",
    "Data Science", "Crypto", "Videogiochi", "Teatro", "Chitarra", "Musica",
    "Montagna", "Fotografia", "Cucina", "Letteratura", "Cinema", "Anime", "Filosofia"
]

num_instances = 500
data = {
    "id_utente": [f"{i}" for i in range(1, num_instances + 1)],
    "macro_argomento": [random.choice(macro_topics) for _ in range(num_instances)],
    "argomento_spazio": [random.choice(recommended_topics) for _ in range(num_instances)],
    "suggerito": [random.choice([True, False]) for _ in range(num_instances)]
}

df = pd.DataFrame(data)

file_path = "social_dataset.csv"
df.to_csv(file_path, index=False)

print(f"Dataset creato e salvato in: {file_path}")