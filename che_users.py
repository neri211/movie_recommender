import pandas as pd
import pickle

# Cargar el encoder de usuarios
with open('model/user_encoder.pkl', 'rb') as f:
    user_encoder = pickle.load(f)

# Cargar los ratings
ratings = pd.read_csv('data/ratings.csv')

print("IDs de usuario en el encoder:", user_encoder.classes_[:10])
print("IDs de usuario únicos en ratings:", sorted(ratings['userId'].unique())[:10])
print("Total de usuarios únicos:", len(ratings['userId'].unique()))