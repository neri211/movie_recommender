import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

class RecommendationModel(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=50):
        super(RecommendationModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, user_ids, movie_ids):
        user_embedded = self.user_embedding(user_ids)
        movie_embedded = self.movie_embedding(movie_ids)
        x = torch.cat([user_embedded, movie_embedded], dim=1)
        return self.fc(x).squeeze()

def prepare_data():
    # Obtener la ruta base del proyecto
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    # Construir rutas completas a los archivos
    ratings_path = os.path.join(data_dir, 'ratings.csv')
    movies_path = os.path.join(data_dir, 'movies.csv')
    
    print(f"Buscando archivos en:")
    print(f"- Ratings: {ratings_path}")
    print(f"- Movies: {movies_path}")
    
    # Verificar que los archivos existen
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(f"No se encuentra el archivo: {ratings_path}")
    if not os.path.exists(movies_path):
        raise FileNotFoundError(f"No se encuentra el archivo: {movies_path}")
    
    # Cargar datos
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    
    # Preprocesamiento
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    
    ratings['userId'] = user_encoder.fit_transform(ratings['userId'])
    ratings['movieId'] = movie_encoder.fit_transform(ratings['movieId'])
    
    # Guardar encoders en la carpeta model
    model_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(model_dir, 'user_encoder.pkl'), 'wb') as f:
        pickle.dump(user_encoder, f)
    with open(os.path.join(model_dir, 'movie_encoder.pkl'), 'wb') as f:
        pickle.dump(movie_encoder, f)
    
    return ratings, movies, user_encoder, movie_encoder

def train_model():
    ratings, movies, user_encoder, movie_encoder = prepare_data()
    
    # Dividir datos
    train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
    
    # Preparar tensores
    train_users = torch.LongTensor(train_data['userId'].values)
    train_movies = torch.LongTensor(train_data['movieId'].values)
    train_ratings = torch.FloatTensor(train_data['rating'].values)
    
    # Modelo
    num_users = len(user_encoder.classes_)
    num_movies = len(movie_encoder.classes_)
    model = RecommendationModel(num_users, num_movies)
    
    # Entrenamiento
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    batch_size = 1024
    num_epochs = 10
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for i in range(0, len(train_users), batch_size):
            batch_users = train_users[i:i+batch_size]
            batch_movies = train_movies[i:i+batch_size]
            batch_ratings = train_ratings[i:i+batch_size]
            
            optimizer.zero_grad()
            predictions = model(batch_users, batch_movies)
            loss = criterion(predictions, batch_ratings)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_users)}')
    
    # Guardar modelo
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pth')
    torch.save(model.state_dict(), model_path)
    print("Modelo entrenado y guardado!")

if __name__ == "__main__":
    train_model()