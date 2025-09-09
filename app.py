from flask import Flask, request, jsonify, render_template
import torch
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import os
import sys
from flask import Flask, request, jsonify, render_template

# Agregar la carpeta model al path de Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

# Ahora intentar importar
try:
    from model.train_model import RecommendationModel
    print("¡Importación exitosa de RecommendationModel!")
except ImportError as e:
    print(f"Error al importar: {e}")
    # Definir una clase dummy para evitar errores
    class RecommendationModel(torch.nn.Module):
        def __init__(self, num_users, num_movies, embedding_dim=50):
            super(RecommendationModel, self).__init__()
            self.user_embedding = torch.nn.Embedding(num_users, embedding_dim)
            self.movie_embedding = torch.nn.Embedding(num_movies, embedding_dim)

app = Flask(__name__)

# Cargar modelo y datos
try:
    # Verificar si el archivo existe y no está vacío
    if not os.path.exists('data/movies.csv') or os.path.getsize('data/movies.csv') == 0:
        raise Exception("El archivo movies.csv no existe o está vacío")
    
    movies = pd.read_csv('data/movies.csv')
    print(f"Películas cargadas: {len(movies)} registros")
    
    # Verificar si los archivos de encoders existen
    if not os.path.exists('model/user_encoder.pkl'):
        raise Exception("El archivo user_encoder.pkl no existe")
    if not os.path.exists('model/movie_encoder.pkl'):
        raise Exception("El archivo movie_encoder.pkl no existe")
    
    user_encoder = pickle.load(open('model/user_encoder.pkl', 'rb'))
    movie_encoder = pickle.load(open('model/movie_encoder.pkl', 'rb'))
    
    num_users = len(user_encoder.classes_)
    num_movies = len(movie_encoder.classes_)
    
    # Verificar si el modelo existe
    if not os.path.exists('model/model.pth'):
        raise Exception("El archivo model.pth no existe")
    
    model = RecommendationModel(num_users, num_movies)
    model.load_state_dict(torch.load('model/model.pth', map_location=torch.device('cpu')))
    model.eval()
    print("¡Modelo cargado correctamente!")
    
except Exception as e:
    print(f"Error al cargar datos o modelo: {e}")
    print("Inicializando con datos de ejemplo...")
    
    # Crear datos de ejemplo
    movies = pd.DataFrame({
        'movieId': [1, 2, 3],
        'title': ['Toy Story', 'Jumanji', 'Grumpier Old Men'],
        'genres': ['Adventure|Animation|Children|Comedy|Fantasy', 'Adventure|Children|Fantasy', 'Comedy|Romance']
    })
    
    # Crear encoders de ejemplo
    user_encoder = LabelEncoder()
    user_encoder.fit([1, 2, 3])
    
    movie_encoder = LabelEncoder()
    movie_encoder.fit([1, 2, 3])
    
    # Crear modelo de ejemplo
    model = RecommendationModel(3, 3)
    print("Sistema inicializado con datos de ejemplo")

# ... el resto de tu código (rutas Flask) ...
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    if model is None:
        return jsonify({'error': 'Modelo no cargado'}), 500
        
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        top_n = data.get('top_n', 10)
        
        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400
        
        # Convertir user_id a entero
        try:
            user_id_int = int(user_id)
        except:
            return jsonify({'error': 'user_id debe ser un número entero'}), 400
        
        # Verificar si el usuario existe en el encoder
        if user_id_int not in user_encoder.classes_:
            return jsonify({'error': 'User not found'}), 404
        
        # Transformar el user_id
        user_encoded = user_encoder.transform([user_id_int])[0]
        
        # Obtener el número de películas del encoder
        num_movies_encoder = len(movie_encoder.classes_)
        
        # Predecir ratings para todas las películas
        user_tensor = torch.LongTensor([user_encoded] * num_movies_encoder)
        movie_tensor = torch.LongTensor(range(num_movies_encoder))
        
        with torch.no_grad():
            predictions = model(user_tensor, movie_tensor).numpy()
        
        # Obtener top N recomendaciones
        top_indices = np.argsort(predictions)[::-1][:top_n]
        recommendations = []
        
        for idx in top_indices:
            # Obtener el movieId original
            movie_id = movie_encoder.inverse_transform([idx])[0]
            # Buscar la película en el DataFrame
            movie_info = movies[movies['movieId'] == movie_id]
            if not movie_info.empty:
                movie_info = movie_info.iloc[0]
                recommendations.append({
                    'movieId': int(movie_id),
                    'title': movie_info['title'],
                    'genres': movie_info['genres'],
                    'predicted_rating': float(predictions[idx])
                })
            else:
                # Si no se encuentra la película, saltar
                continue
        
        return jsonify({
            'user_id': user_id,
            'recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/movies', methods=['GET'])
def get_movies():
    try:
        search = request.args.get('search', '')
        limit = int(request.args.get('limit', 20))
        
        if search:
            filtered_movies = movies[movies['title'].str.contains(search, case=False)]
        else:
            filtered_movies = movies
            
        result = filtered_movies.head(limit).to_dict('records')
        return jsonify({'movies': result})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)