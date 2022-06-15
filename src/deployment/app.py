import os
import joblib
from flask import Flask, render_template,jsonify, request

from config import SAVED_PARAMETER_FOLDER

# NEED TO Test
# - test when no inputs
# - changes to models / route to using different models

def recommend_movie(movie:str, cosine_similarity_array, film_name_index,top_n_results:int=6):
    """
    want to check that film is within the database
        - solutions
            - check within the function
            - sort out in the autocomplete search of the film
    ***Improvement to
    Args:
        movie (_type_): _description_
        cosine_similarity_array (_type_): _description_
        film_name_index (_type_): _description_
    """
    movie_index = film_name_index[film_name_index==movie].index[0]
    similarity = sorted(list(enumerate(cosine_similarity_array[movie_index])),reverse=True,key = lambda x: x[1])
    return [film_name_index[i[0]] for i in similarity[1:top_n_results]]

cos_sim = joblib.load(os.path.join(SAVED_PARAMETER_FOLDER,'cosine_similarity_array.pkl'))
film_nam_ind = joblib.load(os.path.join(SAVED_PARAMETER_FOLDER,'film_name_index.pkl'))

flask_app = Flask(__name__)

@flask_app.route("/" , methods=["GET","POST"])
def home():
    if request.method == "GET":
        movie_names = list(film_nam_ind)
        return render_template("home.html", movie_names=movie_names)
    if request.method == "POST":
        movie_name = request.form.get('movie')
        return str(recommend_movie(movie=movie_name,cosine_similarity_array=cos_sim,film_name_index=film_nam_ind))

@flask_app.route("/predict_api",methods=["POST","GET"])
def predict_api():
    """ to input a json data and to predict from the model"""
    if request.method == "GET":
        return render_template("home.html")
    if request.method == "POST":
        movie_name = request.form.get('movie')
        return str(recommend_movie(movie=movie_name,cosine_similarity_array=cos_sim,film_name_index=film_nam_ind))

