import pandas as pd
import recommender
from recommender import Utility as utility
from recommender import Similarity as similarity


#%% md
#### 3. Loading normalised movies dataset and training data of two models
#%%
movies = pd.read_csv("./dataresult/normalised_movies.csv")
ratings_matrix = pd.read_csv("./dataresult/ratings_matrix.csv",  index_col=0)
content_matrix = pd.read_csv("./dataresult/content_matrix.csv",  index_col=0)
#%%
print(movies.info())
#%%
print(ratings_matrix.info())
#%%
print(content_matrix.info())
#%% md
#### 4. Print top 10 rated movies
#%%
utility.get_top_n_movies(movies, 10)
#%%
movies[movies['title']=="Notting Hill"]
#%% md
#### 5. Make a recommendation with collaborative filtering model
#%%
movie_title = "Notting Hill"
features = ['title', 'genres', 'vote_average']

movie_id = utility.get_movie_by_title(movies, movie_title)
print("10 recommendations for {}".format(movie_title))
similarity.make_recommendation(movie_id, ratings_matrix, movies, features, 10)

#%% md
#### 6. Make recommendation with content based model
#%%

idx = utility.get_idx_by_title(movies, movie_title)
print("10 recommendations for {}".format(movie_title))
similarity.make_recommendation(idx, content_matrix, movies, features, 10)

print("hybrid_evaluate_model.py done")

