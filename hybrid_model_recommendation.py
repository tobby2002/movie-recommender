import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import json
# Parse the stringified features into their corresponding python objects
from ast import literal_eval
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import recommender
from recommender import Utility as utility
from recommender import Similarity as similarity
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

#### 1. Reading data
# define function
#p_ratings = pd.read_csv"/content/drive/My Drive/Test/ratings.csv")
#pd.read_csv("https://raw.githubusercontent.com/diem-ai/colab4test/master/data/ratings_small.csv")
movies = pd.read_csv("./data/movies_metadata.csv")
credits = pd.read_csv("./data/credits.csv")
keywords = pd.read_csv("./data/keywords.csv")
ratings = pd.read_csv("./data/ratings.csv")

print(movies.columns)
print(credits.columns)
print(credits.head(3))
print(keywords.columns)
print(keywords.head(3))
print(keywords.columns)
print(keywords.head(3))
print(ratings.columns)
print(ratings.head(3))

#### 2. Preprocesing Data
# Filter out invalid rows in movies dataset and convert id to interger type so that we can merge movies dataset with others

movies = movies.drop(movies[movies['id']=='1997-08-20'].index)
movies = movies.drop(movies[movies['id']=='2012-09-29'].index)
movies = movies.drop(movies[movies['id']=='2014-01-01'].index)

movies['id'] = movies['id'].astype(int)
credits['id'] = credits['id'].astype(int)
keywords['id'] = keywords['id'].astype(int)

#movies[movies['id'] == '1997-08-20']
#19730

#movies['id'] = movies['id'].astype(int)
#%% md
# genres, cast, crew, keywords in movies, credits and keywords datasets are string json objects. We only take out relevant words for our recommendation model
#%%
# genres column is a json string object. We will take out genre values only.
movies['genres'] = movies['genres'].apply(literal_eval)
#%%
credits['cast'] = credits['cast'].apply(lambda x : literal_eval(x))
#%%
credits['crew'] = credits['crew'].apply(lambda x : literal_eval(x))
#%%
keywords['keywords'] = keywords['keywords'].apply(lambda x : literal_eval(x))
#%%
#get genre only
movies['genres'] = movies['genres'].apply(lambda x : utility.get_genre(x))
credits['cast'] = credits['cast'].apply(lambda x : utility.get_cast(x))
credits['crew'] = credits['crew'].apply(lambda x : utility.get_cast(x))
keywords['keywords'] = keywords['keywords'].apply(lambda x : utility.get_genre(x))
keywords['keywords'] = utility.preprocess_series_text(keywords['keywords'])
#%% md
# Revew above columns after processing
#%%
credits.head()
#%%
keywords.head()
#%%
movies[['id','title', 'genres', 'vote_count', 'vote_average']].head(3)
#%%
# Save processed movies dataset for evaluation purpose
p_movies = movies[['id', 'title', 'genres', 'vote_count', 'vote_average']]
p_movies.to_csv('./dataresult/normalised_movies.csv', index=True, header=True)
#%% md
#### 3 . Building collaborative filtering mode by combing movies and ratings dataset. The system generated uses only information about rating profiles for different users on items.
#%%
ratings = ratings[['userId', 'movieId', 'rating']].merge(movies[['id', 'title', 'genres', 'vote_count', 'vote_average']], left_on='movieId', right_on='id', how='left')
#%%
# see 5 top items after merging
ratings = ratings.sort_values(by=['vote_average'], ascending=False)
ratings.head()
#%%
#drop movieId column. it is the same with id
ratings = ratings.drop(columns=['movieId'])
#remove duplicated movies:
ratings = ratings.drop_duplicates(['id'], keep='first')
ratings.head()
#%%
ratings = ratings.dropna()
ratings['id'] = ratings['id'].astype(int)
ratings.head(2)
#%%
#Build factorization matrix:
## reshape the ratings dataframe: index is movie id, rows are ratings and columns is user id
rating_matrix = ratings.pivot(index = 'id', columns ='userId', values = 'rating').fillna(0)

#%%
n_components = rating_matrix.shape[1]
print("Best number for TSVD: {} ".format(n_components))
svd = TruncatedSVD(n_components=(n_components-1))
latent_rating = svd.fit_transform(rating_matrix)
print(latent_rating[:5])
#%%
#save matrix as ratings_matrix.csv for the model validation
latent_matrix_df = pd.DataFrame(latent_rating, index=rating_matrix.index.tolist())
latent_matrix_df.to_csv("./dataresult/ratings_matrix.csv", index=True, header=True)
#%%
# Plot variance and cumulative variance for ratings based system
expl = svd.explained_variance_ratio_
plt.plot(expl, '.-', ms=15)
plt.title('Variance Explained ');
plt.show()

plt.plot(np.cumsum(expl), '.-', ms=15)
plt.title('Cumulative Variance');
plt.show()
#%%
# Make few tests with rating user based recommendation:
#rating_matrix[278]
#movies[movies['id'] == 278]
#latent_matrix_df = pd.read_csv("/content/drive/My Drive/Test/ratings_matrix.csv", index_col=0)
features = ['title', 'genres', 'vote_average','score']

movie_id = utility.get_movie_by_title(movies, "GoldenEye")
print("movie id {}: ".format(movie_id))

similarity.make_recommendation(movie_id, latent_matrix_df, movies, features, 5)

#%% md
#### 4 . Building Content based Model. The matrix consists of genres, starring and crew. people usually look for movies with same stars and screw
#%%
#left join movies data with credits
content_df = movies[['id', 'title', 'genres']].merge(credits, left_on='id', right_on='id', how='left')
#%%
#left join again with keywords dataset
content_df = content_df.merge(keywords, left_on='id', right_on='id', how='left')
#%%
content_df.head(3)
#%%
#create a new feature tag = genres + cast + crew + keywords
content_df['tag'] = content_df['genres'] + ' ' + content_df['cast'] + ' ' + content_df['crew'] + ' ' + content_df['keywords']
content_df = content_df.reset_index()
content_df = content_df.fillna("")
content_df.head(3)
#%%
# Build TF-IDF matrix with content_df
vectorizer = TfidfVectorizer(max_features=2000)
content_matrix = vectorizer.fit_transform(content_df['tag'])

#tfidf_df = pd.DataFrame(content_matrix.toarray(), index=movies.index.tolist())

#%%
# we have 2000 words in matrix
vectorizer.vocabulary_
#tfidf_df.head()
#movies[movies['id'] == 710]
#print(np.array(tfidf_df.loc(0)).reshape(1, -1))
#%%
"""
print(len(movies.index.tolist()))
print(len(vectorizer.get_feature_names()[1:20]))
print(content_matrix.shape)

"""
#%%
#content_matrix.shape
vectorizer.get_feature_names()[1:10]
#%%
print(movies.shape)
print(content_df.shape)
#%%
n_components = content_matrix.shape[1]
print("Best number for TSVD: {} ".format(n_components))
svd = TruncatedSVD(n_components=(n_components-1))
latent_content = svd.fit_transform(content_matrix)
print(latent_content.shape)

#%%
# plot variance and cumulative variance for content based matrix
expl = svd.explained_variance_ratio_
plt.plot(expl, '.-', ms=15)
plt.title('Variance');
plt.show()

plt.plot(np.cumsum(expl), '.-', ms=15)
plt.title('Cumulative Variance');
plt.show()
#%%
#print(vectorizer.get_feature_names()[:100])
#print(content_matrix.shape)
#print(movies.shape)
content_matrix_df = pd.DataFrame(latent_content, index=content_df.index.tolist())
content_matrix_df.to_csv("./dataresult/content_matrix.csv", index=True, header=True)
#%%
# Make few test with content based model
idx_movie = utility.get_idx_by_title(movies, "GoldenEye")
print("movie id {}: ".format(idx_movie))
features = ['title', 'genres','vote_average','score']
similarity.make_recommendation(idx_movie, content_matrix_df, movies, features, 5)

print("hybrid_model_recommendation.py done")
#%%
#!jupyter nbconvert --to PDF "RecommendUserSimilarity.ipynb"
#%% md

#%%
#rating_matrix[278]
"""



idx_movie = utility.get_idx_by_title(movies, "Grumpier Old Men")
print("movie id {}: ".format(idx_movie))

similarity.make_recommendation(idx_movie, latent_matrix_df, movies, features, 10)

"""

#%%
#rating_matrix[278]

"""



seed_movie = utility.get_idx_by_title(movies, "Father of the Bride Part II")
print("movie id {}: ".format(seed_movie))


similarity.make_recommendation(seed_movie, latent_matrix_df, movies, features, 10)

"""


#%% md
