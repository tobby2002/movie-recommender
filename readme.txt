Hydrid Movie Recommendation system:
project structure:
   |___data
          |___ normalised_movies.csv
          |___ ratings_matrix.csv
          |___ content_matrix.csv
          |___ metadata_movies.csv
          |___ ratings.csv
          |___ credits.csv
          |___ keywords.csv
   |___ HybridRecommendationModel.ipynb
   |___ HydridModelEvaluation.ipynb
   |___ recommender.py
  
 Data Notes:
  - metadata_movies.csv, ratings.csv, credits.csv and keywords can be found on Kaggle.com
  - normalised_movies.csv, ratings_matrix.csv and content_matrix.csv are generated when training model: HybridRecommendationModel.ipynb
  - Those files are the input for HydridModelEvaluation.ipynb
 
 Requirements:
  - Python >= 3.7
  - Jupyter Notebook
 
 Dependencies:
  - pandas
  - sklearn
  - numpy
  - ast
  - matplotlib
  - nltk
  
Run the projecy on local:
 - clone the project: by downloading or git command: git clone https://github.com/diem-ai/movie-recommender.git
 - install libraries in Requirements and Dependencies
 
 Credtis:
 Datasets are downloaded on Kaggle.com
