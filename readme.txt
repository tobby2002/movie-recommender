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
  
 Data/Files Notes:
  - metadata_movies.csv, ratings.csv, credits.csv and keywords can be found on Kaggle.com
  - normalised_movies.csv, ratings_matrix.csv and content_matrix.csv are generated when training model: HybridRecommendationModel.ipynb
  - Those files are the input for HydridModelEvaluation.ipynb
  - recommender.py consist of 2 python classes: 
      - class Utility: help for cleaning data, showing top movies with right format, looking for id/index of movie
      - class Similarity: compute the cosine similariy between sparse matrix (ratings_matrix.csv or content_matrix.csv) and given movie
  - Because of large datasets, all note books are run on Colab. If you want to run them on local, check out the below "Run on Local"
  
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
 - Clone the project: by downloading or git command: git clone https://github.com/diem-ai/movie-recommender.git
 - Install libraries in Requirements and Dependencies
 - Comment the colab set and change the file path
 
 Credtis:
 Datasets are downloaded on Kaggle.com
