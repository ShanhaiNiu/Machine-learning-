import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import scr_matrix
import helper
from sklearn.cluster import KMeans

# import movies dataset
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')

genre_ratings = helper.get_genre_ratings(ratings, movies, ['Romance', 'Sci-Fi'], ['avg_romance_rating', 'avg_scifi_rating'])
genre_ratings.head()

biased_dataset = helper.bias_genre_rating_dataset(genre_ratings, 3.2, 2.5)

print( "Number of records: ", len(biased_dataset))
biased_dataset.head()

get_ipython().run_line_magic('matplotlib', 'inline')

helper.draw_scatterplot(biased_dataset['avg_scifi_rating'],'Avg scifi rating', biased_dataset['avg_romance_rating'], 'Avg romance rating')

# use k-means 

kmeans_1 = KMeans(n_cluster=2)
predictions = kmeans_1.fit_predict(X)
helper.draw_clusters(biased_dataset, predictions)

kmeans_2 = KMeans(n_cluster=3)
predictions_2 = kmeans_2.fit_predict(X)
helper.draw_clusters(biased_dataset, predictions_2)

kmeans_3 = KMeans(n_cluster=3)
predictions_3 = kmeans_3.fit_predict(X)
helper.draw_clusters(biased_dataset, predictions_3)

df = biased_dataset[['avg_scifi_rating','avg_romance_rating']]

# Choose the range of k values to test.
# add a stride of 5 to improve performance.
possible_k_values = range(2, len(X)+1, 5)

errors_per_k = [helper.clustering_errors(k, X) for k in possible_k_values]

list(zip(possible_k_values, errors_per_k))

kmeans_4 = KMeans(n_clusters=7)

# fit_predict to cluster the dataset
predictions_4 = kmeans_4.fit_predict(X)

# Plot
helper.draw_clusters(biased_dataset, predictions_4, cmap='Accent')

biased_dataset_3_genres = helper.get_genre_ratings(ratings, movies, 
                                                     ['Romance', 'Sci-Fi', 'Action'], 
                                                     ['avg_romance_rating', 'avg_scifi_rating', 'avg_action_rating'])
biased_dataset_3_genres = helper.bias_genre_rating_dataset(biased_dataset_3_genres, 3.2, 2.5).dropna()

print( "Number of records: ", len(biased_dataset_3_genres))
biased_dataset_3_genres.head()


X_with_action = biased_dataset_3_genres[['avg_scifi_rating',
                                                           'avg_romance_rating', 
                                                           'avg_action_rating']].values
														   
kmeans_5 = KMeans(n_clusters=7)

predictions_5 = kmeans_5.fit_predict(X_with_action)
helper.draw_clusters_3d(biased_dataset_3_genres, predictions_5)

ratings_title = pd.merge(ratings, movies[['movieId', 'title']], on='movieId' )
user_movie_ratings = pd.pivot_table(ratings_title, index='userId', columns= 'title', values='rating')

print('dataset dimensions: ', user_movie_ratings.shape, '\n\nSubset example:')
user_movie_ratings.iloc[:6, :10]														   
n_movies = 30
n_users = 18
most_rated_movies_users_selection = helper.sort_by_rating_density(user_movie_ratings, n_movies, n_users)

print('dataset dimensions: ', most_rated_movies_users_selection.shape)
most_rated_movies_users_selection.head()

user_movie_ratings =  pd.pivot_table(ratings_title, index='userId', columns= 'title', values='rating')
most_rated_movies_1k = helper.get_most_rated_movies(user_movie_ratings, 1000)

sparse_ratings = csr_matrix(pd.SparseDataFrame(most_rated_movies_1k).to_coo())

predictions = KMeans(n_clusters=20, algorithm='full').fit_predict(sparse_ratings)

max_users = 70
max_movies = 50

clustered = pd.concat([most_rated_movies_1k.reset_index(), pd.DataFrame({'group':predictions})], axis=1)
helper.draw_movie_clusters(clustered, max_users, max_movies)

cluster_number = 11

n_users = 75
n_movies = 300
cluster = clustered[clustered.group == cluster_number].drop(['index', 'group'], axis=1)

cluster = helper.sort_by_rating_density(cluster, n_movies, n_users)
helper.draw_movies_heatmap(cluster, axis_labels=False)

cluster.fillna('').head()

movie_name = "Forrest Gump (1994)"

cluster[movie_name].mean()
cluster.mean().head(20)

user_id = 19

# Get all this user's ratings
user_2_ratings  = cluster.loc[user_id, :]

# Which movies did they not rate? 
user_2_unrated_movies =  user_2_ratings[user_2_ratings.isnull()]

# What are the ratings of these movies the user did not rate?
avg_ratings = pd.concat([user_2_unrated_movies, cluster.mean()], axis=1, join='inner').loc[:,0]

# sort by rating so the highest rated movies are presented first
avg_ratings.sort_values(ascending=False)[:20]







