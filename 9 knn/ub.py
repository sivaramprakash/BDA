import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

cols = ['userId', 'movieId', 'rating', 'timestamp']
df = pd.read_csv('../dataset/ratings.csv')

df['userId'] = pd.to_numeric(df['userId'], errors='coerce')
df['movieId'] = pd.to_numeric(df['movieId'], errors='coerce')
df = df.dropna()

user_item_matrix = df.pivot(index='userId', columns='movieId', values='rating').fillna(0)

user_sim = cosine_similarity(user_item_matrix)
user_sim_df = pd.DataFrame(user_sim, index=user_item_matrix.index, columns=user_item_matrix.index)

def recommend(target_user, k=10, num_recs=5):
    if target_user not in user_sim_df.index:
        return "User ID not found in dataset"
    
    similar_users = user_sim_df[target_user].sort_values(ascending=False)[1:k+1]
    user_ratings = user_item_matrix.loc[target_user]
    unwatched_movies = user_ratings[user_ratings == 0].index
    
    neighbor_ratings = user_item_matrix.loc[similar_users.index, unwatched_movies]
    weights = similar_users.values.reshape(-1, 1)
    
    weighted_scores = (neighbor_ratings * weights).sum(axis=0) / (weights.sum() + 1e-9)
    return weighted_scores.sort_values(ascending=False).head(num_recs)

def evaluate_model(sample_size=100):
    test_set = df.sample(sample_size)
    actuals = []
    predictions = []
    
    for _, row in test_set.iterrows():
        u, m = int(row.userId), int(row.movieId)
        neighbors = user_sim_df[u].sort_values(ascending=False)[1:11].index
        pred = user_item_matrix.loc[neighbors, m].mean()
        actuals.append(row.rating)
        predictions.append(pred if pred > 0 else 3.0)
        
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    print(f"RMSE: {rmse:.4f}")

print("Recommendations for User 1:")
print(recommend(target_user=1))
evaluate_model()