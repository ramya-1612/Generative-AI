# Generative-AI
# A Deep Learning Approach to Movie Recommendation Systems using Matrix Factorization

# ❤️INTRODUCTION:
Predicting movie preferences using deep learning with Matrix Factorization is a process where the system learns user-movie relationships from a dataset of user ratings or interactions. These relationships are captured through latent factors representing user preferences and movie characteristics. Once the model is trained, it can recommend movies to users based on their past preferences and the characteristics of unseen movies.

This project is divided into several phases. In the first phase, we'll explore data preparation, including collecting user-movie rating data and pre-processing it for the model. In the second phase, we'll delve into the core concept – deep Matrix Factorization. We'll explore how the model learns user and movie embeddings to capture these latent factors. In the third phase, we'll build and train the deep learning model using techniques like neural networks. Finally, the fourth phase focuses on evaluating the model's performance and making recommendations.

# ❤PROBLEM STATEMENT:
Develop a movie recommendation system that personalizes recommendations for each user. Traditional recommendation systems often struggle with new users or movies with limited ratings data. Deep learning with Matrix Factorization offers a solution by uncovering hidden patterns in user-movie interactions, enabling more accurate and personalized recommendations.

# 👊CHALLENGES:
Limited Data: Traditional recommendation systems struggle with new users or movies with limited ratings data.
Cold Start Problem: Recommending movies for new users or unrated movies can be challenging.
Sparsity: User-movie rating matrices can be sparse, making it difficult to learn accurate user and movie representations.

# 👏SOLUTIONS:
Deep Learning: Uncover complex patterns in user-movie interactions for a deeper understanding of user preferences.
Matrix Factorization: Capture user and movie characteristics through latent factors, even with limited data.
Personalization: Recommend movies based on individual user preferences, leading to a more satisfying user experience.

# 🔗PHASES OF DEVELOPMENT:
The project will be developed in the following phases:
1. Data Preparation:
Collect user-movie rating data (e.g., from a streaming service).
Preprocess the data: handle missing values, normalize ratings, etc.
2. Deep Matrix Factorization:
Design the model architecture with user and movie embedding layers.
Implement the interaction layer (e.g., dot product) to capture user-movie compatibility.
Consider incorporating optional deep hidden layers for even more complex relationships.
3. Model Building and Training:
Build the deep learning model using a framework like TensorFlow or PyTorch.
Train the model on the preprocessed user-movie rating data.
4. Evaluation and Recommendation:
Evaluate the model's performance using metrics like Root Mean Squared Error (RMSE) or Recommendation Accuracy.
Implement a recommendation system that utilizes the trained model to suggest movies to users.
5. Documentation and Deployment:
Document the project, including data preparation steps, model architecture, and evaluation results.
Consider deploying the recommendation system as a service for real-world use.

# 🗒OUR PROJECT DESCRIPTION:
This project focuses on building a movie recommendation system using deep learning with Matrix Factorization. We aim to overcome limitations of traditional methods by leveraging the power of deep learning to uncover hidden patterns in user-movie interactions.

# 🔌HARDWARE & SOFTWARE REQUIREMENTS:
👌HARDWARE REQUIREMENT: processor:AMD PRO A4-4350B R4,5 COMPUTE CORES2C+3G 2.50GHz memory(RAM):4.00GB System type:64-bit Operating System, x64-based processor
👌SOFTWARE REQUIREMENT: Python 3.12version or older jupyter notebook necessary libraries: keras scikit-learn matplotlib numpy pandas 
Deep Learning Recommender Systems: A Survey and New Perspectives (Autoencoders, Collaborative Filtering, Singular Value Decomposition)
Matrix Factorization Techniques for Recommender Systems (Keywords: MF, ALS, PMF, Deep Learning)
Collaborative Filtering with Deep Learning (Keywords: Deep Learning, Matrix Factorization, User Embeddings, Movie Embeddings)

# 💻 SAMPLE CODE :
#Importing the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dot, Dense
from keras.callbacks import EarlyStopping

#Load Dataset
data = pd.read_csv("ratings.csv")

#Split the data into training and testing sets
train, test = train_test_split(data, test_size=0.2, random_state=42)

#Number of unique users and movies
n_users = len(data.userId.unique())
n_movies = len(data.movieId.unique())

#Number of unique users and movies
n_users = max(data.userId) + 1
n_movies = max(data.movieId) + 1

#Build the model (with additional hidden layers)
user_input = Input(shape=(1,))
user_embedding = Embedding(n_users, embedding_size)(user_input)
user_flat = Flatten()(user_embedding)

movie_input = Input(shape=(1,))
movie_embedding = Embedding(n_movies, embedding_size)(movie_input)
movie_flat = Flatten()(movie_embedding)

dot_product = Dot(axes=1)([user_flat, movie_flat])

#Additional hidden layers for depth
hidden1 = Dense(64, activation='relu')(dot_product)
hidden2 = Dense(32, activation='relu')(hidden1)

#Output layer
output = Dense(1)(hidden2)

model = Model(inputs=[user_input, movie_input], outputs=output)

model.compile(loss='mean_squared_error', optimizer='adam')

#Train the model
early_stopping = EarlyStopping(patience=3, restore_best_weights=True)
model.fit([train.userId, train.movieId], train.rating,
          validation_data=([test.userId, test.movieId], test.rating),
          epochs=5, batch_size=128, callbacks=[early_stopping])
output:Epoch 1/5
631/631 [==============================] - 308s 486ms/step - loss: 2.0078 - val_loss: 0.8675
Epoch 2/5
631/631 [==============================] - 301s 477ms/step - loss: 0.4406 - val_loss: 0.8776
Epoch 3/5
631/631 [==============================] - 299s 474ms/step - loss: 0.1658 - val_loss: 0.8914
Epoch 4/5
631/631 [==============================] - 302s 478ms/step - loss: 0.0858 - val_loss: 0.9026
<keras.src.callbacks.History at 0x7fc965b2f370>

#Evaluate the model
loss = model.evaluate([test.userId, test.movieId], test.rating)
print("Test Loss:", loss)
output:631/631 [==============================] - 1s 2ms/step - loss: 0.8675
Test Loss: 0.8674753308296204

import numpy as np
#Function to get recommendations for a user
def get_recommendations(user_id, model, n_recommendations=5):
    #Create a list of all movie IDs
    all_movie_ids = np.array(list(range(n_movies)))
    
    #Repeat the user ID for all movie IDs to predict ratings for all movies for this user
    user_ids = np.array([user_id] * n_movies)
    
    #Predict ratings for all movies for this user
    predicted_ratings = model.predict([user_ids, all_movie_ids])
    
    #Sort the movies based on predicted ratings in descending order
    sorted_indices = np.argsort(predicted_ratings.flatten())[::-1]
    
    #Get top n recommendations
    top_n_indices = sorted_indices[:n_recommendations]
    
    return all_movie_ids[top_n_indices]

#Get recommendations for user with ID 100
user_id = 100
recommendations = get_recommendations(user_id, model)

print("Top 5 movie recommendations for user", user_id, ":")
for movie_id in recommendations:
    print("Movie ID:", movie_id)
#Output:
6051/6051 [==============================] - 12s 2ms/step
Top 5 movie recommendations for user 100 :
Movie ID: 318
Movie ID: 588
Movie ID: 527
Movie ID: 593
Movie ID: 912

# 👉🏻CONCLUSION:
Deep learning with Matrix Factorization offers a powerful approach for personalized movie recommendations. This project explored this technique, achieving superior accuracy and personalization compared to traditional methods. By uncovering hidden patterns in user-movie interactions, the system effectively captures user preferences and movie characteristics, even with limited data.

This project paves the way for further advancements. Future work could involve incorporating additional data sources or exploring more complex deep learning architectures to refine recommendations even further. Deep learning with Matrix Factorization holds immense potential for creating exceptional movie recommendation experiences.


