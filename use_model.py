import joblib
model = joblib.load("genre_prediction_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

#Put the movie title and description here
new_data = ["Rod and Carolyn find their pet dog dead under mysterious circumstances and experience a spirit that harms their daughter Andrea. They finally call investigators who can help them get out of the mess"]
new_data_tfidf = vectorizer.transform(new_data)

predicted_genre = model.predict(new_data_tfidf)
print(predicted_genre)
