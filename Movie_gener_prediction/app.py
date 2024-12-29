import streamlit as st
import joblib
import string
import plotly.express as px
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Load the trained SVM model
svm_model = joblib.load('svm_model.pkl')

# Simulating the genre distribution in raw_data (replace with your actual data)
data = {
    'Genre': [
        'thriller', 'comedy', 'documentary', 'drama', 'horror', 'short', 'western', 'family',
        'sport', 'romance', 'war', 'game-show', 'biography', 'adult', 'talk-show', 'action', 
        'music', 'crime', 'animation', 'sci-fi', 'adventure', 'reality-tv', 'fantasy', 'mystery',
        'history', 'news', 'musical'
    ]
}
raw_data = pd.DataFrame(data)

# Get the genre counts for the plot
genre_counts = raw_data['Genre'].value_counts().reset_index()
genre_counts.columns = ['Genre', 'Count']

# Plot genre distribution with Plotly
fig = px.bar(genre_counts, x='Genre', y='Count',
             color='Count',
             title='Genre Distribution',
             labels={'Genre': 'Genre', 'Count': 'Number of Movies'},
             color_continuous_scale='Viridis')

fig.update_layout(xaxis_title='Genre', yaxis_title='Count', xaxis=dict(tickangle=-45))

# Streamlit UI
st.title("Movie Genre Classification")
st.write("This app predicts the genre of a movie based on its description.")

# Show genre list
st.subheader("Genres in the Dataset")
genres = raw_data['Genre'].unique()
st.write("Available genres:", ', '.join(genres))

# Input field for movie description
movie_description = st.text_area("Enter Movie Description", "")

# Recreate the vectorizer and label encoder (must match the original configuration)
vectorizer = TfidfVectorizer(stop_words='english')  # Same configuration used during training

# Manually define the label encoder with the known genres from your dataset
label_encoder = LabelEncoder()
label_encoder.fit([
    'thriller', 'comedy', 'documentary', 'drama', 'horror', 'short', 'western', 'family',
    'sport', 'romance', 'war', 'game-show', 'biography', 'adult', 'talk-show', 'action', 
    'music', 'crime', 'animation', 'sci-fi', 'adventure', 'reality-tv', 'fantasy', 'mystery',
    'history', 'news', 'musical'
])

# Prediction function
def predict_genre(description):
    # Preprocess the input description (same as during training)
    description_processed = description.lower()  # Lowercase the text
    description_processed = description_processed.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation

    # Vectorize the description using the recreated vectorizer
    description_vectorized = vectorizer.fit_transform([description_processed])

    # Predict the genre using the trained SVM model
    genre_predicted = svm_model.predict(description_vectorized)

    # Decode the genre using label encoder
    genre = label_encoder.inverse_transform(genre_predicted)

    return genre[0]

# When the user clicks "Classify Genre"
if st.button("Classify Genre"):
    if movie_description:
        predicted_genre = predict_genre(movie_description)
        st.write(f"Predicted Genre: {predicted_genre}")
    else:
        st.write("Please enter a movie description.")

# Display the genre distribution plot
st.plotly_chart(fig)
