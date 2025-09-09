import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from collections import Counter
import re
from wordcloud import WordCloud
import warnings
import requests
from bs4 import BeautifulSoup
import time
import random
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('vader_lexicon')

# Set up the page
st.set_page_config(
    page_title="Movie Review Sentiment Dashboard",
    page_icon="🎬",
    layout="wide"
)

# Initialize the VADER sentiment analyzer
@st.cache_resource
def get_analyzer():
    return SentimentIntensityAnalyzer()

# Function to fetch movie reviews (simulated for demonstration)
def fetch_movie_reviews(movie_name):
    """
    Simulates fetching reviews for a movie.
    In a real application, you would integrate with an API like IMDb, TMDB, or scrape reviews.
    """
    # Sample reviews for different movie categories
    sample_positive_reviews = [
        f"{movie_name} was an absolute masterpiece! The acting was superb and the storyline kept me engaged throughout.",
        f"I loved {movie_name}! The cinematography was stunning and the characters were well-developed.",
        f"{movie_name} is a must-watch. The director did an excellent job bringing this story to life.",
        f"Amazing performance by the cast of {movie_name}. The plot twists were unexpected but satisfying.",
        f"{movie_name} exceeded all my expectations. The visual effects were groundbreaking.",
        f"A perfect blend of action and emotion in {movie_name}. Highly recommended!",
        f"{movie_name} deserves all the awards. The screenplay was brilliant and engaging.",
        f"I was blown away by {movie_name}. The musical score complemented the scenes perfectly.",
        f"{movie_name} is a cinematic triumph. The character development was exceptional.",
        f"Outstanding direction in {movie_name}. Every scene was crafted with precision."
    ]
    
    sample_negative_reviews = [
        f"{movie_name} was a complete disappointment. The plot was predictable and the acting was subpar.",
        f"I regret watching {movie_name}. The storyline was weak and the characters were poorly developed.",
        f"{movie_name} failed to live up to the hype. The pacing was off and the dialogue was cringe-worthy.",
        f"Avoid {movie_name} at all costs. The director made questionable choices throughout the film.",
        f"{movie_name} was boring and uneventful. I struggled to stay awake during the screening.",
        f"The worst part of {movie_name} was the inconsistent character motivations. Nothing made sense.",
        f"{movie_name} had potential but wasted it on clichéd tropes and predictable plotlines.",
        f"I expected more from {movie_name}. The ending was particularly disappointing and rushed.",
        f"{movie_name} felt like a cheap imitation of better films in the same genre.",
        f"Poor execution in {movie_name}. The visual effects looked dated and unconvincing."
    ]
    
    sample_neutral_reviews = [
        f"{movie_name} was okay. Not great, but not terrible either. It had its moments.",
        f"{movie_name} is a decent watch if you have nothing else to do. Don't expect too much though.",
        f"I have mixed feelings about {movie_name}. Some parts were good, others were lacking.",
        f"{movie_name} is average at best. It follows a familiar formula without adding anything new.",
        f"Neither impressive nor disappointing - {movie_name} is just another movie in its genre.",
        f"{movie_name} had potential but played it too safe. The result is a middling experience.",
        f"I'm indifferent about {movie_name}. It didn't leave a lasting impression either way.",
        f"{movie_name} is forgettable. A week from now, I probably won't remember much about it.",
        f"Standard fare for this type of movie. {movie_name} doesn't take any risks.",
        f"{movie_name} is exactly what you'd expect - neither better nor worse than anticipated."
    ]
    
    # Generate a mix of reviews based on movie name (for simulation purposes)
    # This is a simple simulation - in a real app, you would fetch actual reviews
    reviews = []
    
    # More positive reviews for certain movie names
    if "avengers" in movie_name.lower() or "endgame" in movie_name.lower():
        reviews.extend(random.sample(sample_positive_reviews, 7))
        reviews.extend(random.sample(sample_negative_reviews, 2))
        reviews.extend(random.sample(sample_neutral_reviews, 1))
    elif "dark" in movie_name.lower() and "knight" in movie_name.lower():
        reviews.extend(random.sample(sample_positive_reviews, 8))
        reviews.extend(random.sample(sample_negative_reviews, 1))
        reviews.extend(random.sample(sample_neutral_reviews, 1))
    elif "twilight" in movie_name.lower():
        reviews.extend(random.sample(sample_positive_reviews, 3))
        reviews.extend(random.sample(sample_negative_reviews, 5))
        reviews.extend(random.sample(sample_neutral_reviews, 2))
    else:
        # Default mix
        reviews.extend(random.sample(sample_positive_reviews, 4))
        reviews.extend(random.sample(sample_negative_reviews, 3))
        reviews.extend(random.sample(sample_neutral_reviews, 3))
    
    random.shuffle(reviews)
    return reviews

# Function to analyze sentiment of a single text
def analyze_sentiment(text):
    analyzer = get_analyzer()
    scores = analyzer.polarity_scores(text)
    
    # Determine sentiment based on compound score
    if scores['compound'] >= 0.05:
        return 'Positive', scores
    elif scores['compound'] <= -0.05:
        return 'Negative', scores
    else:
        return 'Neutral', scores

# Function to clean text
def clean_text(text):
    # Handle NaN values
    if pd.isna(text):
        return ""
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    # Convert to lowercase
    text = text.lower()
    return text

# Function to extract most common words
def get_common_words(df, text_column, sentiment_filter=None, num_words=10):
    if sentiment_filter:
        filtered_df = df[df['Sentiment'] == sentiment_filter]
        all_text = " ".join(filtered_df[text_column].astype(str))
    else:
        all_text = " ".join(df[text_column].astype(str))
    
    words = all_text.split()
    # Remove common stopwords
    stopwords = set(['the', 'and', 'is', 'in', 'it', 'to', 'of', 'for', 'on', 'with', 'as', 'was', 'that', 'this', 'are', 'by', 'be', 'at', 'from', 'movie', 'film'])
    filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
    
    word_counts = Counter(filtered_words)
    return word_counts.most_common(num_words)

# Function to generate word cloud
def generate_wordcloud(text):
    if not text or text.isspace():
        return None
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

# Function to create sentiment distribution chart
def create_sentiment_chart(sentiment_counts):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#28a745', '#dc3545', '#ffc107']
    ax.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title('Sentiment Distribution')
    return fig

# Function to create rating distribution chart
def create_rating_chart(df):
    # Create a rating based on compound score (0-10 scale)
    df['Rating'] = (df['Compound_Score'] + 1) * 5  # Convert from -1 to 1 to 0 to 10 scale
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['Rating'], bins=10, range=(0, 10), edgecolor='black', alpha=0.7)
    ax.set_xlabel('Rating (0-10)')
    ax.set_ylabel('Number of Reviews')
    ax.set_title('Rating Distribution')
    ax.set_xticks(range(0, 11, 1))
    return fig

# Main app
def main():
    st.title("🎬 Movie Review Sentiment Dashboard")
    st.markdown("Enter a movie name to analyze its reviews and get a detailed sentiment analysis")
    
    # Movie name input
    movie_name = st.text_input("Enter movie name:", placeholder="e.g., The Dark Knight, Avengers: Endgame")
    
    if movie_name:
        with st.spinner(f"Fetching and analyzing reviews for '{movie_name}'..."):
            # Fetch reviews for the movie
            reviews = fetch_movie_reviews(movie_name)
            
            if not reviews:
                st.error("No reviews found for this movie. Please try another movie.")
                return
            
            # Create DataFrame
            df = pd.DataFrame({'Review': reviews})
            
            # Analyze each review
            sentiments = []
            positive_scores = []
            negative_scores = []
            neutral_scores = []
            compound_scores = []
            
            for review in df['Review']:
                sentiment, scores = analyze_sentiment(str(review))
                sentiments.append(sentiment)
                positive_scores.append(scores['pos'])
                negative_scores.append(scores['neg'])
                neutral_scores.append(scores['neu'])
                compound_scores.append(scores['compound'])
            
            # Add results to dataframe
            df['Sentiment'] = sentiments
            df['Positive_Score'] = positive_scores
            df['Negative_Score'] = negative_scores
            df['Neutral_Score'] = neutral_scores
            df['Compound_Score'] = compound_scores
            df['Cleaned_Text'] = df['Review'].apply(clean_text)
            
            # Calculate overall sentiment score
            overall_score = (df['Compound_Score'].mean() + 1) * 5  # Convert to 0-10 scale
            
            # Display results
            st.header(f"Analysis Results for '{movie_name}'")
            
            # Overall rating
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall Rating", f"{overall_score:.1f}/10")
            with col2:
                sentiment_counts = df['Sentiment'].value_counts()
                st.metric("Positive Reviews", f"{sentiment_counts.get('Positive', 0)}")
            with col3:
                st.metric("Total Reviews", len(df))
            
            # Sentiment distribution
            st.subheader("Sentiment Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                fig = create_sentiment_chart(sentiment_counts)
                st.pyplot(fig)
            
            with col2:
                fig = create_rating_chart(df)
                st.pyplot(fig)
            
            # Most common words
            st.subheader("Most Common Words by Sentiment")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Positive Reviews**")
                positive_words = get_common_words(df, 'Cleaned_Text', 'Positive', 8)
                for word, count in positive_words:
                    st.write(f"• {word.title()} ({count})")
            
            with col2:
                st.write("**Negative Reviews**")
                negative_words = get_common_words(df, 'Cleaned_Text', 'Negative', 8)
                for word, count in negative_words:
                    st.write(f"• {word.title()} ({count})")
            
            with col3:
                st.write("**All Reviews**")
                all_words = get_common_words(df, 'Cleaned_Text', None, 8)
                for word, count in all_words:
                    st.write(f"• {word.title()} ({count})")
            
            # Word clouds
            st.subheader("Word Clouds by Sentiment")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Positive Reviews Word Cloud**")
                positive_text = " ".join(df[df['Sentiment'] == 'Positive']['Cleaned_Text'].astype(str))
                if positive_text.strip():
                    fig = generate_wordcloud(positive_text)
                    if fig:
                        st.pyplot(fig)
                    else:
                        st.info("Not enough text to generate word cloud")
                else:
                    st.info("No positive reviews to generate word cloud")
            
            with col2:
                st.write("**Negative Reviews Word Cloud**")
                negative_text = " ".join(df[df['Sentiment'] == 'Negative']['Cleaned_Text'].astype(str))
                if negative_text.strip():
                    fig = generate_wordcloud(negative_text)
                    if fig:
                        st.pyplot(fig)
                    else:
                        st.info("Not enough text to generate word cloud")
                else:
                    st.info("No negative reviews to generate word cloud")
            
            # Sample reviews
            st.subheader("Sample Reviews")
            
            tab1, tab2, tab3 = st.tabs(["Positive", "Negative", "Neutral"])
            
            with tab1:
                positive_reviews = df[df['Sentiment'] == 'Positive']['Review'].head(3)
                for i, review in enumerate(positive_reviews, 1):
                    st.write(f"{i}. {review}")
            
            with tab2:
                negative_reviews = df[df['Sentiment'] == 'Negative']['Review'].head(3)
                for i, review in enumerate(negative_reviews, 1):
                    st.write(f"{i}. {review}")
            
            with tab3:
                neutral_reviews = df[df['Sentiment'] == 'Neutral']['Review'].head(3)
                for i, review in enumerate(neutral_reviews, 1):
                    st.write(f"{i}. {review}")
            
            # Download button
            st.download_button(
                label="Download Analysis Results",
                data=df.to_csv(index=False),
                file_name=f"{movie_name.replace(' ', '_')}_sentiment_analysis.csv",
                mime="text/csv"
            )
    
    else:
        st.info("Please enter a movie name to see its review analysis dashboard.")
        
        # Show sample movie suggestions
        st.subheader("Try these popular movies:")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("The Dark Knight"):
                st.experimental_set_query_params(movie="The Dark Knight")
                st.experimental_rerun()
        with col2:
            if st.button("Avengers: Endgame"):
                st.experimental_set_query_params(movie="Avengers: Endgame")
                st.experimental_rerun()
        with col3:
            if st.button("The Shawshank Redemption"):
                st.experimental_set_query_params(movie="The Shawshank Redemption")
                st.experimental_rerun()
        with col4:
            if st.button("The Godfather"):
                st.experimental_set_query_params(movie="The Godfather")
                st.experimental_rerun()

    # Add some information about the app
    with st.expander("About this app"):
        st.markdown("""
        This **Movie Review Sentiment Dashboard** analyzes reviews for any movie and provides detailed sentiment analysis.
        
        **How it works:**
        - Enter a movie name to fetch and analyze its reviews
        - The app uses NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner) for sentiment analysis
        - Each review is analyzed and classified as Positive, Negative, or Neutral
        - The app provides visualizations including sentiment distribution, word clouds, and common words
        
        **Sentiment Classification:**
        - **Positive**: Compound score ≥ 0.05
        - **Negative**: Compound score ≤ -0.05
        - **Neutral**: Compound score between -0.05 and 0.05
        
        The dashboard helps you understand general audience reception of any movie.
        """)

if __name__ == "__main__":
    main()