import streamlit as st
from textblob import TextBlob
from nltk.tokenize import sent_tokenize
from nltk import download
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import plotly.express as px

# Download necessary NLTK data
download('punkt')

# Initialize VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Streamlit App
st.title("Sentiment Analysis App")

# Text input for user
user_text = st.text_area("Paste your text here for analysis:")

# Choose Sentiment Analysis Method
analysis_method = st.selectbox(
    "Choose sentiment analysis method:",
    ["TextBlob", "VADER"]
)

# Option to tokenize into sentences or analyze the whole text
tokenize_option = st.checkbox("Separate by sentences")

if st.button("Analyze"):
    if user_text.strip() == "":
        st.warning("Please paste some text for analysis.")
    else:
        if tokenize_option:
            # Tokenize into sentences
            sentences = sent_tokenize(user_text)
            sentiment_results = []

            for sentence in sentences:
                if analysis_method == "TextBlob":
                    blob = TextBlob(sentence)
                    sentiment = blob.sentiment
                    sentiment_results.append({
                        'sentence': sentence,
                        'polarity': sentiment.polarity,
                        'subjectivity': sentiment.subjectivity
                    })
                elif analysis_method == "VADER":
                    scores = vader_analyzer.polarity_scores(sentence)
                    sentiment_results.append({
                        'sentence': sentence,
                        'polarity': scores['compound'],
                        'subjectivity': "N/A"
                    })

            # Convert to DataFrame
            df = pd.DataFrame(sentiment_results)

            # Display DataFrame
            st.write("Sentiment Analysis Results:")
            st.dataframe(df)

            # Plot if tokenized
            fig = px.line(
                df, 
                x=df.index, 
                y='polarity', 
                title='Sentiment Polarity Change Over Sentences',
                labels={'index': 'Sentence Index', 'polarity': 'Polarity'},
                hover_data=['sentence'],
                template='plotly_dark'
            )
            st.plotly_chart(fig)
        else:
            # Analyze entire text
            if analysis_method == "TextBlob":
                blob = TextBlob(user_text)
                sentiment = blob.sentiment
                st.write("Overall Sentiment Analysis (TextBlob):")
                st.write(f"Polarity: {sentiment.polarity}")
                st.write(f"Subjectivity: {sentiment.subjectivity}")
            elif analysis_method == "VADER":
                scores = vader_analyzer.polarity_scores(user_text)
                st.write("Overall Sentiment Analysis (VADER):")
                st.write(f"Polarity (compound score): {scores['compound']}")
                st.write(f"Positive: {scores['pos']}, Negative: {scores['neg']}, Neutral: {scores['neu']}")

# cd "/Users/arjunghumman/Downloads/VS Code Stuff/Python/sentiment App"
# streamlit run sentiment.py