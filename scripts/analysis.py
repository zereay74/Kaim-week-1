from imports import pd, plt
from dateutil import parser
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

def check_missing_value(data):
    missing_summary = data.isnull().sum()
    print('Missing Value Summary')
   
    return missing_summary


def headline_length_check(data, column):
    data['headline_length'] = data[column].astype(str).apply(len)
    print('Headline Length Statistics')
    
    return (data['headline_length'].describe())

def count_and_sort(data, column):
    return data[column].value_counts().sort_values()

# convert date
def convert_date(data, column):
    def parse_mixed_dates(date_str):
        try:
            # Use dateutil parser for flexible parsing
            dt = parser.parse(date_str)
            # Localize naive datetimes to UTC
            if dt.tzinfo is None:
                return dt.replace(tzinfo=pd.Timestamp(0).tzinfo)
            return dt
        except Exception:
            return pd.NaT  # Return NaT for invalid dates

    # Apply the custom parser to the 'date' column
    data[column] = data[column].apply(parse_mixed_dates)
    # Ensure all dates are datetime objects and convert to UTC
    data[column] = pd.to_datetime(data[column], utc=True)
    return data.head(12)
    

def extract_date(data, column):
    
    # Extract time features
    data['year'] = data[column].dt.year
    data['month'] = data[column].dt.month
    data['day'] = data[column].dt.day
    data['day_of_week'] = data[column].dt.day_name()

    # Return the extracted features
    return data[[column, 'year', 'month', 'day', 'day_of_week']].head()


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def perform_sentiment_analysis(data, text_column):
    """
    Perform sentiment analysis on a text column using VADER.

    Args:
        data (pd.DataFrame): The dataset.
        text_column (str): The name of the column containing text data.

    Returns:
        pd.DataFrame: Dataframe with sentiment scores and sentiment labels.
    """
    analyzer = SentimentIntensityAnalyzer()
    
    # Apply sentiment analysis
    sentiments = data[text_column].apply(analyzer.polarity_scores)
    sentiment_df = sentiments.apply(pd.Series)
    
    # Add sentiment label based on compound score
    sentiment_df['sentiment_label'] = sentiment_df['compound'].apply(
        lambda x: 'positive' if x > 0.05 else 'negative' if x < -0.05 else 'neutral'
    )
    
    # Combine with the original dataset
    data_with_sentiment = pd.concat([data, sentiment_df], axis=1)
    
    return data_with_sentiment



def nlp_keyword_topic_analysis(data, text_column, n_topics=5, n_top_words=10, ngram_range=(2, 3), tfidf_features=20):
    """
    Perform NLP-based keyword extraction and topic modeling on a text column in a DataFrame.

    Parameters:
    - data (pd.DataFrame): The input DataFrame containing text data.
    - text_column (str): The name of the column containing the text.
    - n_topics (int): Number of topics to extract using LDA.
    - n_top_words (int): Number of top words to display per topic.
    - ngram_range (tuple): Range of n-grams to extract (e.g., (2, 3) for bi- and tri-grams).
    - tfidf_features (int): Number of top features to extract using TF-IDF.

    Returns:
    - None
    """
    # Load Spacy Model
    nlp = spacy.load('en_core_web_sm')

    # Text Preprocessing Function
    def preprocess_text(text):
        doc = nlp(text.lower())
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return ' '.join(tokens)

    # Apply Preprocessing
    data['cleaned_text'] = data[text_column].apply(preprocess_text)

    # N-Gram Extraction
    vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=tfidf_features)
    ngrams = vectorizer.fit_transform(data['cleaned_text'])
    ngram_features = vectorizer.get_feature_names_out()

    print("Common Phrases (N-Grams):")
    print(ngram_features)

    # TF-IDF Extraction
    tfidf = TfidfVectorizer(max_features=tfidf_features)
    tfidf_matrix = tfidf.fit_transform(data['cleaned_text'])
    tfidf_features = tfidf.get_feature_names_out()

    print("Top Keywords (TF-IDF):")
    print(tfidf_features)

    # Topic Modeling with LDA
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(tfidf_matrix)

    print("\nTop Topics with Keywords:")
    for idx, topic in enumerate(lda.components_):
        print(f"Topic {idx + 1}:")
        print([tfidf_features[i] for i in topic.argsort()[:-n_top_words - 1:-1]])

    # Word Cloud for Keywords
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(data['cleaned_text']))
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Common Keywords")
    plt.show()

# Example Usage
# data = pd.DataFrame({'headlines': ["FDA approves new drug for diabetes", "Stock market hits record highs", "Company reports Q4 earnings", "Analyst raises price target"]})
# nlp_keyword_topic_analysis(data, text_column='headlines')



def extract_keywords(data, text_column, max_features=20):
    """
    Extract common keywords from a specified text column.

    Parameters:
    - data (pd.DataFrame): The input DataFrame containing text data.
    - text_column (str): The name of the column containing the text.
    - max_features (int): Maximum number of keywords to extract.

    Returns:
    - keywords (list): A list of the most common keywords.
    """
    vectorizer = CountVectorizer(stop_words='english', max_features=max_features)
    X = vectorizer.fit_transform(data[text_column])
    keywords = vectorizer.get_feature_names_out()
    return keywords

# publisher analysis
def analyze_top_publishers(data, publisher_column, n=10):
    """
    Analyze top publishers and their contribution to the news feed.

    Parameters:
    - data (pd.DataFrame): The input dataset.
    - publisher_column (str): The name of the column containing publisher names.
    - n (int): Number of top publishers to analyze.

    Returns:
    - publisher_counts (pd.Series): Number of articles per publisher.
    """
    publisher_counts = data[publisher_column].value_counts().head(n)
    return publisher_counts

def classify_article_topic(data, text_column):
    """
    Classify articles into topics based on keywords.
    
    Parameters:
    - data (pd.DataFrame): The input dataset.
    - text_column (str): Column containing the article text or headlines.

    Returns:
    - data (pd.DataFrame): The dataset with an additional 'topic' column.
    """
    # Define keywords for topics
    topics = {
        'finance': ['market', 'stock', 'trade', 'investment'],
        'health': ['health', 'drug', 'FDA', 'virus'],
        'technology': ['technology', 'AI', 'tech', 'innovation']
    }

    def assign_topic(text):
        for topic, keywords in topics.items():
            if any(keyword.lower() in text.lower() for keyword in keywords):
                return topic
        return 'other'

    data['topic'] = data[text_column].apply(assign_topic)
    return data
