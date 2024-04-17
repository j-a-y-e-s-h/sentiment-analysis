import string
import matplotlib.pyplot as plt
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Step 1: Data Collection
def read_text_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None


text = read_text_file("read.txt")
if text is None:
    exit()


# Step 2: Data Pre-processing and Labeling
def preprocess_text(text):
    lowercase = text.lower()
    clean_text = lowercase.translate(str.maketrans("", "", string.punctuation))
    return clean_text


def label_sentiments(text, emotion_dict):
    final_words = []
    sentiments = []
    for word in word_tokenize(text, "english"):
        if word not in stopwords.words("english"):
            final_words.append(word)
            sentiment = emotion_dict.get(word, "Neutral")
            sentiments.append(sentiment)
    return final_words, sentiments


emotion_dict = {}
with open("emotions.txt", "r") as file:
    for line in file:
        clear_line = line.strip().replace(",", "").replace("'", "")
        word, emotion = clear_line.split(":")
        emotion_dict[word] = emotion

clean_text = preprocess_text(text)
final_words, sentiments = label_sentiments(clean_text, emotion_dict)

# Step 3: Feature Extraction
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X = tfidf_vectorizer.fit_transform(final_words)

# Step 4: Model selection and training
X_train, X_test, y_train, y_test = train_test_split(
    X, sentiments, test_size=0.2, random_state=42
)
svm_model = LinearSVC(
    dual=True
)  # Explicitly setting dual to True to suppress FutureWarning
svm_model.fit(X_train, y_train)

# Step 5: Model Evaluation and Prediction
y_pred = svm_model.predict(X_test)
print(
    classification_report(y_test, y_pred, zero_division=1)
)  # Handling UndefinedMetricWarning


# Sentiment Analysis function
def analyze_sentiment(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    neg = score["neg"]
    pos = score["pos"]
    if neg > pos:
        return "Negative Sentiment"
    elif pos > neg:
        return "Positive Sentiment"
    else:
        return "Neutral Vibe"


# Analyzing sentiment of the whole text
sentiment = analyze_sentiment(clean_text)
print("Overall Sentiment:", sentiment)

# Plotting emotion distribution
word_count = Counter(sentiments)
fig, ax1 = plt.subplots()
ax1.bar(word_count.keys(), word_count.values())
fig.autofmt_xdate()
plt.show()


