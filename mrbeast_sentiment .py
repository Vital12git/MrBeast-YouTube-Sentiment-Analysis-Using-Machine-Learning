# MrBeast YouTube Sentiment Analysis
# Algorithm: TF-IDF + Logistic Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset
file_path = r"C:\Users\manda\OneDrive\Desktop\ML Project\sentiment_analysis_dataset.csv"

try:
    data = pd.read_csv(
        file_path,
        encoding="utf-8-sig",
        engine="python",
        on_bad_lines="skip"
    )
    print("Dataset loaded successfully!")
except Exception as e:
    print("Error loading CSV:", e)
    raise SystemExit

# 2. Clean column names
data.columns = data.columns.astype(str).str.strip()
print("Dataset columns:", data.columns.tolist())

if "Comment" not in data.columns or "Sentiment" not in data.columns:
    print("ERROR: Comment or Sentiment column not found.")
    raise SystemExit

# 3. Clean dataset
print("\nRows before cleaning:", len(data))

data = data[["Comment", "Sentiment"]].copy()
data = data.dropna(subset=["Comment", "Sentiment"])

data["Comment"] = data["Comment"].astype(str).str.strip()
data["Comment"] = data["Comment"].str.replace(r'http\S+|www\.\S+', '', regex=True)
data["Sentiment"] = data["Sentiment"].astype(str).str.strip().str.capitalize()

# Remove empty comments and labels
data = data[
    (data["Comment"] != "") &
    (data["Sentiment"] != "")
]

# Remove duplicate rows
data = data.drop_duplicates()

# Keep valid sentiment labels
data = data[
    data["Sentiment"].isin(["Positive", "Negative", "Neutral"])
].copy()

# Save cleaned dataset
cleaned_path = r"C:\Users\manda\OneDrive\Desktop\ML Project\cleaned_sentiment_dataset.csv"
data.to_csv(cleaned_path, index=False, encoding="utf-8-sig")

print("Rows after cleaning:", len(data))
print("Cleaned dataset saved at:", cleaned_path)

# 4. Calculate sentiment percentages
total_comments = len(data)

print("\n===== SENTIMENT PERCENTAGE ANALYSIS =====")
print("Total comments:", total_comments)

if total_comments > 0:
    sentiment_counts = data["Sentiment"].value_counts()

    for sentiment in ["Positive", "Negative", "Neutral"]:
        count = sentiment_counts.get(sentiment, 0)
        percentage = (count / total_comments) * 100
        print(f"{sentiment}: {count} comments ({percentage:.2f}%)")
else:
    print("No usable comments found.")
    raise SystemExit

# 5. Prepare input and output
X = data["Comment"]
y = data["Sentiment"]

if len(data) < 5 or y.nunique() < 2:
    print("ERROR: Not enough data or sentiment classes to train the model.")
    raise SystemExit

# 6. Split training and testing data
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
except ValueError:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

# 7. Create the model
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        max_features=10000
       
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

# 8. Train the model
model.fit(X_train, y_train)
print("\nModel trained successfully!")

# 9. Evaluate the model
predictions = model.predict(X_test)

print("\n===== MODEL RESULTS =====")
print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test, predictions, zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# 10. Predict sentiment for a new comment
print("\n===== NEW COMMENT PREDICTION =====")
comment = input("Enter a YouTube comment: ").strip()

if comment:
    result = model.predict([comment])[0]
    print("Predicted sentiment:", result)
else:
    print("No comment entered.")