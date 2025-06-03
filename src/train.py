from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from src.data.load_data import load_and_clean_data
from src.models.text_classifier import build_pipeline

def main():
    df = load_and_clean_data("data/reviews.csv")
    reviews = df["Review"]
    y = df["Sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(reviews, y, test_size=0.2, random_state=42)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

  
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()