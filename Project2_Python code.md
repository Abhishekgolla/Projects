# Project2_movie_success_prediction: Used Python Code
# STEP 1: Create Folder Structure
# ==============================================
base_dir = "Movie_Success_Prediction_Project"
folders = ["data", "models", "visuals", "report"]
for f in folders:
    os.makedirs(os.path.join(base_dir, f), exist_ok=True)

# ==============================================
# STEP 2: Create Sample Datasets
# ==============================================
movies_data = {
    "movie_id": [1, 2, 3, 4, 5],
    "title": ["Avengers", "Inception", "Frozen", "Titanic", "Joker"],
    "runtime_min": [143, 148, 102, 195, 122],
    "budget_usd": [220_000_000, 160_000_000, 150_000_000, 200_000_000, 55_000_000],
    "box_office_usd": [1_500_000_000, 830_000_000, 1_280_000_000, 2_200_000_000, 1_070_000_000],
    "imdb_rating": [8.0, 8.8, 7.4, 7.8, 8.5],
    "genres": ["Action|Adventure|Sci-Fi", "Sci-Fi|Thriller", "Animation|Family|Musical",
               "Romance|Drama", "Crime|Drama|Thriller"]
}
movies = pd.DataFrame(movies_data)
movies.to_csv(os.path.join(base_dir, "data", "movies_sample.csv"), index=False)

reviews_data = {
    "review_id": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    "movie_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    "review_text": [
        "Amazing action and great VFX!", 
        "Too long, but still fun.",
        "Mind-blowing story and visuals!",
        "Confusing but brilliant.",
        "Kids loved it, very colorful!",
        "Songs were a bit annoying.",
        "Emotional masterpiece.",
        "A bit slow but unforgettable.",
        "Dark and intense performance.",
        "Masterpiece acting by Joaquin."
    ],
    "review_rating": [9, 7, 10, 8, 8, 6, 10, 9, 9, 10]
}
reviews = pd.DataFrame(reviews_data)
reviews.to_csv(os.path.join(base_dir, "data", "reviews_sample.csv"), index=False)

print("✅ Sample datasets created")

# ==============================================
# STEP 3: Load Data + Sentiment Analysis
# ==============================================
movies = pd.read_csv(os.path.join(base_dir, "data", "movies_sample.csv"))
reviews = pd.read_csv(os.path.join(base_dir, "data", "reviews_sample.csv"))

sia = SentimentIntensityAnalyzer()
reviews["sentiment"] = reviews["review_text"].apply(lambda x: sia.polarity_scores(x)["compound"])

sentiment_summary = reviews.groupby("movie_id")["sentiment"].mean().reset_index()
sentiment_summary.rename(columns={"sentiment": "avg_sentiment"}, inplace=True)

df = pd.merge(movies, sentiment_summary, on="movie_id")

# ==============================================
# STEP 4: Regression Model
# ==============================================
X = df[["budget_usd", "imdb_rating", "avg_sentiment"]]
y = df["box_office_usd"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train, y_train)

y_pred = reg_model.predict(X_test)

print("\n📊 Regression Model Performance:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

joblib.dump(reg_model, os.path.join(base_dir, "models", "rf_regression_model.joblib"))

# ==============================================
# STEP 5: Classification Model
# ==============================================
df["is_hit"] = (df["box_office_usd"] >= 2 * df["budget_usd"]).astype(int)

X_cls = df[["budget_usd", "imdb_rating", "avg_sentiment"]]
y_cls = df["is_hit"]

X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.3, random_state=42)

clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X_train, y_train)

y_pred_cls = clf_model.predict(X_test)

print("\n📊 Classification Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_cls))
print(classification_report(y_test, y_pred_cls))

joblib.dump(clf_model, os.path.join(base_dir, "models", "rf_classification_model.joblib"))

# ==============================================
# STEP 6: Visualization
# ==============================================
plt.figure(figsize=(8,6))
sns.scatterplot(x=df["imdb_rating"], y=df["avg_sentiment"],
                size=df["box_office_usd"], hue=df["is_hit"], sizes=(50, 500))
plt.title("🎬 Sentiment vs IMDB Rating (Bubble = Box Office, Color = Hit/Flop)")
plt.xlabel("IMDB Rating")
plt.ylabel("Avg Sentiment Score")
plt.legend()
plt.savefig(os.path.join(base_dir, "visuals", "sentiment_vs_rating.png"), dpi=150)
plt.close()

# ==============================================
# STEP 7: Package into ZIP
# ==============================================
shutil.make_archive(base_dir, "zip", base_dir)
print(f"📦 Project packaged as {base_dir}.zip")
