from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# 1. Sample labeled resume data (simulated for training)
# Label: 1 = Good match, 0 = Not a match
resume_texts = [
    "Experienced in python, sql, machine learning, and NLP",       # good
    "Worked on AI projects, data analysis, bachelor degree",       # good
    "Excel and admin experience, communication, filing",           # bad
    "Strong in C++, no experience in data or ML",                  # bad
    "Python developer with NLP and AI background",                 # good
    "Graphic designer, adobe tools, creative design",              # bad
]
labels = [1, 1, 0, 0, 1, 0]

# 2. Create ML pipeline (TF-IDF + Logistic Regression)
model = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression()
)

# 3. Train the model
model.fit(resume_texts, labels)

# 4. Predict new resume
new_resume = """
John Doe has 3 years of experience in machine learning and NLP.
He holds a bachelor's degree in computer science and has worked with Python, SQL, and data analysis.
"""

# 5. Predict and output result
prediction = model.predict([new_resume])[0]
prob = model.predict_proba([new_resume])[0][1]  # confidence score

print(" Resume Prediction Result")
print("--------------------------")
print("Prediction:", "Good Match" if prediction == 1 else "Not a Match")
print("Confidence Score:", round(prob * 100, 2), "%")