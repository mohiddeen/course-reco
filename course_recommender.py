import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('course_dataset.csv')

y = df['recommended_course']
X = df.drop(columns=['recommended_course'])

categorical_columns = ['goal', 'hobby']

encoder_CT = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'),
         categorical_columns)
    ],
    remainder='passthrough'
)

X = encoder_CT.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"Training data size: {X_train.shape}")
print(f"Testing data size: {X_test.shape}")

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

print(f"Model accuracy: {accuracy:.2f}")

#test with real data
test_data = pd.DataFrame({
    'goal': ['job', 'business'],
    'hobby': ['Web Development', 'Design']
})
test_data_encoded = encoder_CT.transform(test_data)
predictions = model.predict(test_data_encoded)

print("Predictions for test data:", predictions)

#Save the model and encoder for future use
import joblib
joblib.dump(model, 'course_recommender_model1.pkl')
joblib.dump(encoder_CT, 'course_recommender_encoder1.pkl')




