
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv('Breast_cancer_data.csv')
df = df[['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness', 'diagnosis']]


df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

df.dropna(inplace=True)

print("Number of samples after dropping missing values:", len(df))

if len(df) < 2:
    print("Insufficient samples for train-test split.")
else:
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    if len(X_train) < 1 or len(X_test) < 1:
        print("Insufficient samples for train-test split after the split.")
    else:

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)


        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)


        y_pred = model.predict(X_test_scaled)


        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        print(f"Accuracy: {accuracy}")
        print(f"Confusion Matrix:\n{conf_matrix}")
        print(f"Classification Report:\n{class_report}")
