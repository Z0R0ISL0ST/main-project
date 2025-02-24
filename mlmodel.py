import os
import numpy as np
import cv2
import pickle
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. TRAINING CODE (SVM, Logistic Regression, Random Forest)
def train_model(dataset_path, model_type="rf= "):
    print("loading dataset...")
    # Load dataset
    images = []
    labels = []

    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.resize(image, (128, 128))  # Resize to fixed size
                    images.append(image.flatten())
                    labels.append(label)

    # Convert to numpy arrays
    print("converting to numpy array")
    X = np.array(images)
    y = np.array(labels)

    # Encode labels
    print("label encoding")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Standardize features
    print("scalling")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split dataset
    print("splitting dataset")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Choose model
    if model_type == "svm":
        print("running svm model")
        model = SVC(probability=True, kernel='rbf', random_state=42)
    elif model_type == "logistic":
        print("running logestic regression model")
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == "rf":
        print("running random forest model")
        model = RandomForestClassifier(n_estimators=150, random_state=42)
        model.fit(X_train, y_train)
    else:
        raise ValueError("Unsupported model type. Choose 'svm' or 'logistic' or random forest.")

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)

    # 1. Plot classification report heatmap
    report_df = pd.DataFrame(report).transpose()

    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="Blues", fmt=".2f")
    plt.title(f"Classification Report - {model_type.upper()}")
    plt.ylabel("Metrics")
    plt.xlabel("Classes")
    plt.show()

    # 2. Plot accuracy as a bar chart
    plt.figure(figsize=(5, 5))
    plt.bar(["Accuracy"], [accuracy], color="skyblue")
    plt.title(f"Model Accuracy - {model_type.upper()}")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.show()

    # Save model and encoders
    model_filename = f"maize_model_{model_type}.pkl"
    with open(model_filename, "wb") as model_file:
        pickle.dump(model, model_file)
    with open("label_encoder.pkl", "wb") as encoder_file:
        pickle.dump(label_encoder, encoder_file)
    with open("scaler.pkl", "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    print(f"Model training completed and saved as {model_filename}.")

# 2. PREDICTION CODE
def predict_image(image_path, model_type="svm"):
    # Load model and encoders
    model_filename = f"maize_model_{model_type}.pkl"
    with open(model_filename, "rb") as model_file:
        model = pickle.load(model_file)
    with open("label_encoder.pkl", "rb") as encoder_file:
        label_encoder = pickle.load(encoder_file)
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return

    image = cv2.resize(image, (128, 128))
    image = image.flatten().reshape(1, -1)

    # Standardize image
    image = scaler.transform(image)

    # Predict class and probability
    probabilities = model.predict_proba(image)[0]
    predicted_class = model.predict(image)[0]
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    print("Predicted Class:", predicted_label)
    print("Probabilities:")
    for label, prob in zip(label_encoder.classes_, probabilities):
        print(f"{label}: {prob * 100:.2f}%")

# Example usage
#train_model("dataset", model_type="svm")
#train_model("dataset", model_type="logistic")
train_model("dataset", model_type="rf")
# predict_image("path_to_image.jpg", model_type="svm")
# predict_image("path_to_image.jpg", model_type="logistic")
