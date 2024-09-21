import streamlit as st
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
import pandas as pd

# Fonction pour charger les données
def load_data():
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    return data, digits.target, digits.images

# Fonction pour entraîner le modèle SVM
def train_svm(data, target, gamma=0.001):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.5, shuffle=False)
    clf = svm.SVC(gamma=gamma)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return clf, X_test, y_test, y_pred, accuracy

# Page de prédiction
def prediction_page(clf, X_test, y_test, y_pred):
    st.title("Predictions with SVM Classifier")
    st.write("""
**Support Vector Machine (SVM)** is a supervised learning algorithm used for classification and regression tasks. It works by finding the hyperplane that best separates different classes in the feature space, maximizing the margin between them.

SVM is particularly effective with the digits dataset because it handles high-dimensional data well and is robust against overfitting, especially when the number of features exceeds the number of samples. The digits dataset consists of 8x8 pixel images of handwritten digits, which translates to a high-dimensional feature space. SVM's ability to create non-linear decision boundaries through kernel functions allows it to capture the complexities of handwritten digit recognition, leading to high accuracy in classification tasks.
""")

    st.write(f"**Precision of the model**: {clf.score(X_test, y_test):.2%}")

    # Affichage des premières prédictions
    st.write("Here are the predictions for a few test images:")
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, y_pred):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")
    
    st.pyplot(fig)

    # Afficher le rapport de classification sous forme de DataFrame
    st.write("**Classification report:**")
    report = metrics.classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    # Matrice de confusion
    st.write("**Confusion matrix:**")
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    st.pyplot(fig)

    st.markdown("""
    We can see from this confusion matrix that the model performs very well for all digits (over 85 correct predictions out of a total of 90 images) except for the label 3, which is sometimes incorrectly predicted as 5, 7, or 8.
    """)

# Main app
def main():

    # Charger les données
    data, target, images = load_data()

    clf, X_test, y_test, y_pred, accuracy = train_svm(data, target)
    prediction_page(clf, X_test, y_test, y_pred)

if __name__ == "__main__":
    main()
