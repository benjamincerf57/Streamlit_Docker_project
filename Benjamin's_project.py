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

def eda_page(images, target):
    st.title("Images du dataset digits de sklearn")
    st.write("Utilisez le slider ci-dessous pour naviguer parmi les images du dataset.")

    # Créer un slider pour sélectionner l'indice de l'image (de 0 à 9)
    index = st.slider("Sélectionnez l'image à afficher", 0, len(images) - 1, 0)

    # Afficher l'image et le label correspondant
    fig, ax = plt.subplots(figsize=(2, 2))  
    ax.imshow(images[index], cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Label: {target[index]}")
    ax.set_axis_off() 

    st.pyplot(fig)

# Page de prédiction
def prediction_page(clf, X_test, y_test, y_pred):
    st.title("Prédictions avec le modèle SVM")
    st.write(f"Précision du modèle: {clf.score(X_test, y_test):.2%}")

    # Affichage des premières prédictions
    st.write("Voici les prédictions pour quelques images test:")
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, y_pred):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")
    
    st.pyplot(fig)

    # Afficher le rapport de classification sous forme de DataFrame
    st.write("Rapport de classification:")
    report = metrics.classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    # Matrice de confusion
    st.write("Matrice de confusion:")
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    st.pyplot(fig)

# Main app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choisissez une page", ["EDA", "Prédictions"])

    # Charger les données
    data, target, images = load_data()

    # Choisir la page
    if page == "EDA":
        eda_page(images[:10], target[:10])
    elif page == "Prédictions":
        clf, X_test, y_test, y_pred, accuracy = train_svm(data, target)
        prediction_page(clf, X_test, y_test, y_pred)

if __name__ == "__main__":
    main()
