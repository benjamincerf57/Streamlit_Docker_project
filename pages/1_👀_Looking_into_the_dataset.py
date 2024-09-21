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

def eda_page(images, target):
    st.title("Images of the dataset digits by sklearn")
    st.write("Use the slider below to navigate through the images of the dataset.")

    # Créer un slider pour sélectionner l'indice de l'image (de 0 à 9)
    index = st.slider("Select the image to display.", 0, len(images) - 1, 0)

    # Afficher l'image et le label correspondant
    fig, ax = plt.subplots(figsize=(2, 2))  
    ax.imshow(images[index], cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Label: {target[index]}")
    ax.set_axis_off() 

    st.pyplot(fig)

# Main app
def main():
    # Charger les données
    data, target, images = load_data()

    eda_page(images[:10], target[:10])

if __name__ == "__main__":
    main()
