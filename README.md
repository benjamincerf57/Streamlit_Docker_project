# Streamlit App for Digit Classification

This project was created as part of the **Tools for Data Scientist** course, which is part of the DSAIB master's program at **HEC Paris** and **École Polytechnique**. The objective of the project was to take an existing data science project, create a Streamlit app, and containerize it using Docker. 

For this project, I have chosen to work with one of the most classic datasets in data science: the **digits** dataset from **scikit-learn**. The handling of this dataset is based on the approach by **Gaël Varoquaux**, founder of scikit-learn.

## Project Overview

The Streamlit app presents a simple exploration and prediction model using the digits dataset, which consists of 8x8 pixel images of handwritten digits. The app allows users to explore the dataset and see predictions made using a Support Vector Machine (SVM) classifier.

- **Streamlit** is used to display the data and results interactively.
- **Docker** is used to containerize the application for easy distribution and deployment.

## Project Requirements

- **Docker**: Ensure you have Docker installed on your machine to run the containerized app.

## How to Run the App

### 1st method: pull this Github repo to have access to the entire project

1. **Clone this Github repositery**

2. **Build the docker image**:
   ```bash
   docker build -t streamlit-app 
   ```
3. **Run the Docker container**: 
   ```bash
   docker run -p 8501:8501 streamlit-app 
   ```
4. Open your browser and go to http://localhost:8501 to see the app.

### 2nd method: pull the image from my Docker repo to only execute the app

To run the Streamlit app using Docker, follow these steps:

1. **Pull the Docker image**   
   In your terminal, run the following command to pull the Docker image from DockerHub:
   
   ```bash
   docker pull benjamincerf/streamlit-app

2. **Run the Docker container**  
   After the image is pulled, use the following command to start the container and run the Streamlit app     on port **8501**:
   ```bash
   docker run -dp 0.0.0.0:8501:8501 benjamincerf/streamlit-app
3. **Access the Streamlit app**  
   Open your browser and go to http://localhost:8501. You should see the Streamlit interface of the app,     where you can explore the digits dataset and see predictions.

#### How to Remove the Docker Container (if wanted)

If you wish to remove the container and image from your machine after use, you can do the following:

1. **Stop the running container**:  
   Find the container's ID by running:
   ```bash
   docker ps
   ```
   Then stop the container by running:
   ```bash
   docker stop [CONTAINER_ID]
   ```
2. **Remove the Docker image**:  
   To remove the image from your machine, run:
   ```bash
   docker rmi benjamincerf/streamlit-app
   ```
   This will free up the space used by the image.
