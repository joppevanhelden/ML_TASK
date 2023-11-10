import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

st.set_option('deprecation.showPyplotGlobalUse', False)

# load the dataset
glass_identification = fetch_ucirepo(id=42)
X = glass_identification.data.features
y = glass_identification.data.targets

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train the baseline model w random forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# train SVM and KNN models
svm_model = SVC()
svm_model.fit(X_train, y_train)

knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)

st.title("Task Machine Learning: Benchmarking")

# sidebar
selected_models = st.sidebar.multiselect("Select Models", ["Random Forest", "SVM", "KNN"])
compare_confusion_matrices = st.sidebar.checkbox("Compare Confusion Matrices")

for selected_model in selected_models:
    st.subheader(f"{selected_model} Performance:")

    # the different ML techniques
    if selected_model == "Random Forest":
        model = rf_model
    elif selected_model == "SVM":
        model = svm_model
    else:
        model = knn_model

    # Display model performance
    accuracy = model.score(X_test, y_test)
    st.write(f"Accuracy: {accuracy}")

    # Confusion Matrix
    if compare_confusion_matrices:
        y_pred = model.predict(X_test)
        conf_matrix = confusion_matrix(y_test, y_pred)
        st.subheader(f"{selected_model} Confusion Matrix:")

        # Plot confusion matrix
        plt.figure(figsize=(9, 7))
        cmap = "Blues" if selected_model == "Random Forest" else "Greens" if selected_model == "SVM" else "Reds"
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap=cmap, cbar=False)
        plt.title(f"{selected_model} Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        st.pyplot()

# EDA visualization: bar chart 'Type_of_glass'
st.subheader("Exploratory Data Analysis (EDA) Visualization:")
plt.figure(figsize=(8, 6))
sns.countplot(x='Type_of_glass', data=glass_identification.data.targets)
plt.title('Distribution of Glass Types')
plt.xlabel('Type of Glass')
plt.ylabel('Count')
st.pyplot()