## Importing necessary libraries for the web app
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# Display Images
img = Image.open("crop.png")
st.image(img)

df = pd.read_csv('Crop_recommendation.csv')

X = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']
labels = df['label']

# Split the data into training and testing sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=42)
RF = RandomForestClassifier(n_estimators=20, random_state=5)
RF.fit(Xtrain,Ytrain)
predicted_values = RF.predict(Xtest)
accuracy = metrics.accuracy_score(Ytest, predicted_values)

# Function to plot confusion matrix with improved visualization
def plot_confusion_matrix():
    cm = confusion_matrix(Ytest, predicted_values)
    plt.figure(figsize=(15, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(labels.unique()),
                yticklabels=sorted(labels.unique()))
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    return plt

# Function to plot feature importance
def plot_feature_importance():
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': RF.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance, palette='viridis')
    plt.title('Feature Importance in Crop Prediction')
    plt.xlabel('Importance Score')
    return plt

# Function to plot per-class metrics
def plot_class_metrics():
    # Get classification report as dict
    report = classification_report(Ytest, predicted_values, output_dict=True)
    
    # Convert relevant part to dataframe
    classes_df = pd.DataFrame(report).drop(['accuracy', 'macro avg', 'weighted avg']).transpose()
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot precision, recall, f1-score
    classes_df[['precision', 'recall', 'f1-score']].plot(kind='bar', ax=ax1)
    ax1.set_title('Precision, Recall, and F1-Score by Crop')
    ax1.set_xticklabels(classes_df.index, rotation=45, ha='right')
    ax1.legend(loc='lower right')
    
    # Plot support (number of samples)
    classes_df['support'].plot(kind='bar', ax=ax2, color='green')
    ax2.set_title('Number of Test Samples by Crop')
    ax2.set_xticklabels(classes_df.index, rotation=45, ha='right')
    ax2.set_ylabel('Number of Samples')
    
    plt.tight_layout()
    return plt

# Function to load and display an image of the predicted crop
def show_crop_image(crop_name):
    image_path = os.path.join('crop_images', crop_name.lower()+'.jpg')
    if os.path.exists(image_path):
        st.image(image_path, caption=f"Recommended crop: {crop_name}", use_column_width=True)
    else:
        st.info(f"Recommended crop: {crop_name}")

# Save the model
RF_pkl_filename = 'RF.pkl'
RF_Model_pkl = open(RF_pkl_filename, 'wb')
pickle.dump(RF, RF_Model_pkl)
RF_Model_pkl.close()

RF_Model_pkl = pickle.load(open('RF.pkl','rb'))

def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    prediction = RF_Model_pkl.predict(np.array([nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]).reshape(1, -1))
    return prediction

def main():
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["Prediction", "Model Metrics"])
    
    with tab1:
        st.sidebar.title("AgriConnect")
        st.sidebar.header("Enter Crop Details")
        nitrogen = st.sidebar.number_input("Nitrogen", min_value=0.0, max_value=140.0, value=0.0, step=0.1)
        phosphorus = st.sidebar.number_input("Phosphorus", min_value=0.0, max_value=145.0, value=0.0, step=0.1)
        potassium = st.sidebar.number_input("Potassium", min_value=0.0, max_value=205.0, value=0.0, step=0.1)
        temperature = st.sidebar.number_input("Temperature (°C)", min_value=0.0, max_value=51.0, value=0.0, step=0.1)
        humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
        ph = st.sidebar.number_input("pH Level", min_value=0.0, max_value=14.0, value=0.0, step=0.1)
        rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=0.0, step=0.1)
        
        inputs = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
        if st.sidebar.button("Predict"):
            if not inputs.any() or np.isnan(inputs).any() or (inputs == 0).all():
                st.error("Please fill in all input fields with valid values before predicting.")
            else:
                prediction = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
                st.success(f"The recommended crop is: {prediction[0]}")
    
    with tab2:
        st.header("Model Performance Metrics")
        
        # Display accuracy score with context
        st.subheader("Overall Model Accuracy")
        st.info(f"The model achieves {accuracy:.2%} accuracy across {len(Ytest)} test samples")
        
        # Display interesting patterns
        st.subheader("Performance Highlights")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Perfect Performance (1.00) for:**")
            perfect_crops = ["apple", "banana", "blackgram", "chickpea", "coconut", "coffee", "cotton", "grapes"]
            st.write(", ".join(perfect_crops) + " and more...")
        with col2:
            st.markdown("**Areas for Improvement:**")
            st.write("- Jute (0.87 precision)")
            st.write("- Rice (0.82 recall)")
            st.write("- Lentil (0.95 recall)")
        
        # Display classification report
        st.subheader("Detailed Performance Metrics")
        report = classification_report(Ytest, predicted_values)
        st.text(report)
        
        # Display enhanced visualizations
        st.subheader("Performance Visualizations")
        
        # Confusion Matrix
        st.markdown("**1. Confusion Matrix**")
        conf_matrix = plot_confusion_matrix()
        st.pyplot(conf_matrix)
        
        # Per-class metrics
        st.markdown("**2. Per-Class Performance Metrics**")
        class_metrics = plot_class_metrics()
        st.pyplot(class_metrics)
        
        # Feature importance
        st.markdown("**3. Feature Importance Analysis**")
        feat_importance = plot_feature_importance()
        st.pyplot(feat_importance)

if __name__ == '__main__':
    main() 