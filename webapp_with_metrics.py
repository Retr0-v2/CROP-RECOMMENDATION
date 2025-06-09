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

if __name__ == '__main__':
    main() 