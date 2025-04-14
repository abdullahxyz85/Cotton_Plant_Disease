import streamlit as st
from PIL import Image
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from skimage import exposure
import os
import joblib
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Cotton Plant Disease Classifier",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .title {
        color: #2e8b57;
        text-align: center;
        font-size: 3em;
    }
    .sidebar .sidebar-content {
        background-color: #e8f5e9;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        border-radius: 10px;
        padding: 20px;
        background-color: #e8f5e9;
        margin-top: 20px;
    }
    .class-progress {
        margin-bottom: 10px;
    }
    .uploaded-image {
        max-width: 100%;
        border-radius: 10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# Class names and descriptions
class_names = {
    0: "Diseased Cotton Leaf",
    1: "Diseased Cotton Plant",
    2: "Fresh Cotton Leaf",
    3: "Fresh Cotton Plant"
}

class_descriptions = {
    0: "The leaf shows signs of disease. Common cotton leaf diseases include bacterial blight, alternaria leaf spot, or cercospora leaf spot.",
    1: "The entire plant appears diseased. This could indicate serious issues like fusarium wilt or verticillium wilt.",
    2: "The cotton leaf appears healthy with no visible signs of disease.",
    3: "The cotton plant appears healthy and thriving with no visible signs of disease."
}

# Function to extract features using HOG (Histogram of Oriented Gradients)
def extract_features(img):
    # Resize image to 64x64
    img = img.resize((64, 64))
    
    # Convert to grayscale
    img_gray = img.convert('L')
    img_array = np.array(img_gray)
    
    # Extract HOG features
    fd, hog_image = hog(img_array, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True)
    
    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    
    return fd, hog_image_rescaled

# Load a pre-trained model (in a real app, you would load your actual trained model)
@st.cache_resource
def load_model():
    # This is a dummy model - in a real app, you would load your trained model
    # Here we'll create a simple SVM classifier with dummy data
    # In practice, you would train this on your actual dataset and save/load it
    
    # Create a simple SVM classifier
    model = SVC(probability=True)
    
    # Create some dummy data for the example
    # In reality, you would train this on your actual dataset
    X_dummy = np.random.rand(100, 144)  # 144 is the HOG feature vector size for 64x64 image
    y_dummy = np.random.randint(0, 4, 100)
    
    # Scale features
    scaler = StandardScaler()
    X_dummy = scaler.fit_transform(X_dummy)
    
    # Train the model
    model.fit(X_dummy, y_dummy)
    
    return model, scaler

model, scaler = load_model()

# Function to make prediction
def predict_image(img):
    try:
        # Extract features
        features, hog_image = extract_features(img)
        
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Make prediction
        prediction = model.predict_proba(features_scaled)
        return prediction[0], hog_image
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

# Main app
def main():
    st.markdown('<h1 class="title">🌱 Cotton Plant Disease Classifier</h1>', unsafe_allow_html=True)
    st.markdown("""
    This application helps identify diseases in cotton plants using machine learning. 
    Upload an image of a cotton leaf or plant, and the system will analyze it for signs of disease.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload an Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image_pil = Image.open(uploaded_file)
            st.image(image_pil, caption="Uploaded Image", use_column_width=True, width=300, output_format="JPEG")
            
            if st.button("Analyze Image"):
                with st.spinner("Analyzing the image..."):
                    # Make prediction
                    prediction, hog_image = predict_image(image_pil)
                    
                    if prediction is not None:
                        predicted_class = np.argmax(prediction)
                        confidence = np.max(prediction)
                        
                        # Display results
                        with col2:
                            st.subheader("Analysis Results")
                            
                            # Prediction box
                            with st.container():
                                st.markdown(f"**Prediction:** {class_names[predicted_class]}")
                                st.markdown(f"**Confidence:** {confidence*100:.2f}%")
                                
                                # Progress bar for confidence
                                st.progress(float(confidence))
                                
                                # Class description
                                st.markdown("**Description:**")
                                st.info(class_descriptions[predicted_class])
                            
                            # Show HOG features visualization
                            st.subheader("Feature Visualization")
                            st.image(hog_image, caption="HOG Features", use_column_width=True)
                            
                            # Confidence for all classes
                            st.markdown("**Confidence Breakdown:**")
                            for i, (name, desc) in enumerate(class_names.items()):
                                conf = prediction[i]
                                st.markdown(f"**{name}:** {conf*100:.2f}%")
                                st.progress(float(conf))
    
    # Add some sample images in the sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This app uses machine learning with HOG features to classify images of cotton plants into:
        - Diseased Cotton Leaf
        - Diseased Cotton Plant
        - Fresh Cotton Leaf
        - Fresh Cotton Plant
        """)
        
        st.header("Sample Images")
        st.markdown("""
        For best results, use images with:
        - Clear view of the leaf or plant
        - Good lighting conditions
        - Minimal background clutter
        """)
        
        st.header("Instructions")
        st.markdown("""
        1. Upload an image of a cotton leaf or plant
        2. Click 'Analyze Image'
        3. View the results and confidence levels
        """)
        
        st.markdown("---")
        st.markdown("Built with ❤️ using scikit-learn and Streamlit")

if __name__ == "__main__":
    main()