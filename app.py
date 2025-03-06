import streamlit as st
import numpy as np
import pandas as pd
import librosa
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import os

# Constants
DATABASE = "mydb.sqlite3"
audio_dir = 'audio_files'
num_mfcc = 100
num_mels = 128
num_chroma = 50

# Load datasets and models
dataset = pd.read_csv('dataset.csv')  # For audio detection
face_model_path = 'F:\\Niranjan Projects\\Deepfake Face Project\\model.h5'
face_model = load_model(face_model_path)  # For face detection

# Function for voice detection
def detect_audio(file_path):
    X, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=num_mfcc).T, axis=0)
    mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=num_mels).T, axis=0)
    chroma_features = np.mean(librosa.feature.chroma_stft(y=X, sr=sample_rate, n_chroma=num_chroma).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=X).T, axis=0)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=X, sr=sample_rate).T, axis=0)
    flatness = np.mean(librosa.feature.spectral_flatness(y=X).T, axis=0)

    features = np.concatenate((mfccs, mel_spectrogram, chroma_features, zcr, spectral_centroid, flatness))
    distances = np.linalg.norm(dataset.iloc[:, :-1] - features, axis=1)
    closest_match_idx = np.argmin(distances)
    closest_match_label = dataset.iloc[closest_match_idx, -1]
    total_distance = np.sum(distances)
    closest_match_prob = 1 - (distances[closest_match_idx] / total_distance)
    closest_match_prob_percentage = "{:.3f}".format(closest_match_prob * 100)

    return closest_match_label, closest_match_prob_percentage

# Function to extract and preprocess video frames
def extract_frames(video_path, frame_rate=1):
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps / frame_rate)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % interval == 0:
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)

    cap.release()
    return np.array(frames)

def preprocess_frames(frames):
    return frames.astype('float32') / 255.0

def predict_deepfake(frames, model):
    predictions = model.predict(frames)
    return np.mean(predictions)

# Sidebar
icon_path = 'F:\\Niranjan Projects\\Deepfake Face Project\\image streamlit\\logo.png'
st.sidebar.image(icon_path, width=300)
st.sidebar.title('Dashboard')
options = st.sidebar.selectbox("Select Detection Type", ["Home", "Voice Detection", "Face Detection", "About"])

# Home Page
if options == 'Home':
    st.header("Deepfake Detection System")
    home_img = Image.open("F:\\Niranjan Projects\\Deepfake Face Project\\image streamlit\\deep1.jpg")
    st.image(home_img)
    st.markdown("""
    **Welcome to the Deepfake Detection System!**
    
    In a world where deepfakes blur the line between real and fake, our Deepfake Detection Platform provides a trusted solution for safeguarding authenticity. Utilizing cutting-edge AI, we specialize in detecting face swaps in videos and audio swaps in voice recordings, ensuring even the most sophisticated manipulations are uncovered. Our advanced algorithms analyze subtle inconsistencies in both visual and audio content, delivering reliable, real-time results. Designed for individuals and organizations alike, we empower you to combat misinformation, protect identities, and maintain trust in the digital era. Experience the power of AI in preserving truth—because authenticity matters.
    """)

# Voice Detection
elif options == 'Voice Detection':
    st.title("Audio Detection")
    st.image('static\\56.webp', width=800)

    uploaded_audio = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    if st.button("Detect"):
        if uploaded_audio is not None:
            file_path = os.path.join(audio_dir, uploaded_audio.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_audio.getbuffer())
            
            st.write("Analyzing audio...")
            closest_match_label, closest_match_prob = detect_audio(file_path)

            # Apply dynamic background color based on detection result
            if closest_match_label == 'deepfake':
                st.markdown(
                    """
                    <style>
                    .stApp {
                        background: linear-gradient(to bottom, #ff6f61, #de1a1a);
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                st.error(f"Result: Fake audio detected with {closest_match_prob}% confidence.")
            else:
                st.markdown(
                    """
                    <style>
                    .stApp {
                        background: linear-gradient(to right,  #00FA9A, #009900);
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                st.success(f"Result: Real audio detected with {closest_match_prob}% confidence.")
            
            os.remove(file_path)


# Face Detection
elif options == 'Face Detection':
    st.title("Deepfake Video Detection")

    st.image('F:\\Niranjan Projects\\Deepfake Face Project\\image streamlit\\deep2.jpg', width=800)
    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    
    if(st.button("Detect")):
        if uploaded_video is not None:
            video_path = 'temp_video.mp4'
            with open(video_path, 'wb') as f:
                f.write(uploaded_video.read())
            
            with st.spinner('Processing video...'):
                video_frames = extract_frames(video_path)
                video_frames = preprocess_frames(video_frames)
                prediction = predict_deepfake(video_frames, face_model)
            
            st.write(f"Prediction: {prediction:.2f}")
            threshold = 0.5
            if prediction < threshold:
                    st.success("The video is classified as REAL.")
                    st.markdown("""
                        <style>
                            .stApp {
                                background: linear-gradient(to right,  #00FA9A, #009900);
                                height: 100vh;
                            }
                        </style>
                    """, unsafe_allow_html=True)
                    st.write('<h1 style="color: black;">Real Video Detected</h1>', unsafe_allow_html=True)


            else:
                st.error("The video is classified as FAKE.")
                st.markdown("""
                     <style>
                        .stApp {
                            background: linear-gradient(to right, #ff6f61, #de1a1a);
                            height: 100vh;
                        }
                    </style>
                """, unsafe_allow_html=True)
                st.write('<h1 style="color: white;">Fake Video Detected</h1>', unsafe_allow_html=True)

# About Page
elif options == 'About':
    st.title("About the Deepfake Detection System")
    st.markdown("""
    This application leverages advanced AI techniques to detect deepfake content in both audio and video files.
    Built using Streamlit, it demonstrates the capabilities of modern deepfake detection technologies.
    """)
    video_path = 'F:\\Niranjan Projects\\Deepfake Face Project\\image streamlit\\Explanatory Video ERROR404.mp4'
    st.video(video_path)
