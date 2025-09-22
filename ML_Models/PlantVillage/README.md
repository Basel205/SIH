# 🌿 PlantVillage Disease Detection - SIH Hackathon

## Overview
Plant disease detection system using EfficientNet-B0 trained on PlantVillage dataset for SIH hackathon.

## Features
- 🌱 39 plant disease classifications
- 📱 Web-based interface via Streamlit
- ☁️ Model hosted on Google Drive
- 🎯 Real-time disease prediction
- 📊 Confidence scoring

## Files Description
- `streamlit_app.py` - Main Streamlit application
- `plantVillage.py` - Training script for the model
- `classes.txt` - List of 39 disease classes
- `requirements.txt` - Python dependencies
- `.gitignore` - Git ignore file

## Setup Instructions

### 1. Google Drive Setup
1. Upload your `plant_disease_model.pth` to Google Drive
2. Set sharing to "Anyone with the link can view"
3. Copy the file ID from the shareable link
4. Update `MODEL_URL` in `streamlit_app.py`

### 2. Local Development
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### 3. Streamlit Cloud Deployment
1. Fork/upload this folder to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub account
4. Deploy from your repository

## Model Information
- **Architecture:** EfficientNet-B0
- **Classes:** 39 plant diseases
- **Dataset:** PlantVillage
- **Framework:** PyTorch

## Usage
1. Upload plant leaf image
2. Click "Analyze Disease"
3. View prediction results
4. Check confidence score

## Team
[Add your SIH team information here]
