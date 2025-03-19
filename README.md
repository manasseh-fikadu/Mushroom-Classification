# Mushroom Classification App

This Streamlit app uses a trained PyTorch EfficientNet-B4 model to classify mushroom images into 9 common genera found in Northern Europe.

## Features

- Upload and classify mushroom images
- View detailed information about each mushroom genus
- See prediction confidence scores
- Fast inference with pre-trained model

## Setup and Running Locally

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Deploying to Streamlit Hub

1. Create a GitHub repository and push your code:
   ```
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. Go to [Streamlit Hub](https://share.streamlit.io/) and sign in with your GitHub account

3. Click "New app" and select your repository, branch, and app file (app.py)

4. Click "Deploy"

## Model

The app uses a pre-trained model:

- **EfficientNet-B4**: An efficient model that balances network depth, width, and resolution for better accuracy and performance

## Mushroom Classes

The app can classify the following mushroom genera:

- Agaricus
- Amanita
- Boletus
- Cortinarius
- Entoloma
- Hygrocybe
- Lactarius
- Russula
- Suillus

## Important Note

This app is for educational purposes only. Never consume wild mushrooms based solely on an app's identification. Always consult with a professional mycologist.

## Implementation Details

The model was trained using PyTorch and PyTorch Lightning with the following techniques:

- Progressive layer unfreezing
- Layer-specific learning rates with AdamW optimizer
- Data augmentation to address class imbalance
- CosineAnnealingWarmRestarts scheduler
- Proper normalization with calculated mean and std values
