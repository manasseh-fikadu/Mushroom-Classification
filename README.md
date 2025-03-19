# Mushroom Classification App

This Streamlit app uses trained PyTorch models (ResNet152 and EfficientNet-B4) to classify mushroom images into 9 common genera found in Northern Europe.

## Features

- Upload and classify mushroom images
- Choose between ResNet152 and EfficientNet-B4 models
- View detailed information about each mushroom genus
- See prediction confidence scores
- Fast inference with pre-trained models
- Automatic model downloading from cloud storage

## Setup and Running Locally

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Host your model files:
   - Upload your `resnet_model.pth` and `efficientnet_model.pth` files to Google Drive or another cloud storage service
   - For Google Drive:
     - Upload each file and get the sharing link
     - Extract the file ID from the link (the part after `/d/` and before `/view`)
     - Replace `YOUR_GOOGLE_DRIVE_FILE_ID` in `app.py` with your actual file IDs for each model
   - For other hosting services:
     - Uncomment the direct URL download section in the `download_model_if_needed` function
     - Replace `YOUR_DIRECT_DOWNLOAD_URL` with your actual download URLs

3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Deploying to Streamlit Hub

1. Create a GitHub repository and push your code (without the large model files):
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

5. When the app runs for the first time, it will automatically download the selected model from your cloud storage

## Models

The app includes two pre-trained models:

- **ResNet152**: A deep residual network with 152 layers, known for its ability to train very deep networks effectively
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

The models were trained using PyTorch and PyTorch Lightning with the following techniques:

- Progressive layer unfreezing
- Layer-specific learning rates with AdamW optimizer
- Data augmentation to address class imbalance
- CosineAnnealingWarmRestarts scheduler
- Proper normalization with calculated mean and std values
