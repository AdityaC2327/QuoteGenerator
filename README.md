# QuoteGenerator – AI-Powered Quote Generation Tool

---

## 📌 Description

QuoteGenerator is a Python-based machine learning application designed to generate meaningful, mood-based quotes using a fine-tuned GPT-2 model. It enables users to generate motivational, emotional, or reflective quotes tailored to specific moods. The system is modular, extensible, and built for experimentation and creative applications.

The project has two major components:

- **Model Training & Fine-Tuning** – Preprocessing, training logic, architecture configuration.
- **Quote Generation** – Use the trained model to generate new quotes conditioned on mood input.

---

## 🚀 Features

🎯 **Mood-Based Generation**  
Generates quotes based on emotional contexts like happiness, sadness, motivation, etc.

🧠 **Fine-Tuned GPT-2**  
Utilizes a GPT-2 model fine-tuned on a custom dataset of mood-labeled quotes.

🧹 **Dataset Cleaning & Preprocessing**  
Includes tools to clean raw quote datasets, tokenize text, one-hot encode moods, and prepare inputs for training.

📈 **Model Training**  
Train the model using PyTorch and Hugging Face Transformers, with support for saving and loading checkpoints.

💬 **Quote Output**  
After training, the model can generate new, creative quotes based on a specified mood or emotion.

---

## 🛠️ Tech Stack 

- **Language**: Python  
- **Libraries**: PyTorch, Hugging Face Transformers, pandas, numpy, scikit-learn
- **Backend**: Flask (for future deployment)  
- **Planned Frontend**: Streamlit or Flask UI

---

## 🏗️ Installation & Setup

### Prerequisites

- Python 3.8 or higher  
- pip  
- (Optional) Virtual environment tool like `venv`

   
### Setup  
1. **Clone the repository :**  
   ```sh
   git clone https://github.com/your-username/kaizen.git
   cd kaizen
   
2. **Create a virtual environment :**
   ```sh
   python -m venv venv

3. **Activate the virtual environment :**
   ```sh
      // For Windows:
      venv\Scripts\activate

      // For macOS/Linux:
      source venv/bin/activate
4. **Install Dependencies : **
   ```sh
   pip install -r requirements.txt

## 🧪 How to Use


1. **To Train the Model**

   ```sh
   python train.py

2. **To Generate Quotes :**  
   ```sh
   python generate.py
You’ll be prompted to enter a mood label, and the model will generate a quote for that mood.

## 📁 Project Structure

train.py – Handles model training

generate.py – Runs the generation pipeline

preprocessing.py – Encodes and tokenizes input

model_arch.py – Contains model architecture setup

test_model.py – Basic test runner for model performance

config/ – Stores model configuration files

logs/ – TensorBoard-compatible logs

trained_model/ – Saved model weights

fine_tuned_gpt2/ – Fine-tuned GPT-2 artifacts

train_dataset/, val_dataset/ – Cleaned and labeled training data

results/ – Final generated quote outputs

quotes_by_mood.csv – Master dataset for training



  

   
