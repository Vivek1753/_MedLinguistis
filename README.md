# 🧠 Quality of Life (QoL) Patient Report Generation Pipeline

This project is a complete, offline-capable NLP system that processes patient narratives and medical metadata to produce structured Quality of Life (QoL) reports. It combines transformer-based thematic classification, sentiment and emotion analysis, and vector search using LangChain for context-aware insights. A FastAPI backend provides easy API access, enabling seamless integration with healthcare applications and local deployments without reliance on external APIs.

---

## 🔧 Tech Stack

- **Python Version:** 3.13.5
- **Frameworks & Libraries:**
  - Transformers (Hugging Face)
  - LangChain
  - ChromaDB
  - FastAPI
  - scikit-learn
  - PyTorch
  - Uvicorn
  - TextBlob
  - Matplotlib, Wordcloud

---

## 🚀 Getting Started

### 1. 📥 Clone the Repository
```bash
cd your-repo-name
git clone https://github.com/Vivek1753/MedLinguistis.git
```

### 2. Create and Activate a Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download and Setup Models
This project uses a fine-tuned model named **`thematic_model`**.  
The model is stored on Google Drive and must be downloaded before running the pipeline.

**Steps:**

1. Download the model from the provided Google Drive link:  
   [📥 Download thematic_model](https://drive.google.com/drive/folders/1ijZBi3NnoHUKqkqBZcTa0stEtPaQZN33?usp=sharing)

2. Create a `model` folder in the project root (if it does not already exist):
  ```bash
   mkdir model
   ```

3. Extract the downloaded model into the model/ folder:
   ```bash
   model/
    └── thematic_model/
        ├── config.json
        ├── pytorch_model.bin
        ├── tokenizer.json
        ├── ...
   ```
   
5. Verify the model path in your code points to:
   ```bash
   MODEL_PATH = "model/thematic_model"
   ```
   






