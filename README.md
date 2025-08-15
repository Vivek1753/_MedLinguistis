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

### 1. Clone the Repository
```bash
git clone https://github.com/Vivek1753/_MedLinguistis.git
cd _MedLinguistis
```

### 2. Create and Activate Virtual Environment
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
   
4. Make sure your code points to:
 ```bash
 MODEL_PATH = "model/thematic_model"
 ```

⚠ Note: The model is not included in the repository due to file size limits. It must be downloaded before running the project.

### 5. Run the FastAPI Application
```bash
uvicorn app:app --reload
```
Default: http://127.0.0.1:8000

### 6. API Endpoints
1️⃣ JSON Input Endpoint
Generates a patient QoL report from a JSON file.
- Method: `POST`
- Endpoint:
  ```arduino
  http://127.0.0.1:8000/generate_report_from_json
  ```
- Headers:
  ```pgsql
  Content-Type: application/json
  ```
- Body Type:  `raw` → `JSON`

2️⃣ PDF Input Endpoint
Generates a patient QoL report from a PDF file.
- Method: `POST`
- Endpoint:
  ```arduino
  http://127.0.0.1:8000/generate_report
  ```
- Headers:
  ```bash
  Content-Type: multipart/form-data
  ```
- Body:
  - Key: `file`
  - Value: Select your `.pdf` file to upload.

### 7. Example Postman Usage
For JSON Input
1. Set method to POST.
2. URL: `http://127.0.0.1:8000/generate_report_from_json`
3. In Headers, set:
   ```pgsql
   Content-Type: application/json
   ```
4. In Body → Raw → JSON → Paste example JSON from testing_data/.

For PDF Input
1. Set method to POST.
2. URL: `http://127.0.0.1:8000/generate_report`
3. In Body → Form-data:
   - Key: `file`
   - Type: File
   - Value: Choose your PDF file.
  



   






