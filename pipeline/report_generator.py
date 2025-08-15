import os
import io
import json
import uuid
import base64
import random
from typing import Any, Dict, List, Optional

# ---- Numeric & plotting ----
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

# ---- Torch / HF ----
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline as hf_pipeline,
)

# ---- Imaging / PDF ----
from pdf2image import convert_from_path
from PIL import Image

# ---- Google Gemini ----
import google.generativeai as genai

# ---- LangChain / RAG ----
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from typing import Optional
from core.models import ReportOptions, ReportResult, ReportArtifacts   
from core.config import PipelineConfig

from dotenv import load_dotenv
load_dotenv()

# --------------------------------------------------------------------------------------
# Helper utilities
# --------------------------------------------------------------------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

STATIC_DIR = os.path.abspath("files")
ensure_dir(STATIC_DIR)

# --------------------------------------------------------------------------------------
# Core pipeline wrapper converted for server use
# --------------------------------------------------------------------------------------
class ReportGenerator:
    """Orchestrates the entire report generation pipeline."""

    def __init__(
            self, 
            pdf_path: Optional[str] = None, 
            options: Optional[ReportOptions] = None, 
            output_dir: Optional[str] = None, 
            patient_data: Optional[Dict[str, Any]] = None
        ):
        self.pdf_path = pdf_path
        self.input_patient_data = patient_data
        self.config = PipelineConfig()
        if options:
            self.config.DETECTION_THRESHOLD = options.detection_threshold
            self.config.TOP_K_RETRIEVAL = options.top_k_retrieval
            self.config.TOP_K_FINAL = options.top_k_final
        self.temperature = options.temperature if options else 0.1

        # NEW: per-run directory under /files using UUID (unless caller provides one)
        # if output_dir:
        #     self.output_dir = output_dir
        # else:
        #     self.run_id = str(uuid.uuid4())
        #     self.output_dir = os.path.join(STATIC_DIR, self.run_id)
        # ensure_dir(self.output_dir)
        # # if caller passed output_dir, still set run_id for URLs
        # if not hasattr(self, "run_id"):
        #     self.run_id = os.path.basename(self.output_dir)

        self.run_id = str(uuid.uuid4()) if not output_dir else os.path.basename(output_dir.rstrip("/\\"))
        self.output_dir = output_dir or os.path.join(STATIC_DIR, self.run_id)
        ensure_dir(self.output_dir)

        self.patient_data: Optional[Dict[str, Any]] = None
        self.thematic_analyzer = None
        self.nlp_analyzer = None
        self.rag_system = None
        self._setup()

    # -------------------- Setup --------------------
    def _setup(self):
        """Initial setup and model loading."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY env var is not set.")
        genai.configure(api_key=api_key)

        self.thematic_analyzer = self.ThematicAnalysis(self.config)
        self.nlp_analyzer = self.AdvancedNLPAnalysis(self.config)
        self.rag_system = self.RAGSystem(self.config)

    # -------------------- Gemini helpers --------------------
    def run_llm_call(self, prompt: str, images: List[Image.Image], temperature: float = 0.1) -> str:
        model = genai.GenerativeModel('gemini-1.5-pro-latest', generation_config={"temperature": temperature})
        # Build a content list with images converted to bytes
        content = [prompt]
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            buf.seek(0)
            content.append({"mime_type": "image/png", "data": buf.read()})
        response = model.generate_content(content)
        return response.text.strip()

    # -------------------- Transcription --------------------
    def transcribe_patient_form(self, images: List[Image.Image]) -> dict:
        TARGET_JSON_SCHEMA = {
          "Patient_ID": "N/A", 
          "Metadata": {"Personal": "N/A", "Date_of_Birth": "N/A", "Age": "N/A", "Occupation_Category": "N/A", "BMI_Category": "N/A", "Gender": "N/A", "Marital_Status": "N/A", "Ethnicity": "N/A", "Job_Title": "N/A", "Previous_Hernia_Repairs": [], "Weight_Current_Kg": "N/A", "Medical_History": {"Prior_Major_Surgeries": [], "Diabetes": "N/A", "Smoking_Status": "N/A", "High_Blood_Pressure": "N/A", "Arthritis": "N/A", "Allergies": "N/A", "Family_History": "N/A", "Medication_Adherence": "N/A"}, "Medications": {}, "VHWG_Grade": "N/A", "Modified_VHWG_Breakdown": {}, "Modified_VHWG_Score": "N/A"},
          "QoL_Checklist": {
              "Symptoms": False, "Body_Image": False, "Mental_Health": False,
              "Relationships_(social_and_sexual)": False, "Employment": False,
              "able_to_lie_flat_comfortably": "N/A"
          },
          "Narratives": {"symptoms_management_of_pain": "N/A", "symptoms_freedom_of_movement": "N/A", "symptoms_restriction_and_adaptation": "N/A", "employment_financial_pressure": "N/A", "employment_return_to_work_issues": "N/A", "employment_costs_to_family": "N/A", "body_image_changes_to_perceptions_of_self": "N/A", "body_image_fears_concerning_perceptions_of_others": "N/A", "mental_health_emotional_responses": "N/A", "mental_health_disruptions_to_previous_identity": "N/A", "mental_health_coping_strategies": "N/A", "interpersonal_relationships_changes_in_sexual_relations": "N/A", "interpersonal_relationships_difficulties_in_connecting_socially": "N/A", "SharedDecisionMaking_Questions": "N/A", "SharedDecisionMaking_Hopes": "N/A", "SharedDecisionMaking_Matters_To_You": "N/A"}
        }
        prompt = f"""
        You are an expert medical data entry specialist. Your primary goal is accuracy.
        Analyze the provided patient form images and populate the given JSON schema.
        Pay special attention to dates and transcribe them exactly as they appear.
        For narratives, provide a verbatim transcription.
        Your output must be ONLY the JSON object.

        JSON schema to fill:
        {json.dumps(TARGET_JSON_SCHEMA, indent=2)}
        """
        response_text = self.run_llm_call(prompt, images, temperature=self.temperature)
        cleaned_response = response_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned_response)

        if not data.get("Patient_ID") or data.get("Patient_ID") in ["N/A", ""]:
             data["Patient_ID"] = f"Patient_{random.randint(10000, 99999)}"
        return data

    # -------------------- Main entry --------------------
    def generate_report(self) -> ReportResult:
        # 1) Convert PDF pages to images
        # pdf_images = convert_from_path(self.pdf_path)
        # pdf_images = convert_from_path(
        #     self.pdf_path,
        #     dpi=300,
        #     fmt="png",
        #     poppler_path=self.config.POPPLER_PATH,
        # )

        # # 2) Transcribe
        # self.patient_data = self.transcribe_patient_form(pdf_images)

        # json_path = os.path.join(self.output_dir, os.path.basename(self.config.JSON_OUTPUT_PATH))
        # with open(json_path, "w", encoding="utf-8") as f:
        #     json.dump(self.patient_data, f, indent=2, ensure_ascii=False)
        # self.config.JSON_OUTPUT_PATH = json_path

        # Source patient_data
        if self.input_patient_data:
            self.patient_data = dict(self.input_patient_data)  # copy
            if not self.patient_data.get("Patient_ID") or self.patient_data.get("Patient_ID") in ["N/A", ""]:
                self.patient_data["Patient_ID"] = f"Patient_{random.randint(10000, 99999)}"
        elif self.pdf_path:
            pdf_images = convert_from_path(
                self.pdf_path, dpi=300, fmt="png",
                poppler_path=self.config.POPPLER_PATH if self.config.POPPLER_PATH else None
            )
            self.patient_data = self.transcribe_patient_form(pdf_images)
        else:
            raise RuntimeError("No input provided. Pass either pdf_path or patient_data.")

        # Save JSON next to images and point config at it (for RAG)
        json_path = os.path.join(self.output_dir, os.path.basename(self.config.JSON_OUTPUT_PATH))
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.patient_data, f, indent=2, ensure_ascii=False)
        self.config.JSON_OUTPUT_PATH = json_path  # ensure RAG reads the per-run file

        # 3) Thematic Analysis
        theme_scores = self.thematic_analyzer.analyze_patient_themes(
            self.patient_data.get("Narratives", {})
        )
        thematic_plot_path = self.thematic_analyzer.create_circular_diagram(
            theme_scores, self.output_dir
        )

        # 4) Advanced NLP Analysis
        nlp_results = self.nlp_analyzer.analyze_narratives(self.patient_data)
        nlp_plot_path = self.nlp_analyzer.generate_nlp_diagram(
            nlp_results, self.output_dir
        )

        # 5) RAG Summary
        self.rag_system.populate_vectorstore_from_files()
        rag_summary = self.rag_system.get_rag_summary_for_patient(
            self.patient_data.get('Patient_ID', 'N/A'),
            "Summarize Quality of Life concerns."
        )

        def image_to_b64(path: str) -> str:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        thematic_b64 = image_to_b64(thematic_plot_path)
        nlp_b64 = image_to_b64(nlp_plot_path)

        artifacts = ReportArtifacts(
            thematic_plot_b64=thematic_b64,
            nlp_plot_b64=nlp_b64
        )

        result = ReportResult(
            patient_id=self.patient_data.get('Patient_ID', 'N/A'),
            metadata=self.patient_data.get('Metadata', {}),
            theme_scores=theme_scores,
            nlp_results=nlp_results,
            rag_summary=rag_summary,
            artifacts=artifacts,
        )

        response_json_path = os.path.join(self.output_dir, "report.json")
        with open(response_json_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(mode="json"), f, indent=2, ensure_ascii=False)

        return result

    # ----------------------------------------------------------------------------------
    # Nested: ThematicAnalysis
    # ----------------------------------------------------------------------------------
    # class ThematicAnalysis:
    #     def __init__(self, config: PipelineConfig):
    #         self.config = config
    #         self.device = "cuda" if torch.cuda.is_available() else "cpu"
    #         self.model = AutoModelForSequenceClassification.from_pretrained(self.config.THEMATIC_MODEL_PATH).to(self.device)
    #         self.tokenizer = AutoTokenizer.from_pretrained(self.config.THEMATIC_MODEL_PATH)

    #     def analyze_patient_themes(self, narratives: Dict[str, str]) -> Dict[str, float]:
    #         all_themes = sorted(list(set(self.config.SUBTHEME_TO_THEME_MAP.values())))
    #         all_subthemes = sorted(list(self.config.SUBTHEME_TO_THEME_MAP.keys()))
    #         all_labels = all_themes + all_subthemes
    #         aggregated_scores = {label: 0.0 for label in all_labels}

    #         def sigmoid(x):
    #             return 1 / (1 + np.exp(-x))

    #         for key, text in narratives.items():
    #             if key in all_labels and isinstance(text, str) and text.strip() and text.strip().upper() != 'N/A':
    #                 inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)
    #                 with torch.no_grad():
    #                     outputs = self.model(**inputs)
    #                 probs = sigmoid(outputs.logits.cpu().numpy()).squeeze()

    #                 if np.ndim(probs) == 0:
    #                     probs = np.array([float(probs)] * len(all_labels))
    #                 for i, label in enumerate(all_labels):
    #                     # score = float(probabilities[i]) if np.ndim(probabilities) else float(probabilities)
    #                     score = float(probs[i]) if i < len(np.atleast_1d(probs)) else 0.0
    #                     if score > aggregated_scores[label]:
    #                         aggregated_scores[label] = score

    #         # return only theme-level scores
    #         return {theme: aggregated_scores.get(theme, 0.0) for theme in set(self.config.SUBTHEME_TO_THEME_MAP.values())}
    
    class ThematicAnalysis:
        def __init__(self, config: PipelineConfig):
            self.config = config
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = AutoModelForSequenceClassification.from_pretrained(self.config.THEMATIC_MODEL_PATH).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.THEMATIC_MODEL_PATH)

        def _get_model_labels(self) -> List[str]:
            """Resolve the model's label list in the correct order using id2label when available."""
            id2label = getattr(self.model.config, "id2label", None)
            if isinstance(id2label, dict) and len(id2label) > 0:
                # keys may be str indices in some checkpoints
                try:
                    return [id2label[i] for i in range(len(id2label))]
                except Exception:
                    # sort by numeric key if they are strings
                    return [id2label[k] for k in sorted(id2label.keys(), key=lambda x: int(x))]
            # Fallback: assume subthemes order as trained
            return sorted(list(self.config.SUBTHEME_TO_THEME_MAP.keys()))

        def analyze_patient_themes(self, narratives: Dict[str, str]) -> Dict[str, float]:
            """
            Robust aggregation:
            1) Read model labels from id2label (or fallback).
            2) For each narrative text, predict probs and max-pool per label.
            3) Compute theme scores as the max of:
            - any DIRECT theme label prob (if your head includes theme labels), and
            - the max of its subtheme probs.
            """
            theme_set = set(self.config.SUBTHEME_TO_THEME_MAP.values())
            subtheme_to_theme = self.config.SUBTHEME_TO_THEME_MAP
            model_labels = self._get_model_labels()

            # storage for per-label max probs
            label_max: Dict[str, float] = {lbl: 0.0 for lbl in model_labels}

            # collect texts (skip empty / N/A)
            texts: List[str] = [
                v for v in narratives.values()
                if isinstance(v, str) and v.strip() and v.strip().upper() != "N/A"
            ]
            if not texts:
                # nothing to analyze
                return {theme: 0.0 for theme in theme_set}

            def _sigmoid(x):
                return 1.0 / (1.0 + np.exp(-x))

            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                logits = outputs.logits.squeeze(0).detach().cpu().numpy()
                # ensure 1D
                logits = np.atleast_1d(logits)
                probs = _sigmoid(logits)

                # align probs to model_labels length
                for i, lbl in enumerate(model_labels[:len(probs)]):
                    p = float(probs[i])
                    if p > label_max[lbl]:
                        label_max[lbl] = p

            # Aggregate to theme scores
            theme_scores: Dict[str, float] = {}
            for theme in theme_set:
                # max over subthemes belonging to this theme
                sub_p = max(
                    (label_max.get(st, 0.0) for st, th in subtheme_to_theme.items() if th == theme),
                    default=0.0
                )
                # if the model includes a DIRECT theme label, consider it too
                direct_theme_p = label_max.get(theme, 0.0)
                theme_scores[theme] = max(direct_theme_p, sub_p)

            return theme_scores

        def create_circular_diagram(self, theme_scores: Dict[str, float], output_dir: str) -> str:
            theme_order = [
                "Symptoms & Function",
                "Employment/Financial Concerns",
                "Interpersonal Relationships",
                "Mental Health",
                "Body Image",
            ]
            scores = [theme_scores.get(theme, 0.0) for theme in theme_order]
            wedge_colors = ["#92A9D1", "#D9C3E6", "#FADBD8", "#F2DFEA", "#D4ECE7"]
            inactive_color = "#EAEAEA"
            colors = [wedge_colors[i] if scores[i] > self.config.DETECTION_THRESHOLD else inactive_color for i in range(len(theme_order))]

            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(aspect="equal"))
            wedges, _ = ax.pie(
                [1]*len(theme_order),
                colors=colors,
                radius=1.2,
                startangle=90,
                wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2)
            )

            for i, p in enumerate(wedges):
                ang = (p.theta2 - p.theta1) / 2. + p.theta1
                y, x = np.sin(np.deg2rad(ang)), np.cos(np.deg2rad(ang))
                ha = { -1: "right", 1: "left" }[int(np.sign(x))]
                label_text = theme_order[i].replace('/', '/\n')
                ax.annotate(
                    f"{label_text}\n\nProb: {scores[i]:.4f}",
                    xy=(x*1.2, y*1.2),
                    xytext=(1.4*np.sign(x), 1.4*y),
                    horizontalalignment=ha,
                    verticalalignment='center',
                    fontsize=10,
                    fontweight='bold',
                    arrowprops=dict(arrowstyle="-", shrinkA=10, shrinkB=5)
                )

            ax.text(0, 0, "AWH", ha='center', va='center', fontsize=40, fontweight='bold')
            plt.title("Quality of Life Theme Analysis", fontsize=16, weight='bold', pad=20)

            out_path = os.path.join( output_dir, os.path.basename(self.config.PLOT_SAVE_PATH))
            plt.savefig(out_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            return out_path

    # ----------------------------------------------------------------------------------
    # Nested: AdvancedNLPAnalysis
    # ----------------------------------------------------------------------------------
    class AdvancedNLPAnalysis:
        def __init__(self, config: PipelineConfig):
            self.config = config
            self.device = 0 if torch.cuda.is_available() else -1
            self.sentiment_analyzer = hf_pipeline('sentiment-analysis', model=self.config.SENTIMENT_MODEL, device=self.device)
            self.emotion_analyzer = hf_pipeline('text-classification', model=self.config.EMOTION_MODEL, return_all_scores=True, device=self.device)
            self.mental_health_classifier = hf_pipeline("zero-shot-classification", model=self.config.ZERO_SHOT_MODEL, device=self.device)
            self.mh_labels = [
                "anxiety", "hopelessness", "social withdrawal", "frustration or irritability", "low mood or depression"
            ]

        def analyze_narratives(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
            full_narrative = ' '.join(
                str(v) for v in patient_data.get("Narratives", {}).values()
                if isinstance(v, str) and v and v.lower() != 'n/a'
            )
            if not full_narrative.strip():
                return {}

            sentiment_result = self.sentiment_analyzer(full_narrative, truncation=True)[0]
            emotions_raw = self.emotion_analyzer(full_narrative, truncation=True)[0]
            emotions_sorted = dict(sorted({e['label']: e['score'] for e in emotions_raw}.items(), key=lambda item: item[1], reverse=True))
            mh_result = self.mental_health_classifier(full_narrative, self.mh_labels, multi_label=True, truncation=True)
            mh_signals = {label: score for label, score in zip(mh_result['labels'], mh_result['scores'])}

            return {
                "sentiment": sentiment_result,
                "emotions": emotions_sorted,
                "mental_health_signals": mh_signals,
            }

        def generate_nlp_diagram(self, analysis: Dict[str, Any], output_dir: str) -> str:
            # Simple matplotlib-only visualization to avoid seaborn dependency
            fig = plt.figure(figsize=(12, 4))
            fig.suptitle('Advanced Linguistic Analysis', fontsize=14, weight='bold')

            # Sentiment - small donut
            ax1 = fig.add_axes([0.05, 0.15, 0.25, 0.7])
            sentiment = analysis.get('sentiment', {})
            sent_label = str(sentiment.get('label', 'N/A')).capitalize()
            sent_score = float(sentiment.get('score', 0.0))
            ax1.pie([sent_score, 1 - sent_score], labels=['', ''], startangle=90, wedgeprops={'width': 0.3, 'edgecolor': 'white'})
            ax1.set_title('Overall Sentiment', fontsize=10, weight='bold')
            ax1.text(0, 0, f"{sent_label}\n{sent_score:.2f}", ha='center', va='center', fontsize=10, weight='bold')

            # Emotions - barh (top 4)
            ax2 = fig.add_axes([0.37, 0.15, 0.25, 0.7])
            emotions = analysis.get('emotions', {})
            top_emotions = dict(list(sorted(emotions.items(), key=lambda x: x[1], reverse=True))[:4])
            ax2.barh(list(top_emotions.keys()), list(top_emotions.values()))
            ax2.set_title('Top Emotions Detected', fontsize=10, weight='bold')
            ax2.invert_yaxis()

            # Mental health - pie (top 3)
            ax3 = fig.add_axes([0.69, 0.15, 0.26, 0.7])
            mh_signals = analysis.get('mental_health_signals', {})
            top_mh = dict(list(sorted(mh_signals.items(), key=lambda x: x[1], reverse=True))[:3])
            if sum(top_mh.values()) == 0:
                # Avoid zero-sum pie
                vals = [1]
                labels = ["No signal"]
            else:
                vals = list(top_mh.values())
                labels = list(top_mh.keys())
            ax3.pie(vals, labels=labels, startangle=140, wedgeprops={'edgecolor': 'white'})
            ax3.set_title('Top Mental Health Patterns', fontsize=10, weight='bold')

            out_path = os.path.join(output_dir, os.path.basename(self.config.NLP_PLOT_OUTPUT_PATH))
            plt.savefig(out_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            return out_path

    # ----------------------------------------------------------------------------------
    # Nested: RAGSystem
    # ----------------------------------------------------------------------------------
    class RAGSystem:
        def __init__(self, config: PipelineConfig):
            self.config = config
            self.embedding_function = SentenceTransformerEmbeddings(model_name=self.config.EMBEDDING_MODEL)
            self.vectorstore = None
            try:
                self.ollama_llm = OllamaLLM(model=self.config.OLLAMA_MODEL, base_url=self.config.OLLAMA_URL)
            except Exception as e:
                self.ollama_llm = None
            self.documents_for_vectorstore: List[Document] = []
            self.reranker = CrossEncoder(self.config.RERANKER_MODEL)

        def _load_and_process_data(self, file_path: str, is_jsonl: bool = False) -> pd.DataFrame:
            try:
                if is_jsonl:
                    df = pd.read_json(file_path, lines=True)
                else:
                    df = pd.read_json(file_path, typ='series').to_frame().T
                df['full_narrative'] = df['Narratives'].apply(
                    lambda n: ' '.join(
                        str(v) for v in n.values() if isinstance(v, str) and v.lower() not in ['n/a', '']
                    )
                )
                return df
            except Exception as e:
                print(f"[RAG] ERROR loading data from {file_path}: {e}")
                return pd.DataFrame()

        def populate_vectorstore_from_files(self) -> None:
            """Load patient data from JSONL and JSON files, process narratives, and populate the vectorstore."""
            all_patients_df = self._load_and_process_data(self.config.JSONL_FULL_DATASET_PATH, is_jsonl=True)
            test_patient_df = self._load_and_process_data(self.config.JSON_OUTPUT_PATH)

            if test_patient_df.empty or all_patients_df.empty:
                print("[RAG] Setup failed: Could not load patient data.")
                return

            combined_df = pd.concat([all_patients_df, test_patient_df]).drop_duplicates(subset=['Patient_ID'], keep='last')
            splitter = RecursiveCharacterTextSplitter(chunk_size=self.config.CHUNK_SIZE, chunk_overlap=self.config.CHUNK_OVERLAP)

            docs: List[Document] = []
            for _, row in combined_df.iterrows():
                chunks = splitter.split_text(row['full_narrative'])
                docs.extend([Document(page_content=ch, metadata={"patient_id": row['Patient_ID']}) for ch in chunks])

            self.documents_for_vectorstore = docs
            # In-memory Chroma
            self.vectorstore = Chroma.from_documents(documents=docs, embedding=self.embedding_function)

        def get_rag_summary_for_patient(self, patient_id: str, query: str) -> str:
            if self.vectorstore is None:
                return "Error: RAG system not initialized. Vectorstore is missing."
            if self.ollama_llm is None:
                return "Error: LLM not available (Ollama unreachable)."

            patient_docs = [d for d in self.documents_for_vectorstore if d.metadata.get("patient_id") == patient_id]
            if not patient_docs:
                return f"No relevant narrative excerpts found for patient {patient_id} to generate a RAG summary."

            patient_vectorstore = Chroma.from_documents(documents=patient_docs, embedding=self.embedding_function)
            dense_retriever = patient_vectorstore.as_retriever(search_kwargs={'k': self.config.TOP_K_RETRIEVAL})
            bm25_retriever = BM25Retriever.from_documents(patient_docs)
            bm25_retriever.k = self.config.TOP_K_RETRIEVAL
            ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, dense_retriever], weights=[0.5, 0.5])

            def rerank_docs(result):
                query_text = result['question']
                docs = result['context']
                pairs = [(query_text, doc.page_content) for doc in docs]
                scores = self.reranker.predict(pairs)
                ranked_docs = [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
                return ranked_docs[:self.config.TOP_K_FINAL]

            prompt = ChatPromptTemplate.from_template(
                """You are an AI assistant for healthcare professionals, analyzing patient narratives.
                Your task is to synthesize information from the provided patient narrative excerpts to generate a concise, empathetic, and structured summary of the patient's Quality of Life (QoL) concerns.

                **CRITICAL INSTRUCTIONS - PLEASE READ CAREFULLY:**
                1.  **STRICTLY USE ONLY THE PROVIDED \"Retrieved Patient Narrative Excerpts\".**
                    * **DO NOT INFER, GUESS, OR ADD EXTERNAL INFORMATION.**
                    * **DO NOT FABRICATE OR HALLUCINATE TEXT.** Every piece of information must come directly from the excerpts.
                    * **NEVER INCLUDE \"Document ID\", \"page_content\", or any raw, truncated text from the excerpts.** Summarize the content, do not quote internal processing details.
                    * **ENSURE ALL INFORMATION DIRECTLY REFLECTS THE PATIENT'S NARRATIVE.** Do not skew or rephrase to introduce new meanings.
                2.  For each of the five QoL themes below, **EXTRACT AND SUMMARIZE SPECIFIC DETAILS, EXAMPLES, AND KEY PHRASES DIRECTLY FROM THE NARRATIVE.**
                    * When quoting directly, preserve original wording, spacing, and punctuation.
                    * Be concise and factual.
                3.  If a theme is genuinely NOT mentioned in the excerpts, state: "Not explicitly mentioned in narrative for this theme."
                4.  Maintain a professional and objective tone. Do not add recommendations.

                Themes:
                * Physical Symptoms and Functional Limitations
                * Body Image Concerns
                * Mental Health Challenges
                * Impact on Interpersonal Relationships
                * Employment and Financial Concerns

                ---
                Retrieved Patient Narrative Excerpts:
                {context}
                ---

                Patient's Overall Quality of Life Summary (structured by themes):
                """
            )

            rag_chain = (
                {"context": ensemble_retriever, "question": RunnablePassthrough()}
                | RunnablePassthrough.assign(context=RunnableLambda(rerank_docs))
                | RunnablePassthrough.assign(context=(lambda x: "\n\n".join(doc.page_content for doc in x["context"])))
                | prompt
                | self.ollama_llm
                | StrOutputParser()
            )
            return rag_chain.invoke(query)