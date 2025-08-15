# --------------------------------------------------------------------------------------
# Your original configuration (kept as-is, with small tweaks)
# --------------------------------------------------------------------------------------
import os

class PipelineConfig:
    """A single class to hold all configuration variables for the pipeline."""
    JSON_OUTPUT_PATH = "final_transcription.json"
    POPPLER_PATH = r"C:\Users\vivek\anaconda3\envs\rag\Library\bin"
    THEMATIC_MODEL_PATH = r"model\thematic_model"  # change if needed
    PLOT_SAVE_PATH = "qol_thematic_diagram.png"
    DETECTION_THRESHOLD = 0.5
    SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
    EMOTION_MODEL = "SamLowe/roberta-base-go_emotions"
    ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
    NLP_PLOT_OUTPUT_PATH = "nlp_analysis_diagram.png"
    JSONL_FULL_DATASET_PATH = "data/synthetic_york_hernia_patients_final.json"  # change if needed
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    OLLAMA_MODEL = "llama3:8b-instruct-q8_0"
    OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    CHROMA_DIR = "data/chroma"  # unused for in-memory, but kept
    CHUNK_SIZE = 400
    CHUNK_OVERLAP = 50
    TOP_K_RETRIEVAL = 10
    TOP_K_FINAL = 5
    # SUBTHEME_TO_THEME_MAP = {
    #     "symptoms_management_of_pain": "Symptoms & Function", "symptoms_freedom_of_movement": "Symptoms & Function", "symptoms_restriction_and_adaptation": "Symptoms & Function",
    #     "body_image_changes_to_perceptions_of_self": "Body Image", "body_image_fears_concerning_perceptions_of_others": "Body Image",
    #     "mental_health_emotional_responses": "Mental Health", "mental_health_disruptions_to_previous_identity": "Mental Health", "mental_health_coping_strategies": "Mental Health",
    #     "interpersonal_relationships_changes_in_sexual_relations": "Interpersonal Relationships", "interpersonal_relationships_difficulties_in_connecting_socially": "Interpersonal Relationships",
    #     "employment_financial_pressure": "Employment/Financial Concerns", "employment_return_to_work_issues": "Employment/Financial Concerns", "employment_costs_to_family": "Employment/Financial Concerns",
    # }
    SUBTHEME_TO_THEME_MAP = {
        "symptoms_management_of_pain": "Symptoms & Function",
        "symptoms_freedom_of_movement": "Symptoms & Function",
        "symptoms_restriction_and_adaptation": "Symptoms & Function",
        "body_image_changes_to_perceptions_of_self": "Body Image",
        "body_image_fears_concerning_perceptions_of_others": "Body Image",
        "mental_health_emotional_responses": "Mental Health",
        "mental_health_disruptions_to_previous_identity": "Mental Health",
        "mental_health_coping_strategies": "Mental Health",
        "interpersonal_relationships_changes_in_sexual_relations": "Interpersonal Relationships",
        "interpersonal_relationships_difficulties_in_connecting_socially": "Interpersonal Relationships",
        "employment_financial_pressure": "Employment/Financial Concerns",
        "employment_return_to_work_issues": "Employment/Financial Concerns",
        "employment_costs_to_family": "Employment/Financial Concerns",
    }
