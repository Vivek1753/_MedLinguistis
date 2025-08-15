import tempfile, os, json
from typing import Optional
from fastapi.responses import JSONResponse
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Body, Query

from core.models import ReportOptions, ReportResult, GenerateJsonPayload
from pipeline.report_generator import ReportGenerator

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok"}

@router.post("/generate_report", response_model=ReportResult)
async def generate_report(
    file: UploadFile = File(...),
    options: Optional[str] = Form(None)  # send JSON string in a form field named "options"
):
    # 1) Save uploaded PDF
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf_path = tmp.name
            tmp.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    # 2) Parse options JSON (if provided)
    opts: Optional[ReportOptions] = None
    if options:
        try:
            opts = ReportOptions(**json.loads(options))
        except Exception as e:
            # Clean up temp file before erroring
            try: os.remove(pdf_path)
            except: pass
            raise HTTPException(status_code=400, detail=f"Invalid 'options' JSON: {e}")

    # 3) Run pipeline
    try:
        generator = ReportGenerator(pdf_path=pdf_path, options=opts)
        result = generator.generate_report()
        return JSONResponse(content=json.loads(result.model_dump_json()))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try: os.remove(pdf_path)
        except: pass

# @router.post("/generate_report_from_json", response_model=ReportResult)
# async def generate_report_from_json(
#     file: UploadFile = File(...),            # the JSON file with the same schema as OCR output
#     options: Optional[str] = Form(None),     # JSON string for ReportOptions
# ):
#     if file.content_type not in {"application/json", "application/octet-stream", "text/plain"}:
#         raise HTTPException(status_code=400, detail="Please upload a JSON file.")

#     # Read and parse the JSON file
#     try:
#         raw = await file.read()
#         patient_data = json.loads(raw.decode("utf-8", errors="ignore"))
#         if not isinstance(patient_data, dict):
#             raise ValueError("Top-level JSON must be an object.")
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Invalid JSON file: {e}")

#     # Parse options (if any)
#     opts: Optional[ReportOptions] = None
#     if options:
#         try:
#             opts = ReportOptions(**json.loads(options))
#         except Exception as e:
#             raise HTTPException(status_code=400, detail=f"Invalid 'options' JSON: {e}")

#     # Run the pipeline using the provided JSON (no Gemini required)
#     try:
#         generator = ReportGenerator(pdf_path=None, patient_data=patient_data, options=opts)
#         result = generator.generate_report()
#         return JSONResponse(content=json.loads(result.model_dump_json()))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate_report_from_json", response_model=ReportResult)
async def generate_report_from_json(
    patient_data: dict = Body(...),              
    detection_threshold: float = Query(0.5),
    temperature: float = Query(0.1),
    top_k_retrieval: int = Query(10),
    top_k_final: int = Query(5),
):
    try:
        opts = ReportOptions(
            detection_threshold=detection_threshold,
            temperature=temperature,
            top_k_retrieval=top_k_retrieval,
            top_k_final=top_k_final,
        )
        generator = ReportGenerator(pdf_path=None, patient_data=patient_data, options=opts)
        result = generator.generate_report()
        return JSONResponse(content=json.loads(result.model_dump_json()))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
