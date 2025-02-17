from fastapi import FastAPI, File, UploadFile
import pdfplumber
from docx import Document
import os
from typing import Dict

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file"""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a DOCX file"""
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text


@app.post("/upload-resume/")
async def upload_resume(file: UploadFile = File(...)) -> Dict[str, str]:
    """API endpoint to upload a resume and extract text"""
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Extract text based on file type
    if file.filename.endswith(".pdf"):
        extracted_text = extract_text_from_pdf(file_path)
    elif file.filename.endswith(".docx"):
        extracted_text = extract_text_from_docx(file_path)
    else:
        return {"error": "Unsupported file type. Please upload a PDF or DOCX file."}

    return {"filename": file.filename, "extracted_text": extracted_text}

