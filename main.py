# main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Union
from datetime import datetime
import shutil
import os
import pandas as pd
import fitz  # PyMuPDF
import uuid
import requests
import json
import re

# === Setup ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# === Models ===
class Section(BaseModel):
    id: int
    section_title: Union[str, None] = None
    text: Union[str, None] = None
    page: Union[int, None] = None
    sheet_name: Union[str, None] = None
    row_number: Union[int, None] = None
    columns: Union[Dict[str, str], None] = None

class FileData(BaseModel):
    filename: str
    filetype: str
    uploaded_at: datetime
    extracted_content: List[Section]
    source_path: str

class SaveResponse(BaseModel):
    status: str
    message: str

# === Routes ===
@app.post("/upload", response_model=FileData)
async def upload_file(file: UploadFile = File(...)):
    file_ext = file.filename.split(".")[-1].lower()
    file_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if file_ext == "pdf":
        extracted = extract_pdf(save_path)
    elif file_ext in ["xls", "xlsx"]:
        extracted = extract_excel(save_path)
    else:
        return {"error": "Unsupported file type"}

    return FileData(
        filename=file.filename,
        filetype=file_ext,
        uploaded_at=datetime.utcnow(),
        extracted_content=extracted,
        source_path=save_path
    )

@app.post("/save", response_model=SaveResponse)
async def save_to_db(file_data: FileData):
    print("Saving the following data to DB:")
    print(file_data.json(indent=2))
    return SaveResponse(status="success", message="Data saved.")

# === Helpers ===

def query_ollama(messages, model="deepseek-r1:8b"):
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False
            }
        )

        if response.status_code == 200:
            response_json = response.json()
            content = response_json.get("message", {}).get("content", "")

            # Debugging: Print raw AI response content
            print("Raw AI response content:")
            print(content)

            match = re.search(r'\{[\s\S]*?\}', content)
            if not match:
                raise ValueError("No valid JSON object found in response.")

            return json.loads(match.group(0))

        else:
            raise RuntimeError(f"Ollama returned status {response.status_code}: {response.text}")

    except Exception as e:
        print(f"[ERROR] Ollama call failed: {e}")
        return None

def extract_pdf(path):
    try:
        doc = fitz.open(path)
        all_data = ""

        for i, page in enumerate(doc):
            text = page.get_text()
            all_data += f"Page {i+1}:\n{text}\n"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI that extracts structured student transcript information from a PDF document. "
                    "Extract only these fields if available:\n\n"
                    "- ID Number\n- Name (split into Lastname, Firstname, Middlename)\n- Date of Birth\n- Sex\n- Status\n"
                    "- Place of Birth\n- Address\n- Parents/Guardian\n- Entrance Data\n- Course\n\n"
                    "Return a single JSON object with exactly these keys. Enclose all keys and string values in double quotes. Do not include markdown formatting or extra text:\n"
                    "id_number, lastname, firstname, middlename, date_of_birth, sex, status, "
                    "place_of_birth, address, parents_guardian, entrance_data, course."
                )
            },
            {
                "role": "user",
                "content": f"The PDF content is:\n\n{all_data}\n\nExtract and return the fields as a single JSON object."
            }
        ]

        student_data = query_ollama(messages)
        if student_data:
            return [Section(
                id=1,
                sheet_name="Student Transcript",
                row_number=1,
                columns={key: student_data.get(key, "") for key in [
                    "id_number", "lastname", "firstname", "middlename", "date_of_birth", "sex",
                    "status", "place_of_birth", "address", "parents_guardian", "entrance_data", "course"
                ]}
            )]
        else:
            return [Section(id=1, section_title="AI Error", text="Failed to extract data from local model.")]

    except Exception as e:
        print(f"[ERROR] Exception during PDF extraction: {str(e)}")
        return [Section(id=1, section_title="Exception", text=f"{str(e)}")]

def extract_excel(path):
    try:
        file_ext = path.split(".")[-1].lower()
        engine = "xlrd" if file_ext == "xls" else "openpyxl"

        xls = pd.ExcelFile(path, engine=engine)
        all_data = ""

        for sheet in xls.sheet_names:
            df = xls.parse(sheet).fillna("").astype(str)
            for i, row in df.iterrows():
                row_dict = row.to_dict()
                row_str = "; ".join([f"{k}: {v}" for k, v in row_dict.items()])
                all_data += f"[{sheet}, Row {i+1}] {row_str}\n"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI that extracts structured student transcript information from Excel. "
                    "Extract only these fields if available:\n\n"
                    "- ID Number\n- Name (split into Lastname, Firstname, Middlename)\n- Date of Birth\n- Sex\n- Status\n"
                    "- Place of Birth\n- Address\n- Parents/Guardian\n"
                    "- Entrance Data\n- Course\n\n"
                    "Return a single JSON object with exactly these keys. Enclose all keys and string values in double quotes. Do not include markdown formatting or extra text:\n"
                    "id_number, lastname, firstname, middlename, date_of_birth, sex, status, "
                    "place_of_birth, address, parents_guardian, entrance_data, course.\n\n"
                    "If the name is in the format 'LASTNAME, FIRSTNAME MIDDLENAME', split it accordingly. "
                    "If there is no middle name, set `middlename` to null or an empty string."
                )
            },
            {
                "role": "user",
                "content": f"The spreadsheet content is:\n\n{all_data}\n\nExtract and return the fields as a single JSON object."
            }
        ]

        student_data = query_ollama(messages)
        if student_data:
            return [Section(
                id=1,
                sheet_name="Student Transcript",
                row_number=1,
                columns={key: student_data.get(key, "") for key in [
                    "id_number", "lastname", "firstname", "middlename", "date_of_birth", "sex",
                    "status", "place_of_birth", "address", "parents_guardian", "entrance_data", "course"
                ]}
            )]
        else:
            return [Section(id=1, section_title="AI Error", text="Failed to extract data from local model.")]

    except Exception as e:
        print(f"[ERROR] Exception during Excel extraction: {str(e)}")
        return [Section(id=1, section_title="Exception", text=f"{str(e)}")]

