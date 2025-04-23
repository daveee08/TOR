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
import zipfile
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

OPENROUTER_API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "deepseek/deepseek-r1:free"

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
                    "Return a single JSON object with exactly these keys:\n"
                    "id_number, lastname, firstname, middlename, date_of_birth, sex, status, "
                    "place_of_birth, address, parents_guardian, entrance_data, course."
                )
            },
            {
                "role": "user",
                "content": f"The PDF content is:\n\n{all_data}\n\nExtract and return the fields as a single JSON object."
            }
        ]

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": 0.3
        }

        response = requests.post(OPENROUTER_URL, headers=headers, json=data)
        
        if response.status_code == 200:
            try:
                response_json = response.json()
                
                # Debugging: Print the entire response to understand its structure
                print(f"[DEBUG] Response JSON:\n{json.dumps(response_json, indent=2)}")
                
                # Check if 'choices' key exists and is non-empty
                choices = response_json.get("choices", [])
                if not choices:
                    print("[ERROR] No choices found in the AI response.")
                    return [Section(id=1, section_title="AI Error", text="No choices found in the response.")]
                
                # Extract the content from the first choice
                content = choices[0].get("message", {}).get("content", "")
                if not content:
                    print("[ERROR] No content in the AI response.")
                    return [Section(id=1, section_title="AI Error", text="No content found in the response.")]
                
                # Extract the JSON string from the content
                def extract_json_string(text):
                    match = re.search(r'\{[\s\S]*?\}', text)
                    return match.group(0) if match else None

                json_str = extract_json_string(content)
                if not json_str:
                    raise ValueError("No valid JSON found in AI response.")

                student_data = json.loads(json_str)

                # Clean up student_data to remove None values
                student_data = {key: (value if value is not None else "") for key, value in student_data.items()}

                return [Section(
                    id=1,
                    sheet_name="Student Transcript",
                    row_number=1,
                    columns=student_data
                )]

            except Exception as parse_err:
                print(f"[ERROR] Failed to parse AI response: {parse_err}")
                return [Section(id=1, section_title="Parsing Error", text=str(parse_err))]
        else:
            print(f"[ERROR] API call failed with status {response.status_code}")
            return [Section(id=1, section_title="AI Error", text=f"Status {response.status_code}: {response.text}")]

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
                    "- Place of Birth\ n- Address\n- Parents/Guardian\n"
                    "- Entrance Data\n- Course\n\n"
                    "Return a single JSON object with exactly these keys:\n"
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

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": 0.3
        }

        response = requests.post(OPENROUTER_URL, headers=headers, json=data)
        print(f"[DEBUG] OpenRouter status: {response.status_code}")
        print(f"[DEBUG] Response text:\n{response.text[:500]}")

        if response.status_code == 200:
            try:
                response_json = response.json()
                content = response_json.get("choices", [])[0].get("message", {}).get("content", "")

                def extract_json_string(text):
                    match = re.search(r'\{[\s\S]*?\}', text)
                    return match.group(0) if match else None

                json_str = extract_json_string(content)
                if not json_str:
                    raise ValueError("No valid JSON found in AI response.")

                student_data = json.loads(json_str)

                return [Section(
                    id=1,
                    sheet_name="Student Transcript",
                    row_number=1,
                    columns=student_data
                )]
            except Exception as parse_err:
                print(f"[ERROR] Failed to parse AI response: {parse_err}")
                return [Section(id=1, section_title="Parsing Error", text=str(parse_err))]
        else:
            print(f"[ERROR] API call failed with status {response.status_code}")
            return [Section(id=1, section_title="AI Error", text=f"Status {response.status_code}: {response.text}")]

    except Exception as e:
        print(f"[ERROR] Exception during Excel extraction: {str(e)}")
        return [Section(id=1, section_title="Exception", text=f"{str(e)}")]
    


