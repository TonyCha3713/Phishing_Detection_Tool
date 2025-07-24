from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from detector import analyze_email_with_chain
import email
import uvicorn

app = FastAPI(title="Phishing Detection API", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "Phishing Detection API is running!"}

@app.post("/analyze_eml")
async def analyze_eml(file: UploadFile = File(...)):
    try:
        eml_bytes = await file.read()
        msg = email.message_from_bytes(eml_bytes)
        subject = msg.get('Subject', '')
        sender = msg.get('From', '')
        # Extract body (handles plain text only for simplicity)
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body += part.get_payload(decode=True).decode(errors="ignore")
        else:
            body = msg.get_payload(decode=True).decode(errors="ignore")
        # Run hybrid model
        result = analyze_email_with_chain(body, sender, subject)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze EML: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": True}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 