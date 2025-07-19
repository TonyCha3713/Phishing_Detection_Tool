from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from detector import SMSPhishingDetector

app = FastAPI(title="SMS Phishing Detection API", version="1.0.0")

# Initialize the detector
print("Initializing SMS detector...")
detector = SMSPhishingDetector()
print("SMS detector initialized successfully!")

class SMSRequest(BaseModel):
    text: str

class SMSResponse(BaseModel):
    text: str
    risk_score: float
    rule_score: float
    similarity_score: float
    feature_score: float
    is_phishing: bool
    confidence: str
    analysis: str
    similar_messages: List[Dict[str, Any]]

@app.get("/")
async def root():
    return {"message": "SMS Phishing Detection API is running!"}

@app.post("/analyze", response_model=SMSResponse)
async def analyze_sms(request: SMSRequest):
    """Analyze an SMS for phishing indicators"""
    try:
        # Analyze the SMS
        result = detector.analyze_text_advanced(request.text)
        
        # Find similar messages
        similar_messages = detector.find_similar_messages(request.text, top_k=3)
        
        return SMSResponse(
            text=result['text'],
            risk_score=result['risk_score'],
            rule_score=result['rule_score'],
            similarity_score=result['similarity_score'],
            feature_score=result['feature_score'],
            is_phishing=result['is_phishing'],
            confidence=result['confidence'],
            analysis=result['analysis'],
            similar_messages=similar_messages
        )
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 