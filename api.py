import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

# Create app
app = FastAPI()

# Load API key from environment variable
API_KEY = os.getenv("API_KEY")

# Define API key header
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

# Verify API key function
def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )

# Input schema
class CropInput(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# Protected endpoint
@app.post("/crop-recommendation")
def recommend_crop(
    data: CropInput,
    _: str = Depends(verify_api_key)
):
    return {
        "recommended_crop": "Rice",
        "message": "API working successfully"
    }

