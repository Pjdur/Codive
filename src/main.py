# Import necessary libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from google import genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the FastAPI app
app = FastAPI(
    title="Codive API",
    description="A powerful AI coding assistant built on Google Gemini Flash 2.0",
    version="1.0.0"
)

# Initialize the Google GenAI client
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

# Define request and response models
class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class CodeRequest(BaseModel):
    prompt: str
    conversation_history: Optional[List[Message]] = None
    temperature: Optional[float] = 0.7
    system_instruction: Optional[str] = None

class CodeResponse(BaseModel):
    completion: str
    tokens_used: int
    model: str
    success: bool
    error: Optional[str] = None

# API endpoint for generating code
@app.post("/generate_code", response_model=CodeResponse)
async def generate_code(request: CodeRequest):
    try:
        # Prepare the conversation
        messages = []
        if request.system_instruction:
            messages.append({"role": "system", "content": request.system_instruction})
        if request.conversation_history:
            messages.extend([msg.dict() for msg in request.conversation_history])
        messages.append({"role": "user", "content": request.prompt})

        # Generate the response
        response = client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=messages,
            generation_config={
                "temperature": request.temperature,
                "max_output_tokens": 2048
            }
        )
        
        return {
            "completion": response.text,
            "tokens_used": len(response.text.split()),
            "model": 'gemini-2.0-flash-001',
            "success": True
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "completion": "",
                "tokens_used": 0,
                "model": 'gemini-2.0-flash-001',
                "success": False,
                "error": str(e)
            }
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "gemini-2.0-flash-001"}

# Run the app using Uvicorn (if running locally)
# Command: uvicorn main:app --reload

