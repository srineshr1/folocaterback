import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime

# 1. Load Environment Variables
load_dotenv()

app = FastAPI()

# 2. CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Database & AI Setup
MONGO_URI = os.getenv("MONGO_URI")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if not MONGO_URI:
    raise ValueError("MONGO_URI is missing from .env")
if not GEMINI_KEY:
    raise ValueError("GEMINI_API_KEY is missing from .env")

# Connect to MongoDB (Async)
mongo_client = AsyncIOMotorClient(MONGO_URI)
db = mongo_client["chat_db"]           # Database name
chat_collection = db["conversations"]  # Collection name

# Connect to Gemini
client = genai.Client(api_key=GEMINI_KEY)

# 4. Data Models
class ChatRequest(BaseModel):
    username: str
    message: str

# 5. Helper: Format History for Gemini
# Gemini 2.x expects contents as a list of strings or proper Content objects
# We will format the DB history into a simple conversation string for context
async def get_formatted_history(username: str, limit: int = 10):
    # Fetch last 'limit' messages, oldest first
    cursor = chat_collection.find({"username": username}).sort("timestamp", 1).limit(limit)
    history_text = ""
    async for doc in cursor:
        role_label = "User" if doc["role"] == "user" else "Model"
        history_text += f"{role_label}: {doc['text']}\n"
    return history_text

# --- ENDPOINTS ---

@app.get("/history/{username}")
async def get_history(username: str):
    """
    Retrieves past chat history for the frontend to display on login.
    """
    history = []
    # Sort by timestamp ascending (oldest to newest)
    cursor = chat_collection.find({"username": username}).sort("timestamp", 1)
    async for doc in cursor:
        history.append({
            "role": doc["role"], # "user" or "model"
            "text": doc["text"],
            "timestamp": doc["timestamp"].isoformat()
        })
    return history

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # A. Save User Message to DB
        await chat_collection.insert_one({
            "username": request.username,
            "role": "user",
            "text": request.message,
            "timestamp": datetime.utcnow()
        })

        # B. Retrieve Context (History)
        # We fetch previous chats so Gemini knows the context
        conversation_context = await get_formatted_history(request.username)
        
        # C. Construct Prompt with Context
        # We wrap the context and the new message
        full_prompt = (
            f"You are a helpful assistant. Here is the conversation history:\n"
            f"{conversation_context}\n"
            f"User: {request.message}\n"
            f"Model:"
        )

        # D. Call Gemini
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", 
            contents=full_prompt
        )
        reply_text = response.text

        # E. Save Model Response to DB
        await chat_collection.insert_one({
            "username": request.username,
            "role": "model",
            "text": reply_text,
            "timestamp": datetime.utcnow()
        })
        
        return {"response": reply_text}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)