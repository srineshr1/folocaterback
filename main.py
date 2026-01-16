import os
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# --- STEP 1: CONFIGURE CORS ---
# Replace the URL below with your actual Firebase URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For testing; change to ["https://your-app.web.app"] later
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- STEP 2: SETUP DB & AI ---
client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
db = client["ChatApp"]
messages_col = db["messages"]

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash-lite")

class ChatRequest(BaseModel):
    username: str
    message: str

# --- STEP 3: THE CHAT LOGIC ---
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        username = request.username
        user_text = request.message

        # Fetch History
        cursor = messages_col.find({"username": username}).sort("timestamp", -1).limit(10)
        raw_history = await cursor.to_list(length=10)
        
        gemini_history = []
        for msg in raw_history[::-1]:
            gemini_history.append({
                "role": "user" if msg["role"] == "user" else "model",
                "parts": [msg["text"]]
            })

        # AI Logic
        chat_session = model.start_chat(history=gemini_history)
        response = chat_session.send_message(user_text)
        ai_text = response.text

        # Save to DB
        now = datetime.now()
        await messages_col.insert_many([
            {"username": username, "role": "user", "text": user_text, "timestamp": now},
            {"username": username, "role": "model", "text": ai_text, "timestamp": now}
        ])

        return {"response": ai_text}

    except Exception as e:
        # THIS WILL PRINT THE ACTUAL ERROR TO YOUR RENDER LOGS
        print(f"CRITICAL ERROR: {str(e)}") 
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{username}")
async def get_history(username: str):
    # Find messages for this user, newest first
    cursor = messages_col.find({"username": username}).sort("timestamp", -1).limit(10)
    raw_history = await cursor.to_list(length=10)
    
    # Format them so the frontend can read them easily
    return [{"role": m["role"], "text": m["text"]} for m in raw_history]