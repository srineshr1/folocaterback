import os
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# 1. Update the import
from google import genai 
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DB & NEW AI CLIENT SETUP ---
client_db = AsyncIOMotorClient(os.getenv("MONGO_URI"))
db = client_db["Chat"]         
messages_col = db["Chat"]   

# 2. Initialize the new Client
ai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class ChatRequest(BaseModel):
    username: str
    message: str

@app.get("/history/{username}")
async def get_history(username: str):
    cursor = messages_col.find({"username": username}).sort("timestamp", 1).limit(50)
    messages = await cursor.to_list(length=50)
    
    # Clean _id for JSON response
    for msg in messages:
        msg["_id"] = str(msg["_id"])
        
    return messages

@app.get("/")
async def root():
    return {"status": "Online", "message": "Backend is running with google-genai!"}

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        username = request.username
        user_text = request.message

        # Fetch History from MongoDB
        cursor = messages_col.find({"username": username}).sort("timestamp", -1).limit(10)
        raw_history = await cursor.to_list(length=10)
        
        # 3. Format history for the new SDK
        history_msgs = []
        for msg in raw_history[::-1]:
            history_msgs.append({
                "role": "user" if msg["role"] == "user" else "model",
                "parts": [{"text": msg["text"]}]
            })

        # 4. Use the new chat creation syntax
        chat_session = ai_client.chats.create(
            model="gemini-1.5-flash",
            history=history_msgs
        )
        
        response = chat_session.send_message(user_text)
        ai_text = response.text

        # Save to MongoDB (Same logic as before)
        now = datetime.now()
        await messages_col.insert_many([
            {"username": username, "role": "user", "text": user_text, "timestamp": now},
            {"username": username, "role": "model", "text": ai_text, "timestamp": now}
        ])

        return {"response": ai_text}

    except Exception as e:
        print(f"Chat Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))