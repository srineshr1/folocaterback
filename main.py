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
model = genai.GenerativeModel("gemini-1.5-flash")

class ChatRequest(BaseModel):
    username: str
    message: str

# --- STEP 3: THE CHAT LOGIC ---
@app.post("/chat")
async def chat(request: ChatRequest):
    username = request.username
    user_text = request.message

    # 1. Fetch History (Last 5 conversations = 10 messages)
    cursor = messages_col.find({"username": username}).sort("timestamp", -1).limit(10)
    raw_history = await cursor.to_list(length=10)
    
    # 2. Format for Gemini (reverse to get Oldest -> Newest)
    gemini_history = []
    for msg in raw_history[::-1]:
        gemini_history.append({
            "role": "user" if msg["role"] == "user" else "model",
            "parts": [msg["text"]]
        })

    try:
        # 3. Get AI Response
        chat_session = model.start_chat(history=gemini_history)
        response = chat_session.send_message(user_text)
        ai_text = response.text

        # 4. Save to MongoDB
        now = datetime.now()
        await messages_col.insert_many([
            {"username": username, "role": "user", "text": user_text, "timestamp": now},
            {"username": username, "role": "model", "text": ai_text, "timestamp": now}
        ])

        return {"response": ai_text}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# For Railway deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)