# Adventura RAGbot - Final Version with LLM-Based Context Detection

# =============================
# 📦 Install Dependencies
# =============================
!pip install -U transformers accelerate bitsandbytes peft psycopg2 datasets pyngrok
!pip install fastapi nest-asyncio uvicorn

# =============================
# 🔐 Tokens Configuration
# =============================
HUGGINGFACE_TOKEN = "your_huggingface_token"
NGROK_TOKEN = "your_ngrok_token"

!huggingface-cli login --token $HUGGINGFACE_TOKEN
!ngrok config add-authtoken $NGROK_TOKEN

# =============================
# 🧠 Model Setup (LLaMA/Mistral)
# =============================
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

llama_model_name = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
model = AutoModelForCausalLM.from_pretrained(
    llama_model_name,
    device_map="cuda",
    torch_dtype=torch.float16
)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print("✅ LLaMA/Mistral model loaded on GPU!")

# =============================
# 🌐 NGROK URL for API access
# =============================
from pyngrok import ngrok
public_url = ngrok.connect(8000)
print(f"🔗 Public FastAPI URL: {public_url}")

# =============================
# 🔌 PostgreSQL/REST Public URL
# =============================
NGROK_POSTGREST_URL = "https://f502-185-134-176-158.ngrok-free.app"  # <-- Change this to your backend ngrok URL

# =============================
# 🔎 Activity Retrieval Functions
# =============================
import requests

def retrieve_activities(category_id, location):
    url = f"{NGROK_POSTGREST_URL}/activities?category_id=eq.{category_id}&location=eq.{location}"
    r = requests.get(url)
    if r.status_code == 200 and r.json():
        return "\n".join([
            f"🔹 {a['name']} - {a['description']} (💰 ${a['price']}, ⏳ {a['duration']} mins, 🏆 Seats: {a['nb_seats']}, ✅ Available: {'Yes' if a['availability_status'] else 'No'})"
            for a in r.json()
        ])
    return "No available activities found."

def retrieve_activities_two(category_id, location):
    url = f"{NGROK_POSTGREST_URL}/activities?category_id=eq.{category_id}&location=eq.{location}&availability_status=is.true"
    r = requests.get(url)
    return [
        {
            "name": a['name'],
            "description": a['description'],
            "price": float(a['price']),
            "duration": int(a['duration']),
            "seats": int(a['nb_seats']),
            "location": a['location']
        } for a in r.json()
    ] if r.status_code == 200 and r.json() else []

# =============================
# 🎯 RAG Response Generator
# =============================
def generate_rag_response_two(user_query, category, location):
    cards = retrieve_activities_two(category, location)
    if not cards:
        retrieved_data = "No available activities were found in Adventura's database."
    else:
        retrieved_data = "\n".join([
            f"{i+1}. {c['name']} ({c['location']}): {c['description']}, Price: ${c['price']}, Duration: {c['duration']} mins, Seats: {c['seats']}"
            for i, c in enumerate(cards[:3])
        ])

    prompt = tokenizer.apply_chat_template([
        {
            "role": "user",
            "content": f"You are Adventura's assistant. Recommend activities based on the query:\n\n**User Query:** {user_query}\n\n**Available Activities:**\n{retrieved_data}\n\nMake it fun, short, and exciting."
        }
    ], tokenize=False)

    result = pipe(prompt, max_new_tokens=700, do_sample=True, temperature=0.7, top_p=0.9, top_k=50)[0]['generated_text']
    reply = result.split("[/INST]")[-1].strip()
    return { "chatbot_reply": reply, "cards": cards }

# =============================
# 🧠 LLM-based Context Detector
# =============================
def is_query_travel_related(user_query):
    prompt = f"You are a travel classifier agent. Is this query related to tourism, travel, or adventure in Lebanon?\nQuery: {user_query}\nReply with only 'yes' or 'no'."
    response = pipe(prompt, max_new_tokens=1, do_sample=False)[0]['generated_text'].strip().lower()
    return response.startswith("yes")

# =============================
# 📌 Unified Query Handler
# =============================
def handle_user_query(user_query, category, location):
    if not is_query_travel_related(user_query):
        return {
            "chatbot_reply": (
                "Hi there! 😊 I'm Adventura's assistant for travel in Lebanon!\n"
                "Ask me about trips, bookings, or destinations and let's plan your next adventure! 🌍"
            ),
            "cards": []
        }
    return generate_rag_response_two(user_query, category, location)

# =============================
# 🚀 FastAPI Setup
# =============================
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import nest_asyncio
import uvicorn

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)
nest_asyncio.apply()

@app.post("/chat")
async def chat_endpoint(request: Request):
    body = await request.json()
    user_query = body.get("query", "")
    category = body.get("category", 1)
    location = body.get("location", "Tripoli")
    print(f"📨 Received: {user_query}")
    response = handle_user_query(user_query, category, location)
    return JSONResponse(content=response)

uvicorn.run(app, host="0.0.0.0", port=8000)
