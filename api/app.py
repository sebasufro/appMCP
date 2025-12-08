import os
from pathlib import Path
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles # Import StaticFiles
from dotenv import load_dotenv
from uuid import uuid4
from datetime import datetime, timezone
from typing import Optional # Import Optional
import asyncio

# DB
from db.mongo import get_db

load_dotenv()

app = FastAPI(title="MCP Server Orchestrator")

# Mount Static Files
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Configuration
FACE_REC_URL = os.getenv("FACE_REC_URL", "http://localhost:8002/verify")
CHATBOT_URL = os.getenv("CHATBOT_URL", "http://localhost:8001/api/ask")
API_TOKEN = os.getenv("API_TOKEN", "super-secret-token")
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Serve Index
@app.get("/")
async def read_index():
    return FileResponse(STATIC_DIR / 'index.html')

# Helper to log service calls (Background or direct)
async def log_service_call(request_id: str, service_type: str, service_name: str, endpoint: str, latency_ms: float, status_code: int, result: dict = None, error: str = None):
    try:
        db = await get_db()
        doc = {
            "request_id": request_id,
            "ts": datetime.now(timezone.utc),
            "service_type": service_type,
            "service_name": service_name,
            "endpoint": endpoint,
            "latency_ms": latency_ms,
            "status_code": status_code,
            "result": result,
            "error": error
        }
        await db.service_logs.insert_one(doc)
    except Exception as e:
        print(f"Error logging service call: {e}")
        # Make sure we see what we missed
        if error:
            print(f"Service Call Failed! Service: {service_name}, Error: {error}")

@app.post("/process")
async def process_request(
    image: UploadFile = File(...),
    question: Optional[str] = Form(None), # Make question optional
    token: str = Depends(verify_token) # Security Requirement
):
    """
    Orchestrates the flow:
    1. Send image to Face Recognition service.
    2. If identified, send question to ChatBot (if provided).
    3. Return combined result.
    Logs interaction to MongoDB.
    """
    request_id = str(uuid4())
    start_time = datetime.now()
    
    # Default User context
    user_context = {"id": "unknown", "type": "unknown", "role": "basic"}
    
    decision = "unknown"
    errors = []
    
    identity_data = None
    chat_answer = None
    
    async with httpx.AsyncClient() as client:
        # 1. Face Recognition
        t0_face = datetime.now()
        face_status_code = 0
        face_error = None
        face_data = None
        
        try:
            image_content = await image.read()
            files = {'image': (image.filename, image_content, image.content_type)}
            
            face_response = await client.post(FACE_REC_URL, files=files)
            face_status_code = face_response.status_code
            
            if face_response.status_code == 200:
                face_data = face_response.json()
                # Updated logic for new Face Rec API
                is_me = face_data.get("is_me", False) 
                decision = "identified" if is_me else "unknown"
                identity_data = face_data
                
                if is_me:
                    user_context["type"] = "student"
                    user_context["id"] = face_data.get("detected_class", "unknown")
            else:
                face_error = face_response.text
                errors.append(f"FaceRec Error: {face_error}")

        except Exception as e:
            face_error = str(e)
            errors.append(f"FaceRec Exception: {e}")
        
        latency_face = (datetime.now() - t0_face).total_seconds() * 1000
        await log_service_call(request_id, "pp2", "FaceRec", FACE_REC_URL, latency_face, face_status_code, face_data, face_error)

        # 2. ChatBot (Only if identified AND question is present)
        if decision == "identified" and question:
            t0_chat = datetime.now()
            chat_status_code = 0
            chat_error = None
            chat_data = None
            
            try:
                chat_payload = {
                    "question": question,
                    "provider": "chatgpt" 
                }
                chat_response = await client.post(CHATBOT_URL, json=chat_payload)
                chat_status_code = chat_response.status_code
                
                if chat_response.status_code == 200:
                    chat_data = chat_response.json()
                    chat_answer = chat_data
                else:
                    chat_error = chat_response.text
                    errors.append(f"ChatBot Error: {chat_error}")

            except Exception as e:
                chat_error = str(e)
                errors.append(f"ChatBot Exception: {e}")
                
            latency_chat = (datetime.now() - t0_chat).total_seconds() * 1000
            await log_service_call(request_id, "pp1", "ChatBot", CHATBOT_URL, latency_chat, chat_status_code, chat_data, chat_error)
    
    # Logging Access Log
    total_latency = (datetime.now() - start_time).total_seconds() * 1000
    status_code = 200 if not errors else 500 # Simplified
    
    try:
        db = await get_db()
        access_doc = {
            "request_id": request_id,
            "ts": datetime.now(timezone.utc),
            "route": "/process",
            "user": user_context,
            "input": {"has_image": True, "has_question": bool(question)},
            "decision": decision,
            "identity": identity_data,
            "timing_ms": total_latency,
            "status_code": status_code,
            "errors": errors if errors else None,
            "pp1_used": (decision == "identified")
        }
        await db.access_logs.insert_one(access_doc)
    except Exception as e:
        print(f"Error logging access log: {e}")
        if errors:
            print(f"Request Failures: {errors}")

    # Response
    if decision != "identified":
        return JSONResponse(status_code=403, content={
            "status": "denied",
            "message": "User not identified",
            "face_data": identity_data
        })
    
    return {
        "status": "success",
        "identity": identity_data,
        "answer": chat_answer,
        "request_id": request_id
    }

# --- Metrics Endpoints ---

@app.get("/metrics/summary")
async def metrics_summary(days: int = 7):
    db = await get_db()
    # Simple count for now, advanced aggregation can be added as needed
    count = await db.access_logs.count_documents({})
    return {"total_requests": count, "days_window": days}

@app.get("/metrics/by-user-type")
async def metrics_by_user_type(days: int = 7):
    db = await get_db()
    pipeline = [
        {"$group": {"_id": "$user.type", "count": {"$sum": 1}}}
    ]
    cursor = db.access_logs.aggregate(pipeline)
    results = await cursor.to_list(length=100)
    return results

@app.get("/metrics/decisions")
async def metrics_decisions(days: int = 7):
    db = await get_db()
    pipeline = [
        {"$group": {"_id": "$decision", "count": {"$sum": 1}}}
    ]
    cursor = db.access_logs.aggregate(pipeline)
    results = await cursor.to_list(length=100)
    return results

@app.get("/metrics/services")
async def metrics_services(days: int = 7):
    db = await get_db()
    pipeline = [
        {"$group": {
            "_id": "$service_name", 
            "avg_latency": {"$avg": "$latency_ms"},
            "total_calls": {"$sum": 1},
            "errors": {"$sum": {"$cond": [{"$ifNull": ["$error", False]}, 1, 0]}}
        }}
    ]
    cursor = db.service_logs.aggregate(pipeline)
    results = await cursor.to_list(length=100)
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
