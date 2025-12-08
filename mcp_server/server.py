from fastmcp import FastMCP
import httpx
import os
from pathlib import Path

# Initialize MCP Server
mcp = FastMCP("UFRO Orchestrator")

# Configuration
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8000/process")
CHATBOT_URL = os.getenv("CHATBOT_URL", "http://localhost:8001/api/ask")
API_TOKEN = os.getenv("API_TOKEN", "super-secret-token")

@mcp.tool()
async def identify_person(image_path: str) -> str:
    """
    Identifies a person from an image file path using the Orchestrator.
    Returns the identity and decision.
    Arguments:
        image_path: Absolute path to the image file.
    """
    if not os.path.exists(image_path):
        return f"Error: File not found at {image_path}"

    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    
    async with httpx.AsyncClient() as client:
        try:
            with open(image_path, "rb") as f:
                files = {"image": (Path(image_path).name, f, "image/jpeg")}
                response = await client.post(ORCHESTRATOR_URL, files=files, headers=headers)
                
            if response.status_code == 200:
                data = response.json()
                return f"Success! Identity: {data.get('identity')}"
            elif response.status_code == 403:
                return "User not identified (Unknown)."
            else:
                return f"Error {response.status_code}: {response.text}"
        except Exception as e:
            return f"Exception: {str(e)}"

@mcp.tool()
async def ask_normativa(question: str) -> str:
    """
    Asks a question about UFRO regulations (Normativa) to the RAG ChatBot.
    Returns the answer and citations.
    """
    async with httpx.AsyncClient() as client:
        try:
            payload = {"question": question, "provider": "chatgpt"}
            response = await client.post(CHATBOT_URL, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "")
                citations = data.get("citations", [])
                
                # Format simple output
                citation_text = "\n".join([f"- {c.get('doc_title', 'Doc')} ({c.get('doc_url', '')})" for c in citations])
                return f"{answer}\n\nFuentes:\n{citation_text}"
            else:
                return f"Error {response.status_code}: {response.text}"
        except Exception as e:
            return f"Exception: {str(e)}"

if __name__ == "__main__":
    mcp.run()
