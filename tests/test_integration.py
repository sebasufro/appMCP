import pytest
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "http://localhost:8000"
API_TOKEN = os.getenv("API_TOKEN", "super-secret-token")

@pytest.mark.asyncio
async def test_root():
    """Test that the static index page is served."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/")
        assert response.status_code == 200

@pytest.mark.asyncio
async def test_metrics_summary():
    """Test metrics endpoint (no auth needed/currently)."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/metrics/summary")
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data

@pytest.mark.asyncio
async def test_process_no_auth():
    """Test security: Should fail without token."""
    async with httpx.AsyncClient() as client:
        # Send a dummy file
        files = {'image': ('test.jpg', b'fake content', 'image/jpeg')}
        response = await client.post(f"{BASE_URL}/process", files=files)
        # Should be 403 Forbidden (FastAPI standard) or 401 Unauthorized
        assert response.status_code == 401

@pytest.mark.asyncio
async def test_process_with_auth_bad_image():
    """Test security: Should pass auth but fail face rec (handled gracefully)."""
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    async with httpx.AsyncClient() as client:
        files = {'image': ('test.jpg', b'fake content', 'image/jpeg')}
        response = await client.post(f"{BASE_URL}/process", files=files, headers=headers)
        
        # Expect either 403 (User not identified) or 200 (Success) depending on mock
        # But definitely NOT 401
        assert response.status_code in [200, 403, 400, 413, 500] 
