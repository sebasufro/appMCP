import os
from motor.motor_asyncio import AsyncIOMotorClient

# Global client to be reused
_client = None

async def get_db():
    global _client
    if _client is None:
        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        _client = AsyncIOMotorClient(mongo_uri)
    
    db_name = os.getenv("DB_NAME", "ufro_master")
    return _client[db_name]
