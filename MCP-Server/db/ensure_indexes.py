import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient

async def main():
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name = os.getenv("DB_NAME", "ufro_master")
    
    print(f"Connecting to {mongo_uri} (DB: {db_name})...")
    client = AsyncIOMotorClient(mongo_uri)
    db = client[db_name]

    print("Creating indexes...")
    
    # access_logs indexes
    await db.access_logs.create_index([("ts", -1)])
    await db.access_logs.create_index([("user.type", 1), ("ts", -1)])
    await db.access_logs.create_index([("route", 1), ("ts", -1)])
    await db.access_logs.create_index([("decision", 1), ("ts", -1)])
    
    # service_logs indexes
    await db.service_logs.create_index([("service_name", 1), ("ts", -1)])
    await db.service_logs.create_index([("service_type", 1), ("ts", -1)])

    print("Indexes created successfully.")

if __name__ == "__main__":
    asyncio.run(main())
