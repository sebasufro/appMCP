#!/bin/bash

# Navigate to project root (one level up from scripts/)
cd "$(dirname "$0")/.."

# Ensure logs directory exists
mkdir -p logs

echo "Starting AppMCP Services..."

# 1. Face Recognition (Port 8002)
# Using nohup to keep running after logout, sending output to log file
nohup python3 Reconocimiento-Facial/api/app.py > logs/face_rec.log 2>&1 &
echo "Started Face Recognition (Port 8002) - PID: $!"

# 2. ChatBot (Port 8001)
nohup python3 ChatBot/app.py > logs/chatbot.log 2>&1 &
echo "Started ChatBot (Port 8001) - PID: $!"

# 3. MCP Server (Port 8000)
nohup python3 MCP-Server/main.py > logs/mcp_server.log 2>&1 &
echo "Started MCP Server (Port 8000) - PID: $!"

echo "---------------------------------------------------"
echo "All services are running in the background!"
echo "You can check logs with: tail -f logs/*.log"
echo "To stop them, use 'pkill -f python3'"
echo "---------------------------------------------------"
