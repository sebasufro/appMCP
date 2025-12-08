$root = Resolve-Path "$PSScriptRoot/.."

Write-Host "Iniciando servicios del Proyecto..." -ForegroundColor Green

# 1. Face Recognition (Port 8002)
Write-Host "Iniciando Reconocimiento Facial (Port 8002)..."
Start-Process "python" -ArgumentList "Reconocimiento-Facial/api/app.py" -WorkingDirectory $root

# 2. ChatBot (Port 8001)
Write-Host "Iniciando ChatBot (Port 8001)..."
Start-Process "python" -ArgumentList "ChatBot/app.py" -WorkingDirectory $root

# 3. MCP Server (Port 8000)
Write-Host "Iniciando MCP Server (Port 8000)..."
Start-Process "python" -ArgumentList "api/app.py" -WorkingDirectory $root

Write-Host "¡Todos los servicios iniciados!" -ForegroundColor Yellow
Write-Host "Recuerda iniciar MongoDB y n8n por separado."
