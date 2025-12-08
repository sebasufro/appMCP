Write-Host "Deteniendo servicios Python..." -ForegroundColor Yellow

# Opción A: Matar todos los procesos python (Agresivo pero efectivo en dev local)
taskkill /IM python.exe /F
# Ojo: Esto cerrará cualquier otro script de python que tengas corriendo.

# Opción B (Más segura si sabes el título de la ventana, pero start_services no pone títulos específicos)
# Si te molestan las ventanas abiertas, ciérralas manualmente con la X.

Write-Host "Se han enviado señales de cierre a Python." -ForegroundColor Green
