# --- run_all.ps1 ---
# 1. Start FastAPI ML service
Write-Host "Starting FastAPI..."
cd .\ml-service
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
Start-Process uvicorn "app:app --reload --port 8000"

# 2. Start Spring Boot
Write-Host "Starting Spring Boot..."
cd ..\backend-spring
Start-Process .\mvnw spring-boot:run

# 3. Open browser to check form
Start-Process "http://localhost:8080/api/check"
