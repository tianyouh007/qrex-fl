"""QREX-FL FastAPI Application"""
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

def create_app():
    app = FastAPI(title="QREX-FL API", version="1.0.0")
    
    @app.get("/")
    async def root():
        return {"message": "QREX-FL API", "status": "operational"}
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "quantum_resistant": True}
    
    return app

if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
