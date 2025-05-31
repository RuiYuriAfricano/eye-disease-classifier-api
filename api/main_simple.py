from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import gdown
import os
import io
import uvicorn
from typing import Dict, List
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurações do modelo
MODEL_PATH = "best_model.keras"
MODEL_URL = "https://drive.google.com/uc?id=1vSIfD3viT5JSxpG4asA8APCwK0JK9Dvu"
CLASS_NAMES = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

# Variável global para o modelo
model = "demo_mode"  # Iniciar sempre em modo demo

app = FastAPI(
    title="Eye Disease Classifier API",
    description="API para classificação de doenças oculares usando deep learning",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Evento executado na inicialização da API"""
    logger.info("🚀 API iniciada em modo demo")
    logger.info("📍 Todas as predições serão simuladas")
    logger.info("✅ API pronta para uso!")

@app.get("/")
async def root():
    """Endpoint raiz com informações da API"""
    return {
        "message": "Eye Disease Classifier API",
        "version": "1.0.0",
        "status": "running",
        "mode": "demo",
        "model_loaded": False,
        "classes": CLASS_NAMES
    }

@app.get("/health")
async def health_check():
    """Endpoint para verificar saúde da API"""
    return {
        "status": "healthy",
        "message": "API funcionando em modo demo",
        "model_loaded": False,
        "demo_mode": True
    }

@app.get("/status")
async def get_status():
    """Retorna status detalhado da API e modelo"""
    return {
        "api_status": "running",
        "model_status": {
            "loaded": False,
            "demo_mode": True,
            "file_exists": os.path.exists(MODEL_PATH),
            "file_size_mb": 0,
            "expected_size_mb": 169
        },
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
            "classes": "/classes"
        }
    }

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)) -> Dict:
    """
    Endpoint para predição de doenças oculares (modo demo)
    """
    # Verificar tipo de arquivo
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Tipo de arquivo não suportado. Use JPG, JPEG ou PNG."
        )

    try:
        # Ler imagem
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        logger.info("🎭 Executando predição em modo demo")
        
        # Simular predição com valores aleatórios mas realistas
        import random
        random.seed(hash(str(image_data)) % 1000)  # Seed baseado na imagem para consistência
        
        # Gerar probabilidades simuladas
        probs = [random.uniform(0.1, 0.9) for _ in CLASS_NAMES]
        total = sum(probs)
        probs = [p/total for p in probs]  # Normalizar para somar 1
        
        predicted_class_idx = probs.index(max(probs))
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = max(probs)
        
        all_predictions = {
            class_name: float(prob)
            for class_name, prob in zip(CLASS_NAMES, probs)
        }
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_predictions": all_predictions,
            "status": "demo_mode",
            "message": "Esta é uma predição simulada para demonstração."
        }
        
    except Exception as e:
        logger.error(f"Erro na predição: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")

@app.get("/classes")
async def get_classes() -> Dict[str, List[str]]:
    """Retorna as classes disponíveis"""
    return {"classes": CLASS_NAMES}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
