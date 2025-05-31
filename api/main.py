from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
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
model = None

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

def download_and_load_model():
    """Baixa e carrega o modelo se necessário"""
    global model

    try:
        logger.info("🔄 Iniciando processo de carregamento do modelo...")

        # Verificar se o modelo já existe e está completo
        if os.path.exists(MODEL_PATH):
            file_size = os.path.getsize(MODEL_PATH)
            if file_size > 100_000_000:  # Pelo menos 100MB
                logger.info("✅ Modelo já existe e parece estar completo!")
            else:
                logger.info("⚠️ Arquivo do modelo existe mas parece incompleto. Removendo...")
                try:
                    os.remove(MODEL_PATH)
                except Exception as e:
                    logger.warning(f"Não foi possível remover arquivo incompleto: {e}")

        # Remover arquivos .part se existirem
        try:
            part_files = [f for f in os.listdir('.') if f.startswith('best_model.keras') and '.part' in f]
            for part_file in part_files:
                logger.info(f"🗑️ Removendo arquivo parcial: {part_file}")
                os.remove(part_file)
        except Exception as e:
            logger.warning(f"Erro ao limpar arquivos parciais: {e}")

        # Baixar modelo se não existir ou foi removido
        if not os.path.exists(MODEL_PATH):
            logger.info("🔽 Baixando modelo do Google Drive...")
            logger.info("📦 Tamanho esperado: ~169MB")
            logger.info("⏳ Isso pode levar alguns minutos...")

            try:
                # Usar gdown com configurações otimizadas
                gdown.download(
                    MODEL_URL,
                    MODEL_PATH,
                    quiet=False,
                    fuzzy=True  # Permite download mesmo com avisos do Google Drive
                )

                # Verificar se o download foi bem-sucedido
                if os.path.exists(MODEL_PATH):
                    file_size = os.path.getsize(MODEL_PATH)
                    logger.info(f"✅ Modelo baixado! Tamanho: {file_size / 1_000_000:.1f}MB")
                else:
                    raise Exception("Falha no download do modelo")
            except Exception as e:
                logger.error(f"❌ Erro no download: {e}")
                raise e

        # Carregar modelo
        logger.info("🔄 Carregando modelo...")
        try:
            # Tentar carregar o modelo com configurações otimizadas
            import os
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduzir logs do TensorFlow

            model = load_model(MODEL_PATH)
            logger.info("✅ Modelo carregado com sucesso!")
            logger.info(f"📊 Modelo pronto para classificar {len(CLASS_NAMES)} classes")

        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo do arquivo: {e}")
            # Se falhar ao carregar, tentar remover arquivo corrompido
            try:
                if os.path.exists(MODEL_PATH):
                    os.remove(MODEL_PATH)
                    logger.info("🗑️ Arquivo do modelo removido (possivelmente corrompido)")
            except Exception as cleanup_error:
                logger.warning(f"Erro ao limpar arquivo: {cleanup_error}")
            raise e

    except Exception as e:
        logger.error(f"❌ Erro geral no carregamento do modelo: {str(e)}")
        logger.info("🔄 Continuando em modo demo (predições simuladas)")
        model = "demo_mode"

@app.on_event("startup")
async def startup_event():
    """Evento executado na inicialização da API"""
    import threading

    logger.info("🚀 Iniciando API do Eye Disease Classifier...")
    logger.info("📍 API estará disponível para healthcheck imediatamente")
    logger.info("🔄 Modelo será carregado em background...")

    # Carregar modelo em thread separada para não bloquear a API
    def load_model_background():
        try:
            download_and_load_model()
        except Exception as e:
            logger.error(f"❌ Erro crítico no carregamento do modelo: {e}")
            # Garantir que a API continue funcionando em modo demo
            global model
            model = "demo_mode"

    thread = threading.Thread(target=load_model_background, daemon=True)
    thread.start()

    logger.info("✅ API iniciada com sucesso!")

@app.get("/")
async def root():
    """Endpoint raiz com informações da API"""
    return {
        "message": "Eye Disease Classifier API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "classes": CLASS_NAMES
    }

@app.get("/health")
async def health_check():
    """Endpoint para verificar saúde da API"""
    global model

    if model is None:
        return {
            "status": "starting",
            "message": "API iniciando, modelo sendo carregado...",
            "model_loaded": False
        }
    elif model == "demo_mode":
        return {
            "status": "healthy",
            "message": "API funcionando em modo demo",
            "model_loaded": False,
            "demo_mode": True
        }
    else:
        return {
            "status": "healthy",
            "message": "API funcionando com modelo carregado",
            "model_loaded": True
        }

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocessa a imagem para o modelo"""
    # Converter para RGB se necessário
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Redimensionar para 256x256
    image = image.resize((256, 256))
    
    # Converter para array numpy
    img_array = np.array(image)
    
    # Preprocessar usando inception_v3
    img_array = preprocess_input(img_array)
    
    # Adicionar dimensão do batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)) -> Dict:
    """
    Endpoint para predição de doenças oculares

    Args:
        file: Arquivo de imagem (JPG, JPEG, PNG)

    Returns:
        Dict com predição e confiança
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo não carregado")

    # Verificar tipo de arquivo
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Tipo de arquivo não suportado. Use JPG, JPEG ou PNG."
        )

    try:
        # Ler e processar imagem
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Verificar se está em modo demo
        if model == "demo_mode":
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
                "message": "Esta é uma predição simulada. O modelo real não está carregado."
            }

        # Preprocessar imagem para o modelo real
        processed_image = preprocess_image(image)

        # Fazer predição com o modelo real
        prediction = model.predict(processed_image)[0]

        # Obter classe predita e confiança
        predicted_class_idx = np.argmax(prediction)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(np.max(prediction))

        # Criar resposta com todas as probabilidades
        all_predictions = {
            class_name: float(prob)
            for class_name, prob in zip(CLASS_NAMES, prediction)
        }

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_predictions": all_predictions,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Erro na predição: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")

@app.get("/classes")
async def get_classes() -> Dict[str, List[str]]:
    """Retorna as classes disponíveis"""
    return {"classes": CLASS_NAMES}

@app.get("/status")
async def get_status():
    """Retorna status detalhado da API e modelo"""
    global model

    # Verificar se arquivo do modelo existe
    model_exists = os.path.exists(MODEL_PATH)
    model_size = 0
    if model_exists:
        model_size = os.path.getsize(MODEL_PATH)

    return {
        "api_status": "running",
        "model_status": {
            "loaded": model is not None and model != "demo_mode",
            "demo_mode": model == "demo_mode",
            "file_exists": model_exists,
            "file_size_mb": round(model_size / 1_000_000, 2) if model_size > 0 else 0,
            "expected_size_mb": 169
        },
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
            "classes": "/classes"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
