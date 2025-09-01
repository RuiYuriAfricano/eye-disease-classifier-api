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
import threading
import time
import hashlib
import gc
import psutil

# Configurar TensorFlow antes de importar - FORÇAR CPU APENAS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suprimir todos os logs do TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Desabilitar completamente CUDA/GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'  # Desabilitar GPU
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desabilitar otimizações OneDNN
os.environ['TF_DISABLE_MKL'] = '1'  # Desabilitar MKL que pode causar problemas
os.environ['TF_NUM_INTEROP_THREADS'] = '1'  # Limitar threads
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'  # Limitar threads
os.environ['OMP_NUM_THREADS'] = '1'  # Limitar OpenMP threads

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importar módulo de retreinamento (após configurar TensorFlow)
try:
    from retrain_endpoint import add_retrain_endpoint, RetrainRequest
    RETRAIN_AVAILABLE = True
    logger.info("✅ Módulo de retreinamento carregado")
except ImportError as e:
    RETRAIN_AVAILABLE = False
    logger.warning(f"⚠️ Módulo de retreinamento não disponível: {e}")

# Configurações do modelo
MODEL_PATH = "best_model.keras"
MODEL_URL = "https://drive.google.com/uc?id=1vSIfD3viT5JSxpG4asA8APCwK0JK9Dvu"
CLASS_NAMES = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
EXPECTED_MODEL_SIZE = 169_000_000  # 169MB em bytes
MODEL_HASH_FILE = "model_hash.txt"

# Possíveis localizações de modelos locais
POSSIBLE_MODEL_PATHS = [
    "best_model.keras",
    "best_model.h5",
    "model.keras",
    "model.h5",
    "eye_disease_model.keras",
    "eye_disease_model.h5",
    "models/best_model.keras",
    "models/best_model.h5",
    "../models/best_model.keras",
    "../models/best_model.h5"
]

# Variáveis globais para o modelo
model = None
model_loading = False
model_load_error = None
model_load_attempts = 0
MAX_LOAD_ATTEMPTS = 3

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

def get_memory_usage():
    """Retorna uso atual de memória"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        "rss_mb": round(memory_info.rss / 1024 / 1024, 2),
        "vms_mb": round(memory_info.vms / 1024 / 1024, 2),
        "percent": round(process.memory_percent(), 2)
    }

def calculate_file_hash(filepath):
    """Calcula hash MD5 do arquivo para verificar integridade"""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Erro ao calcular hash: {e}")
        return None

def verify_model_integrity():
    """Verifica se o modelo está íntegro - flexível para modelos locais"""
    if not os.path.exists(MODEL_PATH):
        return False, "Arquivo não existe"

    file_size = os.path.getsize(MODEL_PATH)

    # Verificação mais flexível para modelos locais
    min_size = 50_000_000  # 50MB mínimo (mais flexível)
    max_size = 300_000_000  # 300MB máximo (para detectar arquivos corrompidos)

    if file_size < min_size:
        return False, f"Arquivo muito pequeno: {file_size / 1_000_000:.1f}MB (mínimo: {min_size / 1_000_000:.1f}MB)"

    if file_size > max_size:
        return False, f"Arquivo muito grande: {file_size / 1_000_000:.1f}MB (máximo: {max_size / 1_000_000:.1f}MB)"

    # Verificar se é um arquivo Keras válido (extensão e estrutura básica)
    if not MODEL_PATH.endswith(('.keras', '.h5')):
        return False, "Extensão de arquivo inválida (deve ser .keras ou .h5)"

    # Verificar hash apenas se disponível (não obrigatório para modelos locais)
    if os.path.exists(MODEL_HASH_FILE):
        try:
            with open(MODEL_HASH_FILE, 'r') as f:
                expected_hash = f.read().strip()
            current_hash = calculate_file_hash(MODEL_PATH)
            if current_hash and current_hash != expected_hash:
                logger.warning("Hash do arquivo não confere, mas continuando (modelo local)")
        except Exception as e:
            logger.warning(f"Erro ao verificar hash: {e}")

    return True, f"Arquivo válido: {file_size / 1_000_000:.1f}MB"

def find_local_model():
    """Procura por modelos locais em diferentes localizações"""
    logger.info("🔍 Procurando por modelos locais...")

    for model_path in POSSIBLE_MODEL_PATHS:
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path)
            logger.info(f"📁 Modelo encontrado: {model_path} ({file_size / 1_000_000:.1f}MB)")

            # Verificar se parece ser um modelo válido
            if file_size > 50_000_000:  # Pelo menos 50MB
                logger.info(f"✅ Modelo local detectado: {model_path}")
                return model_path
            else:
                logger.warning(f"⚠️ Arquivo muito pequeno, ignorando: {model_path}")

    logger.info("❌ Nenhum modelo local encontrado")
    return None

def setup_model_path():
    """Configura o caminho do modelo, priorizando modelos locais"""
    global MODEL_PATH

    # Primeiro, tentar encontrar modelo local
    local_model = find_local_model()
    if local_model:
        MODEL_PATH = local_model
        logger.info(f"🎯 Usando modelo local: {MODEL_PATH}")
        return True

    # Se não encontrou, usar o caminho padrão
    logger.info(f"📦 Usando caminho padrão: {MODEL_PATH}")
    return False

def download_model_with_retry(max_retries=3):
    """Baixa o modelo com retry automático"""
    for attempt in range(max_retries):
        try:
            logger.info(f"🔽 Tentativa {attempt + 1}/{max_retries} de download...")

            # Limpar arquivos parciais
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)

            # Download com timeout
            gdown.download(
                MODEL_URL,
                MODEL_PATH,
                quiet=False,
                fuzzy=True
            )

            # Verificar integridade
            is_valid, message = verify_model_integrity()
            if is_valid:
                logger.info(f"✅ Download bem-sucedido! {message}")
                # Salvar hash para verificações futuras
                file_hash = calculate_file_hash(MODEL_PATH)
                if file_hash:
                    with open(MODEL_HASH_FILE, 'w') as f:
                        f.write(file_hash)
                return True
            else:
                logger.warning(f"⚠️ Download inválido: {message}")
                if attempt < max_retries - 1:
                    time.sleep(5)  # Aguardar antes da próxima tentativa

        except Exception as e:
            logger.error(f"❌ Erro no download (tentativa {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)

    return False

def configure_tensorflow():
    """Configura TensorFlow para uso APENAS de CPU (sem GPU)"""
    try:
        import tensorflow as tf

        # FORÇAR USO APENAS DE CPU
        tf.config.set_visible_devices([], 'GPU')

        # Configurar threads de forma muito conservadora
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)

        # Configurar uso de memória de forma conservadora
        tf.config.experimental.enable_memory_growth = False

        # Verificar se GPU foi realmente desabilitada
        gpus = tf.config.list_physical_devices('GPU')
        visible_gpus = tf.config.get_visible_devices('GPU')

        logger.info(f"🖥️ GPUs físicas detectadas: {len(gpus)}")
        logger.info(f"🖥️ GPUs visíveis (deve ser 0): {len(visible_gpus)}")
        logger.info("✅ TensorFlow configurado para CPU APENAS")

        return True

    except Exception as e:
        logger.error(f"❌ Erro ao configurar TensorFlow: {e}")
        return False

def load_model_safe():
    """Carrega o modelo com configurações de segurança máxima"""
    try:
        logger.info("🔄 Iniciando carregamento seguro do modelo...")

        # Configurar TensorFlow ANTES de qualquer importação
        configure_tensorflow()

        # Importar TensorFlow apenas quando necessário e após configuração
        logger.info("📦 Importando TensorFlow/Keras...")
        from keras.models import load_model
        from keras.applications.inception_v3 import preprocess_input

        # Verificar memória antes do carregamento
        memory_before = get_memory_usage()
        logger.info(f"💾 Memória antes do carregamento: {memory_before['rss_mb']}MB")

        # Carregar modelo com timeout implícito
        logger.info("🔄 Carregando modelo (pode demorar alguns minutos)...")
        logger.info("⚠️ Usando APENAS CPU para máxima estabilidade")

        try:
            model = load_model(MODEL_PATH, compile=False)  # Não compilar para economizar memória
            logger.info("✅ Modelo carregado sem compilação")

            # Compilar manualmente de forma mais segura
            logger.info("🔧 Compilando modelo...")
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            logger.info("✅ Modelo compilado com sucesso")

        except Exception as load_error:
            logger.error(f"❌ Erro no carregamento: {load_error}")
            # Tentar carregamento alternativo
            logger.info("🔄 Tentando carregamento alternativo...")
            model = load_model(MODEL_PATH)

        # Verificar memória após carregamento
        memory_after = get_memory_usage()
        logger.info(f"💾 Memória após carregamento: {memory_after['rss_mb']}MB")
        logger.info(f"📈 Incremento de memória: {memory_after['rss_mb'] - memory_before['rss_mb']:.1f}MB")

        # Fazer uma predição de teste simples
        logger.info("🧪 Testando modelo com predição simples...")
        test_input = np.random.random((1, 256, 256, 3)).astype(np.float32)
        test_prediction = model.predict(test_input, verbose=0)

        logger.info("✅ Modelo carregado e testado com sucesso!")
        logger.info(f"📊 Modelo pronto para classificar {len(CLASS_NAMES)} classes")

        return model

    except Exception as e:
        logger.error(f"❌ Erro ao carregar modelo: {e}")
        logger.error(f"🔍 Tipo do erro: {type(e).__name__}")

        # Forçar limpeza de memória
        logger.info("🧹 Limpando memória...")
        gc.collect()

        # Log de memória após erro
        memory_after_error = get_memory_usage()
        logger.info(f"💾 Memória após erro: {memory_after_error['rss_mb']}MB")

        raise e

def download_and_load_model():
    """Carrega modelo local ou baixa se necessário - com fallback robusto"""
    global model, model_loading, model_load_error, model_load_attempts

    if model_loading:
        logger.info("⏳ Carregamento já em andamento...")
        return

    model_loading = True
    model_load_attempts += 1

    try:
        logger.info(f"🔄 Iniciando carregamento do modelo (tentativa {model_load_attempts}/{MAX_LOAD_ATTEMPTS})...")

        # PRIORIDADE 0: Detectar modelos locais automaticamente
        setup_model_path()

        # PRIORIDADE 1: Verificar se existe modelo local válido
        if os.path.exists(MODEL_PATH):
            logger.info("📁 Modelo local encontrado, verificando integridade...")
            is_valid, message = verify_model_integrity()

            if is_valid:
                logger.info(f"✅ Modelo local válido: {message}")
                # Tentar carregar o modelo local
                try:
                    model = load_model_safe()
                    model_load_error = None
                    logger.info("🎉 Modelo local carregado com sucesso!")
                    return
                except Exception as load_error:
                    logger.warning(f"⚠️ Erro ao carregar modelo local: {load_error}")
                    logger.info("🔄 Tentando redownload...")
                    # Continuar para download se carregamento falhar
            else:
                logger.warning(f"⚠️ Modelo local inválido: {message}")
                logger.info("🗑️ Removendo modelo corrompido...")
                try:
                    os.remove(MODEL_PATH)
                except:
                    pass

        # PRIORIDADE 2: Tentar baixar modelo se não existe ou é inválido
        if not os.path.exists(MODEL_PATH):
            logger.info("🔽 Modelo local não encontrado, tentando download...")
            logger.info("📦 Tamanho esperado: ~169MB")
            logger.info("⏳ Isso pode levar alguns minutos...")

            try:
                if download_model_with_retry():
                    logger.info("✅ Download concluído, carregando modelo...")
                    model = load_model_safe()
                    model_load_error = None
                    logger.info("🎉 Modelo baixado e carregado com sucesso!")
                    return
                else:
                    raise Exception("Falha no download após múltiplas tentativas")
            except Exception as download_error:
                logger.error(f"❌ Erro no download: {download_error}")
                # Continuar para fallback

        # PRIORIDADE 3: Tentar carregar modelo mesmo com problemas de integridade
        if os.path.exists(MODEL_PATH):
            logger.info("🔄 Tentando carregar modelo mesmo com possíveis problemas...")
            try:
                model = load_model_safe()
                model_load_error = None
                logger.info("🎉 Modelo carregado com sucesso (ignorando verificação de integridade)!")
                return
            except Exception as final_load_error:
                logger.error(f"❌ Falha final no carregamento: {final_load_error}")

        # Se chegou aqui, todas as tentativas falharam
        raise Exception("Todas as tentativas de carregamento falharam")

    except Exception as e:
        error_msg = f"Erro no carregamento do modelo: {str(e)}"
        logger.error(f"❌ {error_msg}")
        model_load_error = error_msg

        # FALLBACK: Modo demo
        if model_load_attempts < MAX_LOAD_ATTEMPTS:
            logger.info(f"🔄 Tentativa {model_load_attempts + 1}/{MAX_LOAD_ATTEMPTS} será feita automaticamente")
            model = "demo_mode"
        else:
            logger.error("❌ Máximo de tentativas atingido. Continuando em modo demo.")
            logger.info("🎭 Modo demo permite testar a API sem modelo real")
            model = "demo_mode"

    finally:
        model_loading = False

@app.on_event("startup")
async def startup_event():
    """Evento executado na inicialização da API"""
    logger.info("🚀 Iniciando API do Eye Disease Classifier...")
    logger.info("📍 API estará disponível para healthcheck imediatamente")
    logger.info("🔄 Modelo será carregado em background...")

    # Mostrar informações do sistema
    memory_info = get_memory_usage()
    logger.info(f"💾 Memória inicial: {memory_info['rss_mb']}MB ({memory_info['percent']}%)")

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
    global model, model_loading, model_load_error, model_load_attempts

    memory_info = get_memory_usage()

    base_response = {
        "memory_usage": memory_info,
        "load_attempts": model_load_attempts,
        "max_attempts": MAX_LOAD_ATTEMPTS
    }

    if model is None:
        if model_loading:
            return {
                **base_response,
                "status": "loading",
                "message": "Modelo sendo carregado...",
                "model_loaded": False,
                "loading": True
            }
        else:
            return {
                **base_response,
                "status": "starting",
                "message": "API iniciando, modelo será carregado...",
                "model_loaded": False,
                "error": model_load_error
            }
    elif model == "demo_mode":
        return {
            **base_response,
            "status": "healthy",
            "message": "API funcionando em modo demo",
            "model_loaded": False,
            "demo_mode": True,
            "error": model_load_error
        }
    else:
        return {
            **base_response,
            "status": "healthy",
            "message": "API funcionando com modelo carregado",
            "model_loaded": True
        }

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocessa a imagem para o modelo"""
    try:
        # Converter para RGB se necessário
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Redimensionar para 256x256
        image = image.resize((256, 256))

        # Converter para array numpy
        img_array = np.array(image)

        # Preprocessar usando inception_v3
        from keras.applications.inception_v3 import preprocess_input
        img_array = preprocess_input(img_array)

        # Adicionar dimensão do batch
        img_array = np.expand_dims(img_array, axis=0)

        return img_array
    except Exception as e:
        logger.error(f"Erro no preprocessamento da imagem: {e}")
        raise e

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
    global model, model_loading, model_load_error, model_load_attempts

    # Verificar se arquivo do modelo existe
    model_exists = os.path.exists(MODEL_PATH)
    model_size = 0
    model_hash = None

    if model_exists:
        model_size = os.path.getsize(MODEL_PATH)
        model_hash = calculate_file_hash(MODEL_PATH)

    # Verificar integridade
    is_valid, integrity_message = verify_model_integrity() if model_exists else (False, "Arquivo não existe")

    memory_info = get_memory_usage()

    return {
        "api_status": "running",
        "model_status": {
            "loaded": model is not None and model != "demo_mode",
            "demo_mode": model == "demo_mode",
            "loading": model_loading,
            "file_exists": model_exists,
            "file_size_mb": round(model_size / 1_000_000, 2) if model_size > 0 else 0,
            "expected_size_mb": round(EXPECTED_MODEL_SIZE / 1_000_000, 2),
            "integrity_valid": is_valid,
            "integrity_message": integrity_message,
            "file_hash": model_hash[:16] + "..." if model_hash else None,
            "load_attempts": model_load_attempts,
            "max_attempts": MAX_LOAD_ATTEMPTS,
            "last_error": model_load_error
        },
        "system": {
            "memory_usage": memory_info,
            "tensorflow_configured": True
        },
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
            "classes": "/classes",
            "reload_model": "/reload-model",
            "memory": "/memory"
        }
    }

@app.post("/reload-model")
async def reload_model():
    """Força o recarregamento do modelo"""
    global model, model_loading, model_load_error, model_load_attempts

    if model_loading:
        return {
            "status": "error",
            "message": "Carregamento já em andamento"
        }

    # Reset das variáveis
    model = None
    model_load_error = None
    model_load_attempts = 0

    # Iniciar carregamento em background
    def reload_background():
        download_and_load_model()

    thread = threading.Thread(target=reload_background, daemon=True)
    thread.start()

    return {
        "status": "success",
        "message": "Recarregamento do modelo iniciado"
    }

@app.get("/memory")
async def get_memory_info():
    """Retorna informações detalhadas de memória"""
    memory_info = get_memory_usage()

    # Informações do sistema
    system_memory = psutil.virtual_memory()

    return {
        "process": memory_info,
        "system": {
            "total_mb": round(system_memory.total / 1024 / 1024, 2),
            "available_mb": round(system_memory.available / 1024 / 1024, 2),
            "used_mb": round(system_memory.used / 1024 / 1024, 2),
            "percent": round(system_memory.percent, 2)
        },
        "model_status": {
            "loaded": model is not None and model != "demo_mode",
            "demo_mode": model == "demo_mode"
        }
    }

# === ENDPOINTS DE RETREINAMENTO ===

if RETRAIN_AVAILABLE:
    try:
        add_retrain_endpoint(app, model)
        logger.info("✅ Endpoints de retreinamento adicionados")
    except Exception as e:
        logger.error(f"❌ Erro ao adicionar endpoints de retreinamento: {e}")
else:
    @app.post("/retrain")
    async def retrain_unavailable():
        """Endpoint de retreinamento não disponível"""
        raise HTTPException(
            status_code=503,
            detail="Retreinamento não disponível. Módulo não carregado."
        )

    @app.get("/retrain/status")
    async def retrain_status_unavailable():
        """Status do retreinamento quando não disponível"""
        return {
            "retrain_available": False,
            "error": "Módulo de retreinamento não carregado",
            "model_loaded": model is not None and model != "demo_mode"
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
