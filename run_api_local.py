#!/usr/bin/env python3
"""
Script para executar a API do Eye Disease Classifier LOCALMENTE
Configurado para desenvolvimento local sem Docker
"""

import uvicorn
import sys
import os
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Adicionar o diretório atual ao path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

def setup_local_environment():
    """Configura ambiente para execução local"""
    
    # Forçar uso de CPU apenas
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_NUM_INTEROP_THREADS'] = '2'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '2'
    os.environ['OMP_NUM_THREADS'] = '2'
    
    # Configurações para desenvolvimento local
    os.environ['ENVIRONMENT'] = 'development'
    os.environ['DEBUG'] = 'true'
    
    logger.info("✅ Ambiente local configurado")

def check_dependencies():
    """Verifica se todas as dependências estão instaladas"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'tensorflow',
        'PIL',
        'numpy',
        'gdown',
        'psutil'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        logger.error(f"❌ Pacotes faltando: {', '.join(missing_packages)}")
        logger.error("Execute: pip install -r requirements.txt")
        return False

    logger.info("✅ Todas as dependências estão instaladas")
    return True

def check_local_models():
    """Verifica se existem modelos locais disponíveis"""
    logger.info("🔍 Verificando modelos locais...")

    possible_models = [
        "best_model.keras",
        "best_model.h5",
        "model.keras",
        "model.h5",
        "eye_disease_model.keras",
        "eye_disease_model.h5",
        "models/best_model.keras",
        "models/best_model.h5"
    ]

    found_models = []

    for model_path in possible_models:
        if os.path.exists(model_path):
            try:
                file_size = os.path.getsize(model_path)
                if file_size > 50_000_000:  # Pelo menos 50MB
                    found_models.append((model_path, file_size))
                    logger.info(f"✅ Modelo encontrado: {model_path} ({file_size / 1_000_000:.1f}MB)")
            except:
                pass

    if found_models:
        logger.info(f"🎯 {len(found_models)} modelo(s) local(is) detectado(s)")
        logger.info("📁 A API usará automaticamente o modelo local")
        return True
    else:
        logger.info("📦 Nenhum modelo local encontrado")
        logger.info("🔽 Modelo será baixado automaticamente na primeira execução")
        return False

def main():
    """Função principal"""
    print("🚀 Iniciando API do Eye Disease Classifier - MODO LOCAL")
    print("=" * 60)
    
    # Configurar ambiente
    setup_local_environment()
    
    # Verificar dependências
    if not check_dependencies():
        sys.exit(1)

    # Verificar modelos locais
    has_local_model = check_local_models()
    
    # Configurações locais
    host = "127.0.0.1"  # Apenas localhost para desenvolvimento
    port = 8000
    
    print(f"📍 API estará disponível em: http://{host}:{port}")
    print(f"📖 Documentação automática em: http://{host}:{port}/docs")
    print(f"🔄 Swagger UI em: http://{host}:{port}/docs")
    print(f"📊 Status da API em: http://{host}:{port}/status")
    print(f"🏥 Health check em: http://{host}:{port}/health")
    print("🔄 Para parar a API, pressione Ctrl+C")
    print("🌍 Ambiente: DESENVOLVIMENTO LOCAL")
    print("💻 Usando APENAS CPU (sem GPU)")
    if has_local_model:
        print("🎯 Modelo local detectado - carregamento rápido")
    else:
        print("📦 Modelo será baixado automaticamente (~169MB)")
    print("=" * 60)
    
    try:
        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,
            reload=True,  # Auto-reload em desenvolvimento
            log_level="info",
            access_log=True,
            reload_dirs=[str(current_dir)],  # Monitorar apenas diretório atual
            reload_excludes=["*.keras", "*.h5", "*.pkl", "__pycache__"]  # Excluir arquivos grandes
        )
    except KeyboardInterrupt:
        print("\n🛑 API interrompida pelo usuário")
    except Exception as e:
        logger.error(f"❌ Erro ao iniciar API: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
