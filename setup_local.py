#!/usr/bin/env python3
"""
Script de configuração para execução local da API Eye Disease Classifier
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_header():
    """Imprime cabeçalho do script"""
    print("=" * 70)
    print("🔧 CONFIGURAÇÃO LOCAL - Eye Disease Classifier API")
    print("=" * 70)
    print()

def check_python_version():
    """Verifica se a versão do Python é compatível"""
    print("🐍 Verificando versão do Python...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} não é suportado")
        print("✅ Requerido: Python 3.8 ou superior")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
    return True

def check_pip():
    """Verifica se pip está disponível"""
    print("\n📦 Verificando pip...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        print("✅ pip está disponível")
        return True
    except subprocess.CalledProcessError:
        print("❌ pip não encontrado")
        return False

def install_requirements():
    """Instala dependências do requirements.txt"""
    print("\n📥 Instalando dependências...")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ Arquivo requirements.txt não encontrado")
        return False
    
    try:
        # Atualizar pip primeiro
        print("🔄 Atualizando pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True)
        
        # Instalar dependências
        print("🔄 Instalando dependências do requirements.txt...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        
        print("✅ Dependências instaladas com sucesso")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao instalar dependências: {e}")
        return False

def create_local_config():
    """Cria arquivo de configuração local"""
    print("\n⚙️ Criando configuração local...")
    
    config_content = """# Configuração Local - Eye Disease Classifier API
# Este arquivo é usado para desenvolvimento local

# Configurações de ambiente
ENVIRONMENT=development
DEBUG=true

# Configurações do TensorFlow (CPU apenas)
TF_CPP_MIN_LOG_LEVEL=3
CUDA_VISIBLE_DEVICES=-1
TF_FORCE_GPU_ALLOW_GROWTH=false
TF_ENABLE_ONEDNN_OPTS=0
TF_NUM_INTEROP_THREADS=2
TF_NUM_INTRAOP_THREADS=2
OMP_NUM_THREADS=2

# Configurações da API
API_HOST=127.0.0.1
API_PORT=8000
API_RELOAD=true

# Configurações do modelo
MODEL_PATH=best_model.keras
MODEL_URL=https://drive.google.com/uc?id=1vSIfD3viT5JSxpG4asA8APCwK0JK9Dvu
"""
    
    try:
        with open(".env.local", "w", encoding="utf-8") as f:
            f.write(config_content)
        print("✅ Arquivo .env.local criado")
        return True
    except Exception as e:
        print(f"❌ Erro ao criar configuração: {e}")
        return False

def check_local_models():
    """Verifica se existem modelos locais"""
    print("\n🔍 Verificando modelos locais...")

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
                    print(f"✅ Modelo encontrado: {model_path} ({file_size / 1_000_000:.1f}MB)")
            except:
                pass

    if found_models:
        print(f"🎯 {len(found_models)} modelo(s) local(is) detectado(s)")
        print("⚡ Carregamento será mais rápido (sem download)")
        return True
    else:
        print("📦 Nenhum modelo local encontrado")
        print("🔽 Modelo será baixado automaticamente (~169MB)")
        print("⏳ Primeira execução pode demorar alguns minutos")
        return False

def check_system_resources():
    """Verifica recursos do sistema"""
    print("\n💻 Verificando recursos do sistema...")

    try:
        import psutil

        # Memória RAM
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        print(f"🧠 RAM Total: {memory_gb:.1f} GB")

        if memory_gb < 4:
            print("⚠️ Aviso: Recomendado pelo menos 4GB de RAM")
        else:
            print("✅ RAM suficiente")

        # CPU
        cpu_count = psutil.cpu_count()
        print(f"⚡ CPUs: {cpu_count} cores")

        # Espaço em disco
        disk = psutil.disk_usage('.')
        disk_free_gb = disk.free / (1024**3)
        print(f"💾 Espaço livre: {disk_free_gb:.1f} GB")

        if disk_free_gb < 2:
            print("⚠️ Aviso: Recomendado pelo menos 2GB de espaço livre")
        else:
            print("✅ Espaço em disco suficiente")

    except ImportError:
        print("⚠️ psutil não disponível - pulando verificação de recursos")

def create_run_scripts():
    """Cria scripts de execução para diferentes sistemas"""
    print("\n📝 Criando scripts de execução...")
    
    # Script para Windows
    windows_script = """@echo off
echo 🚀 Iniciando Eye Disease Classifier API - Local
echo.
python run_api_local.py
pause
"""
    
    # Script para Linux/Mac
    unix_script = """#!/bin/bash
echo "🚀 Iniciando Eye Disease Classifier API - Local"
echo
python3 run_api_local.py
"""
    
    try:
        # Windows
        with open("start_local.bat", "w", encoding="utf-8") as f:
            f.write(windows_script)
        
        # Linux/Mac
        with open("start_local.sh", "w", encoding="utf-8") as f:
            f.write(unix_script)
        
        # Tornar executável no Unix
        if platform.system() != "Windows":
            os.chmod("start_local.sh", 0o755)
        
        print("✅ Scripts de execução criados:")
        print("   - Windows: start_local.bat")
        print("   - Linux/Mac: start_local.sh")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao criar scripts: {e}")
        return False

def print_instructions(has_local_model=False):
    """Imprime instruções finais"""
    print("\n" + "=" * 70)
    print("🎉 CONFIGURAÇÃO CONCLUÍDA!")
    print("=" * 70)
    print()
    print("📋 Para executar a API localmente:")
    print()

    if platform.system() == "Windows":
        print("   1. Execute: start_local.bat")
        print("   2. Ou execute: python run_api_local.py")
    else:
        print("   1. Execute: ./start_local.sh")
        print("   2. Ou execute: python3 run_api_local.py")

    print()
    print("🌐 A API estará disponível em:")
    print("   - URL principal: http://127.0.0.1:8000")
    print("   - Documentação: http://127.0.0.1:8000/docs")
    print("   - Status: http://127.0.0.1:8000/status")
    print()
    print("⚠️ IMPORTANTE:")
    print("   - A API usará APENAS CPU (sem GPU)")
    if has_local_model:
        print("   - ✅ Modelo local detectado - carregamento rápido")
        print("   - 🎯 Não será necessário download")
    else:
        print("   - 📦 Modelo será baixado automaticamente (~169MB)")
        print("   - ⏳ Primeira execução pode demorar alguns minutos")
    print()
    print("🔄 Para parar a API: Ctrl+C")
    print("=" * 70)

def main():
    """Função principal"""
    print_header()
    
    # Verificações
    if not check_python_version():
        sys.exit(1)
    
    if not check_pip():
        sys.exit(1)
    
    # Instalação
    if not install_requirements():
        sys.exit(1)
    
    # Configuração
    create_local_config()
    has_local_model = check_local_models()
    check_system_resources()
    create_run_scripts()
    
    # Instruções finais
    print_instructions(has_local_model)

if __name__ == "__main__":
    main()
