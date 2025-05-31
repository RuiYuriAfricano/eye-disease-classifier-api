#!/usr/bin/env python3
"""
Script para executar a API do Eye Disease Classifier
"""

import uvicorn
import sys
import os

# Adicionar o diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("🚀 Iniciando API do Eye Disease Classifier...")

    # Obter porta do ambiente (para hospedagem)
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"

    # Detectar se está em produção
    is_production = os.environ.get("RAILWAY_ENVIRONMENT") or os.environ.get("RENDER") or os.environ.get("HEROKU_APP_NAME")

    print(f"📍 API estará disponível em: http://{host}:{port}")
    print("📖 Documentação automática em: /docs")
    print("🔄 Para parar a API, pressione Ctrl+C")
    print(f"🌍 Ambiente: {'Produção' if is_production else 'Desenvolvimento'}")
    print("-" * 50)

    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=not is_production,  # Não recarregar em produção
        log_level="info",
        access_log=True
    )
