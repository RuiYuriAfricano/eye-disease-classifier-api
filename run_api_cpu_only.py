#!/usr/bin/env python3
"""
Script para executar a API usando apenas CPU (versão ultra-segura)
"""

import os
import sys
import uvicorn

def main():
    """Executa a API em modo CPU-only"""
    
    # Configurações de ambiente para forçar CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_DISABLE_MKL'] = '1'
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    
    # Configurações do servidor
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("API_PORT", 8000)))
    
    print("🛡️ Iniciando API em modo CPU-only...")
    print(f"🌍 Host: {host}")
    print(f"🔌 Porta: {port}")
    print("📍 API estará disponível em: http://{}:{}".format(host, port))
    print("📖 Documentação automática em: /docs")
    print("🔄 Para parar a API, pressione Ctrl+C")
    print("🖥️ Modo: CPU APENAS (sem GPU)")
    print("-" * 50)
    
    try:
        # Executar servidor
        uvicorn.run(
            "api.main_cpu_only:app",
            host=host,
            port=port,
            reload=False,  # Desabilitar reload em produção
            access_log=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 API encerrada pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro ao iniciar API: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
