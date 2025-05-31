#!/usr/bin/env python3
"""
Script para executar a aplicação Streamlit
"""

import streamlit.web.cli as stcli
import sys
import os

if __name__ == "__main__":
    print("🎨 Iniciando aplicação Streamlit...")
    print("📍 App estará disponível em: http://localhost:8501")
    print("🔄 Para parar o app, pressione Ctrl+C")
    print("-" * 50)
    
    # Executar streamlit
    sys.argv = ["streamlit", "run", "app.py", "--server.port=8501"]
    sys.exit(stcli.main())
