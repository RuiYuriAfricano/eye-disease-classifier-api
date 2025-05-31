#!/usr/bin/env python3
"""
Script de teste para o cliente da API
"""

import sys
import os
from client.client import EyeDiseaseClassifierClient
import json

def test_api():
    """Testa a funcionalidade da API"""
    print("🧪 Testando API do Eye Disease Classifier")
    print("=" * 50)
    
    # Criar cliente
    client = EyeDiseaseClassifierClient("http://localhost:8000")
    
    # 1. Teste de saúde
    print("\n1️⃣ Testando saúde da API...")
    health = client.health_check()
    print(f"Resultado: {json.dumps(health, indent=2)}")
    
    if health.get("status") != "healthy":
        print("❌ API não está saudável. Verifique se está rodando.")
        return False
    
    # 2. Teste de classes
    print("\n2️⃣ Obtendo classes disponíveis...")
    classes = client.get_classes()
    print(f"Classes: {json.dumps(classes, indent=2)}")
    
    # 3. Teste de predição (se houver imagem de exemplo)
    example_images = ["exemplo.jpg", "test.jpg", "sample.png"]
    image_found = False
    
    for img_path in example_images:
        if os.path.exists(img_path):
            print(f"\n3️⃣ Testando predição com {img_path}...")
            result = client.predict_from_file(img_path)
            print(f"Resultado: {json.dumps(result, indent=2)}")
            image_found = True
            break
    
    if not image_found:
        print("\n3️⃣ Nenhuma imagem de exemplo encontrada.")
        print("Para testar predição, adicione uma imagem (exemplo.jpg, test.jpg, ou sample.png)")
    
    print("\n✅ Teste concluído!")
    return True

if __name__ == "__main__":
    try:
        test_api()
    except KeyboardInterrupt:
        print("\n\n⏹️ Teste interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro durante o teste: {e}")
        sys.exit(1)
