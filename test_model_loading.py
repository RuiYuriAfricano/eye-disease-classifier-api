#!/usr/bin/env python3
"""
Script para testar o carregamento robusto do modelo
"""

import requests
import time
import json
from typing import Dict

API_URL = "http://localhost:8000"

def test_api_endpoint(endpoint: str, method: str = "GET") -> Dict:
    """Testa um endpoint da API"""
    try:
        url = f"{API_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, timeout=10)
        
        response.raise_for_status()
        return {"success": True, "data": response.json()}
    except Exception as e:
        return {"success": False, "error": str(e)}

def monitor_model_loading(max_wait_minutes: int = 10):
    """Monitora o carregamento do modelo"""
    print("🔄 Monitorando carregamento do modelo...")
    print(f"⏰ Tempo máximo de espera: {max_wait_minutes} minutos")
    
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    
    while time.time() - start_time < max_wait_seconds:
        # Verificar status
        health_result = test_api_endpoint("/health")
        
        if health_result["success"]:
            health_data = health_result["data"]
            status = health_data.get("status", "unknown")
            model_loaded = health_data.get("model_loaded", False)
            memory_usage = health_data.get("memory_usage", {})
            
            elapsed_time = int(time.time() - start_time)
            
            print(f"[{elapsed_time:03d}s] Status: {status} | Modelo: {'✅' if model_loaded else '❌'} | Memória: {memory_usage.get('rss_mb', 0):.1f}MB")
            
            if model_loaded:
                print("🎉 Modelo carregado com sucesso!")
                return True
            elif status == "healthy" and health_data.get("demo_mode"):
                print("⚠️ API funcionando em modo demo")
                error = health_data.get("error")
                if error:
                    print(f"❌ Erro: {error}")
                return False
        else:
            print(f"❌ Erro ao conectar com a API: {health_result['error']}")
        
        time.sleep(5)
    
    print(f"⏰ Timeout após {max_wait_minutes} minutos")
    return False

def test_all_endpoints():
    """Testa todos os endpoints da API"""
    endpoints = [
        ("/", "GET"),
        ("/health", "GET"),
        ("/status", "GET"),
        ("/classes", "GET"),
        ("/memory", "GET")
    ]
    
    print("\n🧪 Testando endpoints...")
    
    for endpoint, method in endpoints:
        result = test_api_endpoint(endpoint, method)
        status = "✅" if result["success"] else "❌"
        print(f"{status} {method} {endpoint}")
        
        if result["success"]:
            # Mostrar informações relevantes
            data = result["data"]
            if endpoint == "/health":
                print(f"   Status: {data.get('status')}")
                print(f"   Modelo: {'Carregado' if data.get('model_loaded') else 'Não carregado'}")
            elif endpoint == "/memory":
                process = data.get("process", {})
                system = data.get("system", {})
                print(f"   Processo: {process.get('rss_mb', 0):.1f}MB")
                print(f"   Sistema: {system.get('used_mb', 0):.1f}MB / {system.get('total_mb', 0):.1f}MB")
            elif endpoint == "/status":
                model_status = data.get("model_status", {})
                print(f"   Arquivo existe: {model_status.get('file_exists')}")
                print(f"   Tamanho: {model_status.get('file_size_mb', 0):.1f}MB")
                print(f"   Integridade: {model_status.get('integrity_valid')}")
        else:
            print(f"   Erro: {result['error']}")

def test_reload_model():
    """Testa o endpoint de reload do modelo"""
    print("\n🔄 Testando reload do modelo...")
    
    result = test_api_endpoint("/reload-model", "POST")
    if result["success"]:
        print("✅ Reload iniciado com sucesso")
        print("⏳ Aguardando carregamento...")
        
        # Monitorar o carregamento
        success = monitor_model_loading(max_wait_minutes=5)
        return success
    else:
        print(f"❌ Erro ao iniciar reload: {result['error']}")
        return False

def main():
    """Função principal"""
    print("🚀 Teste do Sistema de Carregamento Robusto do Modelo")
    print("=" * 60)
    
    # Verificar se a API está rodando
    print("1. Verificando conexão com a API...")
    health_result = test_api_endpoint("/health")
    
    if not health_result["success"]:
        print(f"❌ Não foi possível conectar com a API: {health_result['error']}")
        print("💡 Certifique-se de que a API está rodando em http://localhost:8000")
        return
    
    print("✅ API conectada!")
    
    # Testar endpoints
    test_all_endpoints()
    
    # Verificar se o modelo já está carregado
    health_data = health_result["data"]
    if health_data.get("model_loaded"):
        print("\n✅ Modelo já está carregado!")
    else:
        print("\n⚠️ Modelo não está carregado")
        
        # Verificar se está carregando
        if health_data.get("loading"):
            print("⏳ Modelo está sendo carregado...")
            monitor_model_loading()
        else:
            # Tentar forçar reload
            print("🔄 Tentando forçar carregamento...")
            test_reload_model()
    
    print("\n📊 Teste concluído!")

if __name__ == "__main__":
    main()
