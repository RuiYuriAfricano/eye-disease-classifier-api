# 🔬 Eye Disease Classifier API - Execução Local

Guia completo para executar a API de classificação de doenças oculares localmente, sem Docker.

## 🎯 Visão Geral

Esta API utiliza deep learning (TensorFlow/Keras) para classificar doenças oculares em imagens. Configurada para rodar localmente usando apenas CPU, ideal para desenvolvimento e testes.

### Classes Suportadas
- **Normal**: Olho saudável
- **Cataract**: Catarata
- **Diabetic Retinopathy**: Retinopatia diabética
- **Glaucoma**: Glaucoma

## 🚀 Configuração Rápida

### 1. Configuração Automática (Recomendado)

```bash
# Execute o script de configuração
python setup_local.py
```

Este script irá:
- ✅ Verificar versão do Python (3.8+)
- ✅ Instalar dependências automaticamente
- ✅ **Detectar modelos locais automaticamente**
- ✅ Criar configuração local
- ✅ Verificar recursos do sistema
- ✅ Criar scripts de execução

### 🎯 Modelos Locais Suportados

A API detecta automaticamente modelos em:
- `best_model.keras` (padrão)
- `best_model.h5`
- `model.keras` / `model.h5`
- `eye_disease_model.keras` / `eye_disease_model.h5`
- `models/best_model.keras` / `models/best_model.h5`

**Vantagens do modelo local:**
- ⚡ **Carregamento instantâneo** (sem download)
- 🔒 **Funciona offline** completamente
- 💾 **Economia de banda** (sem redownload)
- 🎯 **Prioridade automática** sobre download

### 2. Executar a API

**Windows:**
```bash
start_local.bat
```

**Linux/Mac:**
```bash
./start_local.sh
```

**Ou diretamente:**
```bash
python run_api_local.py
```

## 📋 Configuração Manual

### 1. Pré-requisitos

- **Python 3.8+** (recomendado 3.9 ou 3.10)
- **4GB+ RAM** (recomendado)
- **2GB+ espaço livre** (para modelo)
- **Conexão com internet** (download inicial do modelo)

### 2. Instalação de Dependências

```bash
# Atualizar pip
python -m pip install --upgrade pip

# Instalar dependências
pip install -r requirements.txt
```

### 3. Verificar Instalação

```bash
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import fastapi; print('FastAPI instalado com sucesso')"
```

## ⚙️ Configurações

### Arquivo `.env.local` (criado automaticamente)

```bash
# Configuração Local - Eye Disease Classifier API
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
```

### Personalizar Configurações

Edite `run_api_local.py` para alterar:
- **Porta**: Altere `port = 8000`
- **Host**: Altere `host = "127.0.0.1"`
- **Threads**: Modifique variáveis de ambiente TensorFlow

## 🌐 Endpoints da API

### Principais

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/` | GET | Informações da API |
| `/health` | GET | Status de saúde |
| `/predict` | POST | Classificar imagem |
| `/classes` | GET | Classes disponíveis |
| `/status` | GET | Status detalhado |
| `/docs` | GET | Documentação Swagger |

### Exemplos de Uso

#### 1. Verificar Status
```bash
curl http://127.0.0.1:8000/health
```

#### 2. Classificar Imagem
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@imagem_olho.jpg"
```

#### 3. Obter Classes
```bash
curl http://127.0.0.1:8000/classes
```

## 🐍 Cliente Python

### Uso Básico

```python
from client.client import EyeDiseaseClassifierClient

# Criar cliente
client = EyeDiseaseClassifierClient("http://127.0.0.1:8000")

# Verificar saúde
health = client.health_check()
print("Status:", health)

# Classificar imagem
result = client.predict_from_file("imagem_olho.jpg")
print("Doença detectada:", result['predicted_class'])
print("Confiança:", f"{result['confidence']:.2%}")
```

### Exemplo Completo

```python
import json
from client.client import EyeDiseaseClassifierClient

def main():
    # Configurar cliente
    client = EyeDiseaseClassifierClient("http://127.0.0.1:8000")
    
    # Verificar se API está funcionando
    health = client.health_check()
    if health.get("status") != "healthy":
        print("❌ API não está funcionando")
        return
    
    print("✅ API funcionando!")
    
    # Obter classes disponíveis
    classes = client.get_classes()
    print("Classes:", classes["classes"])
    
    # Classificar imagem (substitua pelo caminho real)
    image_path = "exemplo_olho.jpg"
    try:
        result = client.predict_from_file(image_path)
        
        print("\n🔮 Resultado da Classificação:")
        print(f"Doença: {result['predicted_class']}")
        print(f"Confiança: {result['confidence']:.2%}")
        
        print("\n📊 Todas as Probabilidades:")
        for disease, prob in result['all_predictions'].items():
            print(f"  {disease}: {prob:.2%}")
            
    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {image_path}")
    except Exception as e:
        print(f"❌ Erro: {e}")

if __name__ == "__main__":
    main()
```

## 🔧 Solução de Problemas

### Erro: "Modelo não carregado"
**Causa**: Modelo ainda está sendo baixado ou falhou
**Solução**:
1. **Se você tem modelo local**: Coloque em uma das localizações suportadas
2. Aguarde alguns minutos (primeira execução com download)
3. Verifique conexão com internet (para download)
4. Verifique espaço em disco (2GB+)
5. Reinicie a API

### Como usar modelo já baixado
**Se você já tem o modelo:**
1. Renomeie para `best_model.keras` ou `best_model.h5`
2. Coloque na pasta `ai-kumona-classifier/`
3. Execute `python run_api_local.py`
4. A API detectará automaticamente

### Erro: "TensorFlow não encontrado"
**Solução**:
```bash
pip install tensorflow==2.13.0
```

### Erro: "Porta já em uso"
**Solução**:
1. Altere a porta em `run_api_local.py`
2. Ou mate o processo existente:
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

### Performance Lenta
**Soluções**:
1. **Aumentar threads**: Edite variáveis TF_NUM_*_THREADS
2. **Mais RAM**: Feche outros programas
3. **SSD**: Mova projeto para SSD se possível

### Erro de Memória
**Soluções**:
1. **Fechar programas**: Libere RAM
2. **Reduzir threads**: TF_NUM_INTEROP_THREADS=1
3. **Reiniciar**: Reinicie o sistema

## 📊 Monitoramento

### Logs da API
A API mostra logs detalhados:
- 🔄 Status do carregamento do modelo
- 💾 Uso de memória
- 🔮 Resultados de predições
- ❌ Erros e warnings

### Métricas de Performance
Acesse `/memory` para ver:
- Uso de RAM do processo
- Uso de RAM do sistema
- Status do modelo

## 🎯 Integração com Frontend

### React/JavaScript
```javascript
const classifyImage = async (imageFile) => {
  const formData = new FormData();
  formData.append('file', imageFile);
  
  const response = await fetch('http://127.0.0.1:8000/predict', {
    method: 'POST',
    body: formData
  });
  
  return await response.json();
};
```

### Configurar CORS (se necessário)
A API já está configurada para aceitar requisições de qualquer origem durante desenvolvimento.

## 🚀 Próximos Passos

1. **Teste a API** com imagens de exemplo
2. **Integre com frontend** React
3. **Configure produção** quando necessário
4. **Monitore performance** e ajuste conforme necessário

---

**Desenvolvido para o projeto Kumona Vision** 👁️✨
