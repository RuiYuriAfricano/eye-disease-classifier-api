# 🔬 Eye Disease Classifier API

Uma API REST completa para classificação de doenças oculares usando deep learning, pronta para hospedagem e integração com aplicações React.

## 🎯 Funcionalidades

- **🤖 Classificação automática** de doenças oculares em imagens
- **🌐 API REST** com documentação automática (Swagger)
- **☁️ Pronta para hospedagem** (Railway, Render, Heroku, Docker)
- **⚡ Download automático** do modelo do Google Drive
- **🎭 Modo demo** quando o modelo não está disponível
- **🔒 CORS configurado** para integração com React
- **📱 Clientes de exemplo** (React, JavaScript, Python)

## 🏥 Classes Suportadas

- **Catarata** (cataract)
- **Retinopatia Diabética** (diabetic_retinopathy)
- **Glaucoma** (glaucoma)
- **Normal** (normal)

## 🚀 Hospedagem Rápida

### Railway (Recomendado)
1. Faça fork deste repositório
2. Conecte em [railway.app](https://railway.app)
3. Deploy automático! 🎉

### Render
1. Conecte seu repositório em [render.com](https://render.com)
2. Escolha "Web Service"
3. Deploy automático! 🎉

### Docker
```bash
docker build -t eye-disease-api .
docker run -p 8000:8000 eye-disease-api
```

📖 **Guia completo de hospedagem:** [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

## 🏗️ Arquitetura

O sistema é composto por três componentes principais:

1. **API (FastAPI)** - Serve o modelo de IA remotamente
2. **Cliente Python** - Biblioteca para consumir a API
3. **Interface Web (Streamlit)** - Interface amigável para usuários

## 🚀 Como Executar

### 1. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 2. Executar a API

```bash
# Opção 1: Usando o script
python run_api.py

# Opção 2: Diretamente com uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

A API estará disponível em:
- **Endpoint principal**: http://localhost:8000
- **Documentação automática**: http://localhost:8000/docs
- **Documentação alternativa**: http://localhost:8000/redoc

### 3. Executar a Interface Web

```bash
# Em outro terminal
python run_app.py

# Ou diretamente:
streamlit run app.py
```

A interface estará disponível em: http://localhost:8501

## 📡 Endpoints da API

### `GET /`
Informações gerais da API

### `GET /health`
Verificação de saúde da API

### `GET /classes`
Lista das classes de doenças disponíveis

### `POST /predict`
Predição de doença ocular
- **Input**: Arquivo de imagem (JPG, JPEG, PNG)
- **Output**: JSON com predição e confiança

## 🐍 Usando o Cliente Python

```python
from client.client import EyeDiseaseClassifierClient

# Criar cliente
client = EyeDiseaseClassifierClient("http://localhost:8000")

# Verificar saúde da API
health = client.health_check()
print(health)

# Fazer predição
result = client.predict_from_file("imagem.jpg")
print(f"Doença: {result['predicted_class']}")
print(f"Confiança: {result['confidence']:.2%}")
```

## 🔬 Classes de Doenças

- **normal**: Olho saudável
- **cataract**: Catarata
- **diabetic_retinopathy**: Retinopatia diabética
- **glaucoma**: Glaucoma

## 🛠️ Tecnologias Utilizadas

- **FastAPI**: Framework web moderno para APIs
- **Streamlit**: Interface web interativa
- **TensorFlow/Keras**: Framework de deep learning
- **PIL/Pillow**: Processamento de imagens
- **Requests**: Cliente HTTP
- **Uvicorn**: Servidor ASGI

## 📁 Estrutura do Projeto

```
ai-kumona-classifier/
├── api/
│   └── main.py              # API FastAPI
├── client/
│   └── client.py            # Cliente Python
├── app.py                   # Interface Streamlit
├── run_api.py              # Script para executar API
├── run_app.py              # Script para executar Streamlit
├── requirements.txt         # Dependências
└── README.md               # Este arquivo
```

## 🔧 Configuração

### Variáveis de Ambiente (Opcionais)

```bash
export API_HOST=0.0.0.0
export API_PORT=8000
export MODEL_URL="https://drive.google.com/uc?id=SEU_ID_AQUI"
```

### Personalização da URL do Modelo

Edite o arquivo `api/main.py` e altere a variável `MODEL_URL` com o ID do seu modelo no Google Drive.

## 🚨 Solução de Problemas

### Erro de Conexão com a API
1. Verifique se a API está rodando em http://localhost:8000
2. Teste o endpoint de saúde: http://localhost:8000/health

### Erro de Download do Modelo
1. Verifique se o ID do Google Drive está correto
2. Certifique-se de que o arquivo está público
3. Verifique sua conexão com a internet

### Erro de Dependências
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 📝 Exemplo de Resposta da API

```json
{
  "predicted_class": "normal",
  "confidence": 0.95,
  "all_predictions": {
    "normal": 0.95,
    "cataract": 0.03,
    "diabetic_retinopathy": 0.01,
    "glaucoma": 0.01
  },
  "status": "success"
}
```

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.
