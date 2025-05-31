# Guia de Hospedagem - Eye Disease Classifier API

Este guia mostra como hospedar sua API de classificação de doenças oculares em diferentes plataformas.

## 🚀 Opções de Hospedagem

### 1. Railway (Recomendado - Fácil e Gratuito)

Railway é uma plataforma moderna que facilita o deploy de aplicações.

**Passos:**
1. Crie uma conta em [railway.app](https://railway.app)
2. Conecte seu repositório GitHub
3. O Railway detectará automaticamente o `railway.toml` e fará o deploy
4. Sua API estará disponível em uma URL como: `https://seu-projeto.railway.app`

**Vantagens:**
- Deploy automático
- SSL gratuito
- Fácil configuração
- Plano gratuito generoso

### 2. Render

Render é outra excelente opção para hospedagem gratuita.

**Passos:**
1. Crie uma conta em [render.com](https://render.com)
2. Conecte seu repositório GitHub
3. Escolha "Web Service"
4. O Render detectará o `render.yaml` automaticamente
5. Sua API estará disponível em: `https://seu-app.onrender.com`

**Vantagens:**
- SSL gratuito
- Deploy automático
- Boa documentação

### 3. Heroku

**Passos:**
1. Instale o Heroku CLI
2. Crie um app: `heroku create seu-app-name`
3. Faça deploy: `git push heroku main`
4. Configure variáveis se necessário

### 4. Docker (Para qualquer provedor)

Use o `Dockerfile` incluído para fazer deploy em qualquer provedor que suporte Docker:

```bash
# Build da imagem
docker build -t eye-disease-api .

# Executar localmente
docker run -p 8000:8000 eye-disease-api

# Ou usar docker-compose
docker-compose up
```

## 🔧 Configuração

### Variáveis de Ambiente

Você pode configurar estas variáveis de ambiente:

- `PORT`: Porta da aplicação (padrão: 8000)
- `PYTHONUNBUFFERED`: Para logs em tempo real (recomendado: 1)

### Recursos Necessários

- **RAM**: Mínimo 1GB (recomendado 2GB para o modelo)
- **Armazenamento**: ~500MB (para o modelo baixado)
- **CPU**: 1 vCPU é suficiente

## 📱 Consumindo a API

### Endpoints Disponíveis

- `GET /`: Informações da API
- `GET /health`: Status de saúde
- `GET /docs`: Documentação automática (Swagger)
- `POST /predict`: Classificar imagem
- `GET /classes`: Classes disponíveis

### Exemplo de Uso com JavaScript

```javascript
const formData = new FormData();
formData.append('file', imageFile);

const response = await fetch('https://sua-api.railway.app/predict', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(result);
```

### Exemplo com Python

```python
import requests

url = "https://sua-api.railway.app/predict"
files = {"file": open("imagem.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## 🔒 Segurança

Para produção, considere:

1. **CORS**: Configurar domínios específicos em vez de "*"
2. **Rate Limiting**: Implementar limite de requisições
3. **Autenticação**: Adicionar API keys se necessário
4. **HTTPS**: Sempre usar HTTPS (incluído nas plataformas recomendadas)

## 📊 Monitoramento

- Use `/health` para health checks
- Monitore logs da aplicação
- Configure alertas para falhas

## 🐛 Troubleshooting

### Problema: Modelo não carrega
- **Solução**: A API funcionará em modo demo até o modelo ser baixado

### Problema: Timeout no deploy
- **Solução**: Aumente o timeout de health check para 300s

### Problema: Falta de memória
- **Solução**: Upgrade para um plano com mais RAM

## 📞 Suporte

Se tiver problemas:
1. Verifique os logs da aplicação
2. Teste o endpoint `/health`
3. Verifique a documentação em `/docs`
