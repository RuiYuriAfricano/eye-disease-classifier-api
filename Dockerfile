# Use uma imagem base do Python
FROM python:3.9-slim

# Definir diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar arquivos de requisitos
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY . .

# Configurar variáveis de ambiente para CPU-only
ENV CUDA_VISIBLE_DEVICES=-1
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TF_FORCE_GPU_ALLOW_GROWTH=false
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV TF_DISABLE_MKL=1
ENV TF_NUM_INTEROP_THREADS=1
ENV TF_NUM_INTRAOP_THREADS=1
ENV OMP_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1

# Expor porta
EXPOSE 8000

# Comando para executar a aplicação (usando versão simples)
CMD ["uvicorn", "api.main_simple:app", "--host", "0.0.0.0", "--port", "8000"]
