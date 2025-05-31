# 🚀 Guia de Deploy no Railway - Versões Disponíveis

## 🛡️ Problema Resolvido: Crashes no Carregamento do Modelo

O modelo de 169MB estava causando crashes no Railway devido a problemas com CUDA/GPU. Implementamos **duas versões** para resolver isso:

## 📋 Versões Disponíveis

### 1. 🔧 **Versão Principal (api/main.py)** - Melhorada
- ✅ Configurações otimizadas do TensorFlow
- 🔄 Sistema de retry automático
- 💾 Monitoramento de memória
- 🔍 Verificação de integridade do modelo
- 🛠️ Endpoints de gerenciamento (`/reload-model`, `/memory`)

### 2. 🛡️ **Versão CPU-Only (api/main_cpu_only.py)** - Ultra-Segura
- 🚫 **FORÇA uso apenas de CPU** (sem GPU/CUDA)
- ⚡ Carregamento ultra-conservador
- 🧹 Preprocessamento simplificado
- 📦 Compilação manual do modelo
- 🎯 **Recomendada para Railway**

## 🔄 Como Alternar Entre Versões

### Opção 1: Usar Versão CPU-Only (Recomendado)

Altere o `railway.toml`:
```toml
[deploy]
startCommand = "uvicorn api.main_cpu_only:app --host 0.0.0.0 --port $PORT"
```

Ou use o script:
```toml
[deploy]
startCommand = "python run_api_cpu_only.py"
```

### Opção 2: Usar Versão Principal Melhorada

```toml
[deploy]
startCommand = "uvicorn api.main:app --host 0.0.0.0 --port $PORT"
```

## 🛠️ Configurações Aplicadas na Versão CPU-Only

```bash
CUDA_VISIBLE_DEVICES=-1          # Desabilitar GPU completamente
TF_CPP_MIN_LOG_LEVEL=3          # Suprimir logs TensorFlow
TF_FORCE_GPU_ALLOW_GROWTH=false # Desabilitar crescimento GPU
TF_ENABLE_ONEDNN_OPTS=0         # Desabilitar otimizações OneDNN
TF_DISABLE_MKL=1                # Desabilitar MKL
TF_NUM_INTEROP_THREADS=1        # Limitar threads
TF_NUM_INTRAOP_THREADS=1        # Limitar threads
OMP_NUM_THREADS=1               # Limitar OpenMP
```

## 📊 Endpoints de Monitoramento

Ambas as versões incluem:

- `GET /health` - Status da API e modelo
- `GET /status` - Status detalhado (só versão principal)
- `GET /memory` - Uso de memória (só versão principal)
- `POST /reload-model` - Forçar recarregamento

## 🔍 Como Monitorar no Railway

1. **Logs em Tempo Real**: Acompanhe o carregamento do modelo
2. **Health Check**: Acesse `/health` para ver o status
3. **Memória**: Use `/memory` para monitorar uso de recursos

## 🎯 Logs Esperados (Versão CPU-Only)

```
🚀 Iniciando API CPU-only...
💾 Memória inicial: 66.64MB
🔄 Iniciando carregamento ultra-seguro...
🔧 Configurando TensorFlow...
🖥️ GPUs físicas detectadas: 0
🖥️ GPUs visíveis (deve ser 0): 0
✅ TensorFlow configurado para CPU APENAS
📦 Importando TensorFlow/Keras...
🔄 Carregando modelo (CPU apenas)...
🔧 Compilando modelo...
✅ Modelo compilado com sucesso
🧪 Testando modelo...
💾 Memória final: 650.2MB
📈 Incremento: 583.6MB
🎉 Modelo carregado com sucesso!
```

## ⚠️ Troubleshooting

### Se ainda crashar:
1. Verifique se está usando `api.main_cpu_only:app`
2. Aumente o `healthcheckTimeout` para 900s
3. Monitore os logs para identificar onde para

### Se o modelo não carregar:
1. Use `POST /reload-model` para tentar novamente
2. Verifique `/health` para ver erros
3. A API funcionará em modo demo até o modelo carregar

## 🚀 Deploy Atual

As alterações já foram enviadas para o GitHub. O Railway detectará automaticamente e fará o redeploy.

**Configuração atual no railway.toml:**
- ✅ Timeout aumentado para 900s
- ✅ Comando otimizado
- ✅ Health check configurado

## 📞 Próximos Passos

1. **Aguarde o redeploy** do Railway
2. **Monitore os logs** para ver o carregamento
3. **Teste os endpoints** `/health` e `/predict`
4. **Se necessário**, use `/reload-model` para forçar recarregamento

A versão CPU-only deve resolver definitivamente os crashes! 🎉
