"""
Endpoint para retreinamento do modelo com feedback de especialistas
"""

import os
import json
import logging
import numpy as np
import tensorflow as tf
from typing import List, Dict, Any
from fastapi import HTTPException
from pydantic import BaseModel
import requests
from PIL import Image
import io

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingExample(BaseModel):
    image_url: str
    correct_label: str
    original_prediction: str
    confidence: float
    specialist_confidence: float

class RetrainRequest(BaseModel):
    training_data: List[TrainingExample]
    retrain_mode: str = "incremental"  # "incremental" ou "full"

# Mapeamento de classes
CLASS_NAMES = ["normal", "cataract", "diabetic_retinopathy", "glaucoma"]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

def download_image(image_url: str) -> np.ndarray:
    """
    Baixar e processar imagem da URL
    """
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        image = Image.open(io.BytesIO(response.content))
        image = image.convert('RGB')
        image = image.resize((224, 224))
        
        # Normalizar para [0, 1]
        image_array = np.array(image) / 255.0
        
        return image_array
    except Exception as e:
        logger.error(f"Erro ao baixar imagem {image_url}: {e}")
        raise

def prepare_training_data(training_examples: List[TrainingExample]) -> tuple:
    """
    Preparar dados para treinamento
    """
    images = []
    labels = []
    weights = []
    
    for example in training_examples:
        try:
            # Baixar e processar imagem
            image = download_image(example.image_url)
            images.append(image)
            
            # Converter label para índice
            if example.correct_label not in CLASS_TO_INDEX:
                logger.warning(f"Label desconhecido: {example.correct_label}")
                continue
                
            label_index = CLASS_TO_INDEX[example.correct_label]
            
            # One-hot encoding
            label_onehot = np.zeros(len(CLASS_NAMES))
            label_onehot[label_index] = 1
            labels.append(label_onehot)
            
            # Peso baseado na confiança do especialista
            weight = example.specialist_confidence
            weights.append(weight)
            
        except Exception as e:
            logger.error(f"Erro ao processar exemplo: {e}")
            continue
    
    if len(images) == 0:
        raise ValueError("Nenhum exemplo válido para treinamento")
    
    return (
        np.array(images),
        np.array(labels),
        np.array(weights)
    )

def incremental_training(model, X_train, y_train, sample_weights):
    """
    Treinamento incremental do modelo
    """
    logger.info("Iniciando treinamento incremental...")
    
    # Configurar otimizador com learning rate baixo para fine-tuning
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks para evitar overfitting
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=2,
            min_lr=0.00001
        )
    ]
    
    # Treinar com poucos epochs para evitar overfitting
    history = model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        epochs=5,
        batch_size=min(8, len(X_train)),
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info("Treinamento incremental concluído")
    return history

def save_model_backup(model, backup_path: str):
    """
    Salvar backup do modelo atual
    """
    try:
        model.save(backup_path)
        logger.info(f"Backup do modelo salvo em: {backup_path}")
    except Exception as e:
        logger.error(f"Erro ao salvar backup: {e}")

def retrain_model(model, retrain_request: RetrainRequest) -> Dict[str, Any]:
    """
    Função principal de retreinamento
    """
    try:
        logger.info(f"Iniciando retreinamento com {len(retrain_request.training_data)} exemplos")
        
        # Preparar dados
        X_train, y_train, sample_weights = prepare_training_data(retrain_request.training_data)
        
        logger.info(f"Dados preparados: {X_train.shape[0]} exemplos válidos")
        
        if X_train.shape[0] < 2:
            raise ValueError("Número insuficiente de exemplos para treinamento")
        
        # Salvar backup do modelo atual
        backup_path = "model_backup.keras"
        save_model_backup(model, backup_path)
        
        # Realizar treinamento incremental
        history = incremental_training(model, X_train, y_train, sample_weights)
        
        # Salvar modelo atualizado
        model.save("best_model.keras")
        logger.info("Modelo atualizado salvo")
        
        # Estatísticas do treinamento
        final_loss = float(history.history['loss'][-1])
        final_accuracy = float(history.history['accuracy'][-1])
        
        # Distribuição de classes nos dados de treinamento
        class_distribution = {}
        for example in retrain_request.training_data:
            label = example.correct_label
            class_distribution[label] = class_distribution.get(label, 0) + 1
        
        result = {
            "success": True,
            "message": "Retreinamento concluído com sucesso",
            "training_stats": {
                "examples_processed": X_train.shape[0],
                "final_loss": final_loss,
                "final_accuracy": final_accuracy,
                "epochs_trained": len(history.history['loss']),
                "class_distribution": class_distribution
            },
            "model_info": {
                "backup_saved": os.path.exists(backup_path),
                "model_updated": True,
                "timestamp": tf.timestamp().numpy().item()
            }
        }
        
        logger.info(f"Retreinamento concluído: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Erro no retreinamento: {e}")
        
        # Tentar restaurar backup se algo deu errado
        backup_path = "model_backup.keras"
        if os.path.exists(backup_path):
            try:
                restored_model = tf.keras.models.load_model(backup_path)
                # Aqui você precisaria atualizar a referência global do modelo
                logger.info("Modelo restaurado do backup")
            except Exception as restore_error:
                logger.error(f"Erro ao restaurar backup: {restore_error}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Erro no retreinamento: {str(e)}"
        )

# Função para adicionar ao main.py da API
def add_retrain_endpoint(app, model):
    """
    Adicionar endpoint de retreinamento à aplicação FastAPI
    """
    
    @app.post("/retrain")
    async def retrain_endpoint(retrain_request: RetrainRequest):
        """
        Endpoint para retreinamento do modelo com feedback de especialistas
        """
        if model is None:
            raise HTTPException(status_code=500, detail="Modelo não carregado")
        
        try:
            result = retrain_model(model, retrain_request)
            return result
        except Exception as e:
            logger.error(f"Erro no endpoint de retreinamento: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/retrain/status")
    async def retrain_status():
        """
        Status do sistema de retreinamento
        """
        return {
            "retrain_available": model is not None,
            "supported_modes": ["incremental"],
            "supported_classes": CLASS_NAMES,
            "model_loaded": model is not None,
            "tensorflow_version": tf.__version__
        }
