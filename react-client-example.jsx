import React, { useState } from 'react';
import axios from 'axios';

const EyeDiseaseClassifier = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // URL da sua API hospedada - substitua pela URL real
  const API_URL = 'https://sua-api-hospedada.com'; // ou 'http://localhost:8000' para desenvolvimento

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPrediction(null);
      setError(null);
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      setError('Por favor, selecione uma imagem');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API_URL}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setPrediction(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Erro ao fazer predição');
    } finally {
      setLoading(false);
    }
  };

  const formatConfidence = (confidence) => {
    return (confidence * 100).toFixed(2);
  };

  return (
    <div style={{ maxWidth: '600px', margin: '0 auto', padding: '20px' }}>
      <h1>Classificador de Doenças Oculares</h1>
      
      <div style={{ marginBottom: '20px' }}>
        <input
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          style={{ marginBottom: '10px' }}
        />
        
        {selectedFile && (
          <div>
            <p>Arquivo selecionado: {selectedFile.name}</p>
            <img
              src={URL.createObjectURL(selectedFile)}
              alt="Preview"
              style={{ maxWidth: '300px', maxHeight: '300px', objectFit: 'contain' }}
            />
          </div>
        )}
      </div>

      <button
        onClick={handlePredict}
        disabled={!selectedFile || loading}
        style={{
          padding: '10px 20px',
          backgroundColor: loading ? '#ccc' : '#007bff',
          color: 'white',
          border: 'none',
          borderRadius: '5px',
          cursor: loading ? 'not-allowed' : 'pointer'
        }}
      >
        {loading ? 'Analisando...' : 'Classificar Imagem'}
      </button>

      {error && (
        <div style={{ color: 'red', marginTop: '20px' }}>
          <strong>Erro:</strong> {error}
        </div>
      )}

      {prediction && (
        <div style={{ marginTop: '20px', padding: '20px', border: '1px solid #ddd', borderRadius: '5px' }}>
          <h3>Resultado da Classificação</h3>
          
          {prediction.status === 'demo_mode' && (
            <div style={{ color: 'orange', marginBottom: '10px' }}>
              <strong>⚠️ Modo Demo:</strong> {prediction.message}
            </div>
          )}
          
          <div style={{ marginBottom: '15px' }}>
            <strong>Classe Predita:</strong> {prediction.predicted_class}
          </div>
          
          <div style={{ marginBottom: '15px' }}>
            <strong>Confiança:</strong> {formatConfidence(prediction.confidence)}%
          </div>
          
          <div>
            <strong>Todas as Probabilidades:</strong>
            <ul style={{ marginTop: '10px' }}>
              {Object.entries(prediction.all_predictions).map(([className, prob]) => (
                <li key={className}>
                  {className}: {formatConfidence(prob)}%
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};

export default EyeDiseaseClassifier;
