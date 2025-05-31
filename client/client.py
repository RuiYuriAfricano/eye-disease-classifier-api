import requests
import json
from typing import Dict, Optional
from PIL import Image
import io
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EyeDiseaseClassifierClient:
    """Cliente para consumir a API de classificação de doenças oculares"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        """
        Inicializa o cliente
        
        Args:
            api_url: URL base da API
        """
        self.api_url = api_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict:
        """
        Verifica se a API está funcionando
        
        Returns:
            Dict com status da API
        """
        try:
            response = self.session.get(f"{self.api_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Erro ao verificar saúde da API: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_classes(self) -> Dict:
        """
        Obtém as classes disponíveis
        
        Returns:
            Dict com lista de classes
        """
        try:
            response = self.session.get(f"{self.api_url}/classes")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Erro ao obter classes: {e}")
            return {"error": str(e)}
    
    def predict_from_file(self, image_path: str) -> Dict:
        """
        Faz predição a partir de um arquivo de imagem
        
        Args:
            image_path: Caminho para o arquivo de imagem
            
        Returns:
            Dict com resultado da predição
        """
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (image_path, f, 'image/jpeg')}
                response = self.session.post(f"{self.api_url}/predict", files=files)
                response.raise_for_status()
                return response.json()
        except FileNotFoundError:
            return {"error": f"Arquivo não encontrado: {image_path}"}
        except requests.exceptions.RequestException as e:
            logger.error(f"Erro na predição: {e}")
            return {"error": str(e)}
    
    def predict_from_image(self, image: Image.Image, filename: str = "image.jpg") -> Dict:
        """
        Faz predição a partir de um objeto PIL Image
        
        Args:
            image: Objeto PIL Image
            filename: Nome do arquivo (para o cabeçalho HTTP)
            
        Returns:
            Dict com resultado da predição
        """
        try:
            # Converter imagem para bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            
            files = {'file': (filename, img_byte_arr, 'image/jpeg')}
            response = self.session.post(f"{self.api_url}/predict", files=files)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Erro na predição: {e}")
            return {"error": str(e)}
    
    def predict_from_bytes(self, image_bytes: bytes, filename: str = "image.jpg") -> Dict:
        """
        Faz predição a partir de bytes de imagem
        
        Args:
            image_bytes: Bytes da imagem
            filename: Nome do arquivo (para o cabeçalho HTTP)
            
        Returns:
            Dict com resultado da predição
        """
        try:
            files = {'file': (filename, io.BytesIO(image_bytes), 'image/jpeg')}
            response = self.session.post(f"{self.api_url}/predict", files=files)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Erro na predição: {e}")
            return {"error": str(e)}

def main():
    """Exemplo de uso do cliente"""
    # Criar cliente
    client = EyeDiseaseClassifierClient()
    
    # Verificar saúde da API
    print("🔍 Verificando saúde da API...")
    health = client.health_check()
    print(f"Status da API: {health}")
    
    # Obter classes disponíveis
    print("\n📋 Obtendo classes disponíveis...")
    classes = client.get_classes()
    print(f"Classes: {classes}")
    
    # Exemplo de predição (substitua pelo caminho real da imagem)
    # image_path = "exemplo.jpg"
    # if os.path.exists(image_path):
    #     print(f"\n🔮 Fazendo predição para {image_path}...")
    #     result = client.predict_from_file(image_path)
    #     print(f"Resultado: {json.dumps(result, indent=2)}")

if __name__ == "__main__":
    main()
