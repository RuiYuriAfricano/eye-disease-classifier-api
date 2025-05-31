import streamlit as st
from PIL import Image
import requests
import io
import json
from client.client import EyeDiseaseClassifierClient

# Configuração da página
st.set_page_config(
    page_title="Eye Disease Classifier",
    page_icon="👁️",
    layout="wide"
)

# Configuração da API
API_URL = st.sidebar.text_input("URL da API", value="http://localhost:8000")

# Inicializar cliente
@st.cache_resource
def get_client(api_url):
    return EyeDiseaseClassifierClient(api_url)

def main():
    st.title("👁️ Eye Disease Classifier")
    st.write("Faça upload de uma imagem do olho para detectar doenças usando nossa API.")
    
    # Obter cliente
    client = get_client(API_URL)
    
    # Verificar status da API
    with st.sidebar:
        st.subheader("🔧 Status da API")
        if st.button("Verificar Conexão"):
            with st.spinner("Verificando conexão..."):
                health = client.health_check()
                if health.get("status") == "healthy":
                    st.success("✅ API conectada!")
                    st.json(health)
                else:
                    st.error("❌ Erro na conexão com a API")
                    st.json(health)
        
        # Mostrar classes disponíveis
        st.subheader("📋 Classes Disponíveis")
        try:
            classes_response = client.get_classes()
            if "classes" in classes_response:
                for i, class_name in enumerate(classes_response["classes"], 1):
                    st.write(f"{i}. {class_name}")
            else:
                st.error("Erro ao obter classes")
        except Exception as e:
            st.error(f"Erro: {e}")
    
    # Upload de arquivo
    uploaded_file = st.file_uploader(
        "Escolha uma imagem...", 
        type=["jpg", "jpeg", "png"],
        help="Formatos suportados: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        # Mostrar imagem
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Imagem Carregada")
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagem carregada", use_container_width=True)
            
            # Informações da imagem
            st.write(f"**Formato:** {image.format}")
            st.write(f"**Tamanho:** {image.size}")
            st.write(f"**Modo:** {image.mode}")
        
        with col2:
            st.subheader("🔮 Resultado da Predição")
            
            if st.button("🚀 Analisar Imagem", type="primary"):
                with st.spinner("Analisando imagem..."):
                    try:
                        # Fazer predição usando o cliente
                        result = client.predict_from_image(image, uploaded_file.name)
                        
                        if "error" in result:
                            st.error(f"❌ Erro na predição: {result['error']}")
                        else:
                            # Mostrar resultado principal
                            st.success("✅ Análise concluída!")
                            
                            # Resultado principal
                            predicted_class = result.get("predicted_class", "N/A")
                            confidence = result.get("confidence", 0)
                            
                            st.metric(
                                label="Doença Detectada",
                                value=predicted_class.replace("_", " ").title(),
                                delta=f"{confidence:.2%} de confiança"
                            )
                            
                            # Barra de progresso para confiança
                            st.progress(confidence)
                            
                            # Mostrar todas as probabilidades
                            if "all_predictions" in result:
                                st.subheader("📊 Todas as Probabilidades")
                                all_preds = result["all_predictions"]
                                
                                # Ordenar por probabilidade
                                sorted_preds = sorted(
                                    all_preds.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True
                                )
                                
                                for class_name, prob in sorted_preds:
                                    display_name = class_name.replace("_", " ").title()
                                    st.write(f"**{display_name}:** {prob:.2%}")
                                    st.progress(prob)
                            
                            # Mostrar resposta completa em JSON (expansível)
                            with st.expander("🔍 Ver Resposta Completa da API"):
                                st.json(result)
                    
                    except Exception as e:
                        st.error(f"❌ Erro inesperado: {str(e)}")
    
    # Informações adicionais
    with st.expander("ℹ️ Sobre o Sistema"):
        st.write("""
        ### Como funciona:
        1. **Upload**: Carregue uma imagem do olho
        2. **API**: A imagem é enviada para nossa API de classificação
        3. **IA**: O modelo de deep learning analisa a imagem
        4. **Resultado**: Receba a classificação e nível de confiança
        
        ### Classes de Doenças:
        - **Normal**: Olho saudável
        - **Cataract**: Catarata
        - **Diabetic Retinopathy**: Retinopatia diabética
        - **Glaucoma**: Glaucoma
        
        ### Tecnologias Utilizadas:
        - **Frontend**: Streamlit
        - **API**: FastAPI
        - **IA**: TensorFlow/Keras com InceptionV3
        - **Cliente**: Requests HTTP
        """)

if __name__ == "__main__":
    main()
