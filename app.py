import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- CONFIGURACIÓN INICIAL ---
# Descargamos recursos de NLTK silenciosamente por si faltan
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

# --- CARGAR MODELOS ---
# Usamos @st.cache_resource para que no recargue el modelo cada vez que alguien escribe algo
# Esto hace que la web sea MUY rápida.
@st.cache_resource
def cargar_modelos():
    modelo = joblib.load('C:/Users/Raúl Perú/mis_modelos/modelo_sentimiento.pkl')
    vectorizador = joblib.load('C:/Users/Raúl Perú/mis_modelos/vectorizador_tfidf.pkl')
    return modelo, vectorizador

modelo, tfidf = cargar_modelos()

# --- FUNCIÓN DE LIMPIEZA (La misma que usaste antes) ---
def limpiar_reseña(texto):
    texto = texto.lower()
    texto = re.sub(r'<.*?>', '', texto)
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)
    palabras = word_tokenize(texto)
    stop_words = set(stopwords.words('english'))
    palabras_filtradas = [w for w in palabras if w not in stop_words]
    return " ".join(palabras_filtradas)

# --- INTERFAZ GRÁFICA (FRONTEND) ---
st.title("🎬 Analizador de Sentimientos de Cine")
st.write("Escribe una reseña en inglés y la IA te dirá si es Positiva o Negativa.")

# Caja de texto para el usuario
texto_usuario = st.text_area("Escribe tu reseña aquí:", height=150)

# Botón de predicción
if st.button("Analizar Sentimiento"):
    if texto_usuario:
        # 1. Limpieza
        texto_limpio = limpiar_reseña(texto_usuario)
        
        # 2. Vectorización
        texto_vec = tfidf.transform([texto_limpio]).toarray()
        
        # 3. Predicción
        prediccion = modelo.predict(texto_vec)[0]
        probabilidad = modelo.predict_proba(texto_vec).max()
        
        # 4. Mostrar resultados con estilo
        if prediccion == 1:
            st.success(f"### ¡Es una reseña POSITIVA! 👍")
            st.write(f"Confianza del modelo: **{probabilidad:.2%}**")
        else:
            st.error(f"### Es una reseña NEGATIVA 👎")
            st.write(f"Confianza del modelo: **{probabilidad:.2%}**")
    else:
        st.warning("Por favor, escribe algo antes de analizar.")

st.markdown("---")
st.caption("Modelo entrenado con Regresión Logística y TF-IDF")