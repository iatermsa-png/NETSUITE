import streamlit as st
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    load_index_from_storage, 
    PromptTemplate
)
from llama_index.core.memory import ChatMemoryBuffer # <--- MEJORA 1: MEMORIA

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Asistente Virtual de Netsuite", page_icon="💼", layout="centered")

# --- BLOQUE DE ACCESO (MANTENIDO IGUAL) ---
if "acceso_concedido" not in st.session_state:
    st.session_state.acceso_concedido = False

if not st.session_state.acceso_concedido:
    st.title("🔐 Acceso al Asistente")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        user_auth = st.text_input("Usuario")
        pass_auth = st.text_input("Contraseña", type="password")
        if st.button("Ingresar", use_container_width=True):
            if user_auth == "hola123" and pass_auth == "123":
                st.session_state.acceso_concedido = True
                st.rerun() 
            else:
                st.error("Credenciales incorrectas")
    st.stop()

# --- BLOQUE DE ESTILOS CSS (TU DISEÑO ORIGINAL SIN CAMBIOS) ---
st.markdown("""
    <style>
    .stApp, [data-testid="stHeader"], [data-testid="stBottom"], 
    [data-testid="stBottomBlockContainer"], .st-emotion-cache-18ni7ap {
        background-color: #FFFFFF !important;
    }
    [data-testid="stExpander"], .st-emotion-cache-p5msec {
        background-color: #F8F9FB !important;
        border: 1px solid #E0E4E8 !important;
        border-radius: 10px !important;
    }
    .stDownloadButton>button {
        background-color: #FFFFFF !important;
        color: #00417B !important;
        border: 1px solid #00417B !important;
        border-radius: 8px !important;
    }
    [data-testid="stChatInput"] { background-color: #FFFFFF !important; }
    h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown { color: #1A1C1E !important; }
    [data-testid="stSidebar"] { background-color: #F8F9FB !important; }
    footer {display: none !important;}
    </style>
    """, unsafe_allow_html=True)

# --- 2. LÓGICA DE CARGA ORIGINAL ---
@st.cache_resource(show_spinner="Cargando base de conocimiento... 🧠")
def cargar_indice():
    storage_path = "./storage"
    data_path = "./datos"
    metadata_file = os.path.join(storage_path, "metadata.json")

    if not os.path.exists(data_path) or not os.listdir(data_path):
        return None, f"⚠️ La carpeta '{data_path}' está vacía o no existe."

    def _es_puntero_lfs(ruta):
        # Si el índice se sirviera por Git LFS en un entorno que no lo resuelve
        # (p. ej. el build de Railway), los .json llegarían como punteros de
        # texto ("version https://git-lfs...") en vez de JSON real.
        try:
            with open(ruta, "rb") as f:
                return f.read(64).startswith(b"version https://git-lfs")
        except OSError:
            return False

    def _storage_utilizable():
        requeridos = ["docstore.json", "index_store.json", "default__vector_store.json"]
        for nombre in requeridos:
            ruta = os.path.join(storage_path, nombre)
            if not (os.path.exists(ruta) and os.path.getsize(ruta) > 0):
                return False
            if _es_puntero_lfs(ruta):
                return False
        return True

    try:
        if _storage_utilizable():
            try:
                storage_context = StorageContext.from_defaults(persist_dir=storage_path)
                index = load_index_from_storage(storage_context)

                num_files = 0
                if os.path.exists(metadata_file):
                    with open(metadata_file, "r") as f:
                        try:
                            num_files = json.load(f).get("num_files", 0)
                        except json.JSONDecodeError:
                            num_files = 0

                if num_files > 0:
                    return index, f"✅ Índice cargado con éxito ({num_files} archivos)."
                return index, "✅ Índice cargado con éxito."
            except Exception:
                # El índice existe pero no se pudo leer; se reconstruye desde ./datos.
                pass

        # No hay índice utilizable: se construye desde los PDF y se persiste en
        # ./storage (montado como Volumen de Railway para sobrevivir redeploys).
        docs = SimpleDirectoryReader(data_path).load_data()

        num_files = 0
        if docs:
            file_names = {doc.metadata.get('file_name') for doc in docs if doc.metadata and 'file_name' in doc.metadata}
            num_files = len(file_names)

        index = VectorStoreIndex.from_documents(docs)

        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
        index.storage_context.persist(persist_dir=storage_path)
        with open(metadata_file, "w") as f:
            json.dump({"num_files": num_files}, f)

        return index, f"🆕 Índice creado con {num_files} archivos."
    except Exception as e:
        return None, f"❌ Error: {e}"

load_dotenv()
index, status_msg = cargar_indice()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pre_filled_prompt" not in st.session_state:
    st.session_state.pre_filled_prompt = None
if "send_prompt" not in st.session_state:
    st.session_state.send_prompt = False

# --- 3. BARRA LATERAL ORIGINAL ---
with st.sidebar:
    st.title("🤖 Panel de Control")
    st.markdown("---")
    st.subheader("⚙️ Estado del Sistema")
    if "❌" in status_msg or "⚠️" in status_msg:
        st.error(status_msg)
    else:
        st.success(status_msg)
    st.markdown("---")
    if st.button("🗑️ Limpiar Chat", use_container_width=True):
        st.session_state.messages = []
        if "chat_engine" in st.session_state:
            st.session_state.chat_engine.reset() # Reset de memoria
        st.rerun()
    if st.button("🚪 Cerrar Sesión", use_container_width=True):
        st.session_state.acceso_concedido = False
        st.rerun()

# --- 4. DISEÑO PRINCIPAL ---
col1, col2 = st.columns([1, 3])
with col1:
    if os.path.exists("imagenes/logo.jpg"): st.image("imagenes/logo.jpg", width=150)
with col2:
    st.title("💼 Asistente Virtual de Netsuite")

if not st.session_state.messages:
    st.markdown("---")
    st.subheader("Ejemplos de Preguntas:")
    example_questions = [
        "Dime paso a paso cómo puedo crear un nuevo registro de cliente en NetSuite.",
        "Dime paso a paso cuál es el proceso para generar un informe de ventas en NetSuite.",
        "Dime paso a paso cómo se maneja la contabilidad general en NetSuite.",
        "Dime paso a paso cómo puedo personalizar un dashboard en NetSuite."
    ]
    cols = st.columns(2)
    for i, question in enumerate(example_questions):
        with cols[i % 2]:
            if st.button(question, key=f"q_button_{i}", use_container_width=True):
                st.session_state.pre_filled_prompt = question
                st.session_state.send_prompt = True

# --- 5. MOTOR DE CHAT (MEJORA: REEMPLAZA QUERY_ENGINE POR CHAT_ENGINE) ---
if index:
    if "chat_engine" not in st.session_state:
        # Buffer de memoria para recordar el contexto de la charla
        memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
        st.session_state.chat_engine = index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            system_prompt="Eres un experto en NetSuite. Responde de forma profesional, clara y estructurada. Usa negritas para puntos clave."
        )
else:
    st.stop()

def mostrar_fuentes_persistentes(fuentes_list, id_mensaje):
    with st.expander("🔍 Ver referencia exacta (Archivo y Página)"):
        for i, f in enumerate(fuentes_list):
            st.markdown(f"**Referencia #{i+1}:**")
            st.markdown(f"📄 **Archivo:** `{f['nombre']}`")
            st.markdown(f"📑 **Página:** `{f['pagina']}`")
            if os.path.exists(f['ruta']):
                with open(f['ruta'], "rb") as file_data:
                    st.download_button(label=f"📖 Abrir manual: {f['nombre']}", data=file_data, file_name=f['nombre'], mime="application/pdf", key=f"btn_{id_mensaje}_{i}")
            st.markdown(f"📝 **Texto original:** _{f['texto']}_")
            st.divider()

# Mostrar historial de mensajes
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "fuentes" in msg:
            mostrar_fuentes_persistentes(msg["fuentes"], idx)

# --- 6. PROCESAMIENTO DE INPUT (MEJORA: STREAMING) ---
user_input = st.chat_input("Escribe tu duda sobre NetSuite aquí...")
triggered_prompt = None

if st.session_state.send_prompt and st.session_state.pre_filled_prompt:
    triggered_prompt = st.session_state.pre_filled_prompt
    st.session_state.pre_filled_prompt = None
    st.session_state.send_prompt = False
    
prompt_to_process = user_input or triggered_prompt

if prompt_to_process:
    st.session_state.messages.append({"role": "user", "content": prompt_to_process})
    with st.chat_message("user"):
        st.markdown(prompt_to_process)

    with st.chat_message("assistant"):
        # Contenedor vacío para el efecto de escritura palabra por palabra
        response_placeholder = st.empty()
        full_response = ""
        
        # Llamada con Streaming
        response = st.session_state.chat_engine.stream_chat(prompt_to_process)
        
        for token in response.response_gen:
            full_response += token
            response_placeholder.markdown(full_response + "▌") # Cursor de escritura
        
        response_placeholder.markdown(full_response)
        
        # Procesar fuentes del nodo de respuesta
        fuentes_encontradas = []
        for nodo in response.source_nodes:
            fuentes_encontradas.append({
                "nombre": nodo.metadata.get('file_name', 'Desconocido'),
                "ruta": nodo.metadata.get('file_path', f"./datos/{nodo.metadata.get('file_name')}"),
                "pagina": nodo.metadata.get('page_label', 'N/A'),
                "texto": nodo.get_text()[:200]
            })
        
        mostrar_fuentes_persistentes(fuentes_encontradas, len(st.session_state.messages))
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_response,
            "fuentes": fuentes_encontradas
        })