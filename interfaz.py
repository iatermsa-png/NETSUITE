import streamlit as st
import os
import json
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    load_index_from_storage, 
    PromptTemplate
)

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Asistente Virtual de Netsuite", page_icon="💼", layout="centered")

# --- NUEVA ESTRUCTURA DE PESTAÑAS (Sin modificar tu diseño) ---
tab_login, tab_chatbot = st.tabs(["🔐 Acceso", "🤖 Asistente Virtual"])

with tab_login:
    st.subheader("Control de Acceso")
    col_a, col_b = st.columns(2)
    with col_a:
        user_input_auth = st.text_input("Usuario")
        pass_input_auth = st.text_input("Contraseña", type="password")
        if st.button("Ingresar"):
            if user_input_auth == "hola123" and pass_input_auth == "123":
                st.session_state.acceso_concedido = True
                st.success("Acceso correcto. Haz clic en la pestaña 'Asistente Virtual'.")
            else:
                st.error("Credenciales incorrectas")

with tab_chatbot:
    # Condición para que no se vea el chat si no hay login
    if "acceso_concedido" not in st.session_state or not st.session_state.acceso_concedido:
        st.warning("⚠️ Por favor, inicia sesión en la pestaña 'Acceso'.")
    else:
        # --- AQUÍ EMPIEZA TU CÓDIGO ORIGINAL TAL CUAL ---
        
        # --- BLOQUE DE ESTILOS CSS (TU DISEÑO ORIGINAL) ---
        st.markdown("""
            <style>
            /* Blanqueo de fondo general y áreas de sistema */
            .stApp, [data-testid="stHeader"], [data-testid="stBottom"], 
            [data-testid="stBottomBlockContainer"], .st-emotion-cache-18ni7ap {
                background-color: #FFFFFF !important;
                background: #FFFFFF !important;
            }

            /* Eliminación de negro en Referencias y Expanders */
            [data-testid="stExpander"], .st-emotion-cache-p5msec {
                background-color: #F8F9FB !important;
                border: 1px solid #E0E4E8 !important;
                border-radius: 10px !important;
            }
            
            [data-testid="stExpander"] details, [data-testid="stExpander"] summary {
                background-color: #F8F9FB !important;
                color: #1A1C1E !important;
            }

            /* Botón Abrir Manual (Download Button) limpio */
            .stDownloadButton>button {
                background-color: #FFFFFF !important;
                color: #00417B !important;
                border: 1px solid #00417B !important;
                border-radius: 8px !important;
                font-weight: 600 !important;
            }
            
            .stDownloadButton>button:hover {
                background-color: #00417B !important;
                color: #FFFFFF !important;
            }

            /* Chat Input: Limpieza de laterales y texto negro */
            [data-testid="stChatInput"] {
                background-color: #FFFFFF !important;
            }
            
            [data-testid="stChatInput"] > div {
                background-color: #FFFFFF !important; 
                border: 1px solid #E0E4E8 !important;
                border-radius: 12px !important;
            }

            .stChatInput textarea {
                color: #1A1C1E !important;
            }
            
            .stChatInput textarea::placeholder {
                color: #808495 !important;
            }

            /* Botón enviar en azul */
            [data-testid="stChatInput"] button {
                background-color: #00417B !important;
                border-radius: 50% !important;
            }
            
            [data-testid="stChatInput"] button svg {
                fill: #FFFFFF !important;
            }

            /* Forzar textos a negro para visibilidad */
            h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown, [data-testid="stWidgetLabel"] {
                color: #1A1C1E !important;
            }

            /* Sidebar y Botones de ejemplo */
            [data-testid="stSidebar"] { background-color: #F8F9FB !important; }
            
            /* Estilo para botones de ejemplo en el área principal */
            .stButton>button {
                background-color: #FFFFFF !important;
                color: #1A1C1E !important;
                border: 1px solid #E0E4E8 !important;
                border-radius: 8px !important;
                font-weight: 500 !important;
                transition: all 0.2s ease-in-out;
            }

            .stButton>button:hover {
                background-color: #F8F9FB !important;
                color: #00417B !important;
                border-color: #00417B !important;
                transform: translateY(-2px);
            }

            /* Anulación para botones dentro de la barra lateral (Sidebar) */
            [data-testid="stSidebar"] .stButton>button {
                background-color: #F0F2F6 !important;
                color: #1A1C1E !important;
                border: 1px solid #D0D4D9 !important;
                transform: none !important; /* Resetea la animación de hover */
            }

            [data-testid="stSidebar"] .stButton>button:hover {
                background-color: #E0E4E8 !important;
                color: #1A1C1E !important;
                border-color: #C0C4C9 !important;
            }

            footer {display: none !important;}
            </style>
            """, unsafe_allow_html=True)

        # --- 2. TU LÓGICA DE CARGA ORIGINAL ---
        @st.cache_resource(show_spinner="Cargando base de conocimiento... 🧠")
        def cargar_indice():
            storage_path = "./storage"
            data_path = "./datos"
            
            if not os.path.exists(data_path) or not os.listdir(data_path):
                return None, f"⚠️ La carpeta '{data_path}' está vacía o no existe."

            try:
                if os.path.exists(storage_path) and os.listdir(storage_path):
                    storage_context = StorageContext.from_defaults(persist_dir=storage_path)
                    index = load_index_from_storage(storage_context)
                    return index, "✅ Índice cargado con éxito."

                docs = SimpleDirectoryReader(data_path).load_data()
                index = VectorStoreIndex.from_documents(docs)
                index.storage_context.persist(persist_dir=storage_path)
                return index, f"🆕 Índice creado con {len(docs)} documentos."
            
            except Exception as e:
                return None, f"❌ Error: {e}"

        # Inicialización
        load_dotenv()
        index, status_msg = cargar_indice()

        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "pre_filled_prompt" not in st.session_state:
            st.session_state.pre_filled_prompt = None
        if "send_prompt" not in st.session_state:
            st.session_state.send_prompt = False

        # --- 3. TU BARRA LATERAL ORIGINAL ---
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
                st.rerun()
            
            st.markdown("---")
            with st.expander("ℹ️ Acerca de"):
                st.info("Asistente técnico basado en manuales de NetSuite.")

        # --- 4. DISEÑO PRINCIPAL (MANTENIDO) ---
        col1, col2 = st.columns([1, 3])
        with col1:
            logo_path = "imagenes/logo.jpg" 
            if os.path.exists(logo_path):
                st.image(logo_path, width=150)

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

        # --- 5. MOTOR DE CONSULTAS Y FUENTES (TU LÓGICA INTACTA) ---
        if index:
            query_engine = index.as_query_engine()
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
                            st.download_button(
                                label=f"📖 Abrir manual: {f['nombre']}",
                                data=file_data, file_name=f['nombre'],
                                mime="application/pdf", key=f"btn_{id_mensaje}_{i}"
                            )
                    st.markdown(f"📝 **Texto original:** _{f['texto']}_")
                    st.divider()

        for idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "fuentes" in msg:
                    mostrar_fuentes_persistentes(msg["fuentes"], idx)

        # --- 6. PROCESAMIENTO DE INPUT ---
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
                with st.spinner("Consultando información..."):
                    respuesta = query_engine.query(prompt_to_process)
                    st.markdown(respuesta.response)
                    
                    fuentes_encontradas = []
                    for nodo in respuesta.source_nodes:
                        fuentes_encontradas.append({
                            "nombre": nodo.metadata.get('file_name', 'Desconocido'),
                            "ruta": nodo.metadata.get('file_path', f"./datos/{nodo.metadata.get('file_name')}"),
                            "pagina": nodo.metadata.get('page_label', 'N/A'),
                            "texto": nodo.get_text()[:200]
                        })
                    
                    mostrar_fuentes_persistentes(fuentes_encontradas, len(st.session_state.messages))
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": respuesta.response,
                        "fuentes": fuentes_encontradas
                    })