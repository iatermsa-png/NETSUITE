import streamlit as st
import os
import re
import json
from datetime import datetime
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    PromptTemplate,
    Settings,
    Document,
)
from pypdf import PdfReader  # loader por página (preserva page_label)
from llama_index.core.memory import ChatMemoryBuffer # <--- MEJORA 1: MEMORIA
from llama_index.core.llms import ChatMessage, MessageRole  # para inyectar memoria en cache hits
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # embeddings locales (gratis)
from llama_index.llms.openai import OpenAI

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
def _cargar_documentos(data_path):
    """Carga los PDF como un Document por página, preservando page_label.

    El lector por defecto concatenaba cada PDF en un único Document sin número
    de página, lo que rompía las referencias "archivo y página" y generaba un
    blob enorme. pypdf nos da las páginas explícitamente.
    """
    documentos = []
    pdfs = [f for f in sorted(os.listdir(data_path)) if f.lower().endswith(".pdf")]
    for idx, fname in enumerate(pdfs, 1):
        fpath = os.path.join(data_path, fname)
        try:
            lector = PdfReader(fpath)
        except Exception:
            continue
        for i, page in enumerate(lector.pages):
            texto = page.extract_text() or ""
            if not texto.strip():
                continue
            documentos.append(Document(
                text=texto,
                metadata={
                    "file_name": fname,
                    "file_path": fpath,
                    "page_label": str(i + 1),
                },
            ))
        if idx % 40 == 0:
            print(f"[loader] {idx}/{len(pdfs)} PDFs procesados", flush=True)
    return documentos


def _config_postgres():
    """Parámetros de conexión a Postgres si Railway los inyectó, o None.

    Railway expone DATABASE_URL (y/o PGHOST/PGPORT/...) al enlazar un plugin
    Postgres. Sin estas variables (p. ej. en local) devuelve None y la app usa
    el índice JSON de ./storage de siempre.
    """
    if os.environ.get("DATABASE_URL"):
        return {"url": os.environ["DATABASE_URL"]}
    if os.environ.get("PGHOST"):
        return {
            "host": os.environ["PGHOST"],
            "port": int(os.environ.get("PGPORT", "5432")),
            "user": os.environ.get("PGUSER", "postgres"),
            "password": os.environ.get("PGPASSWORD", ""),
            "database": os.environ.get("PGDATABASE", "railway"),
        }
    return None


def _params_pg(pg):
    """Normaliza la config (URL o PG* sueltas) a host/port/user/password/database."""
    if "url" in pg:
        from sqlalchemy import make_url
        u = make_url(pg["url"])
        return {
            "host": u.host,
            "port": u.port or 5432,
            "user": u.username,
            "password": u.password,
            "database": u.database,
        }
    return pg


def _pgvector_tiene_datos(params, full_table):
    """True si la tabla del vector store ya tiene filas (índice ya construido)."""
    import psycopg2
    try:
        conn = psycopg2.connect(
            host=params["host"], port=params["port"], user=params["user"],
            password=params["password"], dbname=params["database"],
        )
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT to_regclass(%s)", (full_table,))
                if cur.fetchone()[0] is None:
                    return False
                cur.execute(f"SELECT COUNT(*) FROM {full_table}")
                return cur.fetchone()[0] > 0
        finally:
            conn.close()
    except Exception:
        return False


def _cargar_indice_postgres(pg, data_path, dim_actual):
    """Carga el índice desde Postgres/pgvector, o lo construye una sola vez.

    Persistir los vectores en Postgres elimina el cold-start de Railway: tras la
    primera construcción, cada redeploy solo los lee (segundos) en vez de volver
    a embeber ~25k chunks.
    """
    from llama_index.vector_stores.postgres import PGVectorStore

    params = _params_pg(pg)
    table_name = "netsuite_docs"
    embed_dim = dim_actual or 384

    vector_store = PGVectorStore.from_params(
        host=params["host"], port=params["port"], user=params["user"],
        password=params["password"], database=params["database"],
        table_name=table_name, embed_dim=embed_dim,
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )

    # PGVectorStore crea la tabla real con prefijo "data_".
    if _pgvector_tiene_datos(params, f"data_{table_name}"):
        index = VectorStoreIndex.from_vector_store(vector_store)
        return index, "Índice de la base de datos cargado correctamente."

    # Tabla vacía o inexistente: se construye desde los PDF y queda persistida.
    docs = _cargar_documentos(data_path)
    num_files = len({
        doc.metadata.get("file_name")
        for doc in docs if doc.metadata and "file_name" in doc.metadata
    }) if docs else 0
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        docs, storage_context=storage_context, show_progress=True,
    )
    return index, f"🆕 Índice creado en Postgres con {num_files} archivos."


def _guardar_comentario_pg(params, registro):
    """Inserta un comentario en la tabla `comentarios` (la crea si no existe)."""
    import psycopg2
    conn = psycopg2.connect(
        host=params["host"], port=params["port"], user=params["user"],
        password=params["password"], dbname=params["database"],
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                """CREATE TABLE IF NOT EXISTS comentarios (
                    id SERIAL PRIMARY KEY,
                    fecha TIMESTAMPTZ DEFAULT now(),
                    usuario TEXT,
                    comentario TEXT NOT NULL
                )"""
            )
            cur.execute(
                "INSERT INTO comentarios (usuario, comentario) VALUES (%s, %s)",
                (registro["usuario"], registro["comentario"]),
            )
        conn.commit()
    finally:
        conn.close()


def guardar_comentario(texto, usuario=None):
    """Persiste un comentario del usuario.

    Usa Postgres si Railway lo tiene configurado (sobrevive a los redeploys); en
    local, o si la BD falla, cae a un archivo `comentarios.jsonl` para no perder
    el comentario.
    """
    texto = (texto or "").strip()
    if not texto:
        return False, "Escribe un comentario antes de enviar."

    registro = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "usuario": usuario or "anónimo",
        "comentario": texto,
    }

    pg = _config_postgres()
    if pg is not None:
        try:
            _guardar_comentario_pg(_params_pg(pg), registro)
            return True, "¡Gracias! Tu comentario fue enviado. 🙌"
        except Exception:
            pass  # si la BD falla no perdemos el comentario: archivo local.

    try:
        with open("comentarios.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
        return True, "¡Gracias! Tu comentario fue enviado. 🙌"
    except Exception as e:
        return False, f"No se pudo guardar el comentario: {e}"


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

    def _leer_metadata():
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    # Dimensión del modelo de embeddings activo. Un índice persistido con otro
    # modelo (p. ej. el viejo de OpenAI, 1536-dim) es incompatible y debe
    # reconstruirse; si no, el retrieval falla con "shapes not aligned".
    try:
        dim_actual = len(Settings.embed_model.get_query_embedding("dim_check"))
    except Exception:
        dim_actual = None

    # --- CAPA 2: backend pgvector (Railway). Si hay Postgres configurado, el
    # índice vive en la BD y sobrevive a los redeploys (sin cold-start). En
    # local, sin variables PG*, se cae al índice JSON de ./storage de abajo. ---
    pg = _config_postgres()
    if pg is not None:
        try:
            return _cargar_indice_postgres(pg, data_path, dim_actual)
        except Exception as e:
            return None, f"❌ Error con Postgres: {e}"

    try:
        meta = _leer_metadata()
        indice_compatible = (
            _storage_utilizable()
            and dim_actual is not None
            and meta.get("embed_dim") == dim_actual
        )
        if indice_compatible:
            try:
                storage_context = StorageContext.from_defaults(persist_dir=storage_path)
                index = load_index_from_storage(storage_context)
                num_files = meta.get("num_files", 0)
                if num_files > 0:
                    return index, f"✅ Índice cargado con éxito ({num_files} archivos)."
                return index, "✅ Índice cargado con éxito."
            except Exception:
                # El índice existe pero no se pudo leer; se reconstruye desde ./datos.
                pass

        # No hay índice utilizable (falta, está dañado, son punteros LFS o cambió
        # el modelo de embeddings): se construye desde los PDF y se persiste en
        # ./storage (montado como Volumen de Railway para sobrevivir redeploys).
        docs = _cargar_documentos(data_path)

        num_files = 0
        if docs:
            file_names = {doc.metadata.get('file_name') for doc in docs if doc.metadata and 'file_name' in doc.metadata}
            num_files = len(file_names)

        # show_progress imprime una barra en el terminal mientras se generan los
        # embeddings (lento en CPU: ~25k chunks); sin esto parece "colgado".
        index = VectorStoreIndex.from_documents(docs, show_progress=True)

        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
        index.storage_context.persist(persist_dir=storage_path)
        with open(metadata_file, "w") as f:
            json.dump({"num_files": num_files, "embed_dim": dim_actual}, f)

        return index, f"🆕 Índice creado con {num_files} archivos."
    except Exception as e:
        return None, f"❌ Error: {e}"

@st.cache_resource(show_spinner="Cargando modelo de embeddings local... 🔤")
def configurar_modelos():
    # Embeddings locales (gratis, sin OpenAI): multilingüe para preguntas en
    # español sobre documentación en inglés. Se cachea en el Volumen.
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-base",
        cache_folder="./storage/.hf_cache",
        query_instruction="query: ",   # e5 requiere estos prefijos para
        text_instruction="passage: ",  # buen retrieval
        embed_batch_size=64,           # lotes grandes aprovechan la GPU (MPS)
    )
    # chunk_size alineado al límite del modelo (512 tok) para no truncar.
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    # OpenAI solo para generar las respuestas (modelo económico).
    Settings.llm = OpenAI(model="gpt-4o-mini")
    return True

load_dotenv()
configurar_modelos()
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

    st.markdown("---")
    st.subheader("💬 Comentarios")
    # clear_on_submit limpia el recuadro tras enviar; el form agrupa el text
    # area y el botón para que solo se procese al pulsar "Enviar comentarios".
    with st.form("form_comentarios", clear_on_submit=True):
        comentario_txt = st.text_area(
            "¿Tienes una sugerencia o reportas un problema?",
            placeholder="Escribe tu comentario aquí...",
            height=100,
        )
        if st.form_submit_button("Enviar comentarios", use_container_width=True):
            ok, msg = guardar_comentario(comentario_txt)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

# --- 4. DISEÑO PRINCIPAL ---
col1, col2 = st.columns([1, 3])
with col1:
    if os.path.exists("imagenes/logo.jpg"): st.image("imagenes/logo.jpg", width=150)
with col2:
    st.title("💼 Asistente Virtual de Netsuite")

def _seleccionar_ejemplo(pregunta):
    # Callback: corre ANTES del rerun, así el clic SIEMPRE queda registrado.
    # Antes, los botones se re-renderizaban junto a la primera respuesta y al
    # pulsar otro ya no existían en el siguiente run: el clic se perdía y el
    # chat parecía congelarse.
    st.session_state.pre_filled_prompt = pregunta
    st.session_state.send_prompt = True

# Ocultamos los ejemplos en cuanto hay historial o una pregunta ya en cola.
if not st.session_state.messages and not st.session_state.send_prompt:
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
            st.button(
                question,
                key=f"q_button_{i}",
                use_container_width=True,
                on_click=_seleccionar_ejemplo,
                args=(question,),
            )

# --- CAPA 3: caché de respuestas. Preguntas repetidas (botones de ejemplo,
# dudas frecuentes) se sirven al instante y sin gastar tokens de OpenAI. El dict
# vive en el proceso (compartido entre sesiones) gracias a cache_resource. ---
@st.cache_resource(show_spinner=False)
def _cache_respuestas():
    return {}

# --- 5. MOTOR DE CHAT (MEJORA: REEMPLAZA QUERY_ENGINE POR CHAT_ENGINE) ---
if index:
    if "chat_engine" not in st.session_state:
        # CAPA 3: menos historial por llamada (3900 -> 1500 tokens) recorta los
        # tokens enviados a OpenAI sin perder el contexto inmediato.
        memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
        st.session_state.chat_memory = memory  # ref para inyectar en cache hits
        st.session_state.chat_engine = index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            similarity_top_k=2,  # CAPA 3: 2 chunks en vez del default reduce contexto
            system_prompt="Eres un experto en NetSuite. Responde de forma concisa, clara y estructurada. Usa negritas para los puntos clave."
        )
else:
    st.stop()

def _resolver_ruta_pdf(nombre, ruta):
    """Devuelve una ruta existente al PDF de origen, o None.

    El índice guarda file_path como ruta ABSOLUTA del Mac donde se construyó
    (/Users/.../datos/X.pdf), que no existe en Railway. Por eso resolvemos
    primero por nombre contra ./datos (versionado en git, presente en prod) y
    solo usamos la ruta guardada como último recurso (sirve en local).
    """
    candidatos = []
    if nombre:
        candidatos.append(os.path.join("datos", nombre))
    if ruta:
        candidatos.append(os.path.join("datos", os.path.basename(ruta)))
        candidatos.append(ruta)
    for c in candidatos:
        if c and os.path.exists(c):
            return c
    return None


def mostrar_fuentes_persistentes(fuentes_list, id_mensaje):
    # Descarga directa del/los manual(es) de donde salió la respuesta, visible
    # sin abrir el expander y sin repetir archivos.
    vistos = []
    nombres_vistos = set()
    for f in fuentes_list:
        if f['nombre'] in nombres_vistos:
            continue
        nombres_vistos.add(f['nombre'])
        vistos.append(f)

    if vistos:
        st.markdown("**📥 Descargar manual de origen:**")
        cols = st.columns(min(len(vistos), 3))
        for j, f in enumerate(vistos):
            with cols[j % len(cols)]:
                ruta_real = _resolver_ruta_pdf(f['nombre'], f.get('ruta'))
                if ruta_real:
                    with open(ruta_real, "rb") as file_data:
                        st.download_button(
                            label=f"📖 {f['nombre']}",
                            data=file_data,
                            file_name=f['nombre'],
                            mime="application/pdf",
                            key=f"dl_{id_mensaje}_{j}",
                            use_container_width=True,
                        )
                else:
                    st.caption(f"📄 {f['nombre']} (archivo no disponible)")

    with st.expander("🔍 Ver referencia exacta (Archivo y Página)"):
        for i, f in enumerate(fuentes_list):
            st.markdown(f"**Referencia #{i+1}:**")
            st.markdown(f"📄 **Archivo:** `{f['nombre']}`")
            st.markdown(f"📑 **Página:** `{f['pagina']}`")
            st.markdown(f"📝 **Texto original:** _{f['texto']}_")
            st.divider()


def generar_sugerencias(pregunta, respuesta, n=3):
    """Propone preguntas de seguimiento NUEVAS y relevantes a partir de la
    respuesta dada. Devuelve [] si algo falla, para nunca romper el chat."""
    try:
        prompt = (
            "Eres un asistente de NetSuite. A partir de esta conversación, "
            f"propón {n} preguntas de seguimiento breves, claras y DISTINTAS "
            "entre sí que el usuario podría querer hacer a continuación. "
            "Responde SOLO con las preguntas, una por línea, en español, sin "
            "numeración ni viñetas.\n\n"
            f"Pregunta del usuario: {pregunta}\n\n"
            f"Respuesta:\n{respuesta[:1500]}\n\n"
            "Preguntas de seguimiento:"
        )
        salida = str(Settings.llm.complete(prompt))
        preguntas = []
        for linea in salida.splitlines():
            t = re.sub(r"^\s*(?:\d+[\.\)]|[-*•])\s*", "", linea).strip()
            if len(t) > 8 and t.endswith("?"):
                preguntas.append(t)
        return preguntas[:n]
    except Exception:
        return []


def mostrar_sugerencias(sugerencias, id_mensaje):
    """Botones clicables con las preguntas de seguimiento. Al pulsar uno se
    envía esa pregunta (mismo mecanismo que los ejemplos iniciales)."""
    if not sugerencias:
        return
    st.markdown("**💡 Preguntas relacionadas:**")
    cols = st.columns(min(len(sugerencias), 3))
    for j, q in enumerate(sugerencias):
        with cols[j % len(cols)]:
            st.button(
                q,
                key=f"sug_{id_mensaje}_{j}",
                use_container_width=True,
                on_click=_seleccionar_ejemplo,
                args=(q,),
            )


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

        cache = _cache_respuestas()
        clave = prompt_to_process.strip().lower()
        # Solo cacheamos preguntas sin historial previo: así la respuesta depende
        # únicamente de la pregunta (determinista) y el cache es siempre válido.
        # (el mensaje del usuario ya se agregó arriba, por eso <= 1).
        sin_historial = len(st.session_state.messages) <= 1

        if sin_historial and clave in cache:
            # CACHE HIT: respuesta instantánea, 0 tokens a OpenAI.
            cached = cache[clave]
            full_response, fuentes_encontradas = cached[0], cached[1]
            sugerencias = cached[2] if len(cached) > 2 else []
            response_placeholder.markdown(full_response)
            # Mantener la memoria del chat coherente para posibles follow-ups.
            if "chat_memory" in st.session_state:
                st.session_state.chat_memory.put(
                    ChatMessage(role=MessageRole.USER, content=prompt_to_process))
                st.session_state.chat_memory.put(
                    ChatMessage(role=MessageRole.ASSISTANT, content=full_response))
        else:
            # CACHE MISS: streaming normal (palabra por palabra) vía OpenAI.
            full_response = ""
            response = st.session_state.chat_engine.stream_chat(prompt_to_process)
            for token in response.response_gen:
                full_response += token
                response_placeholder.markdown(full_response + "▌")  # Cursor de escritura
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

            # Preguntas de seguimiento nuevas, basadas en la respuesta recién dada.
            sugerencias = generar_sugerencias(prompt_to_process, full_response)

            # Guardar en caché solo preguntas sin historial (cap de 200 entradas).
            if sin_historial and len(cache) < 200:
                cache[clave] = (full_response, fuentes_encontradas, sugerencias)

        mostrar_fuentes_persistentes(fuentes_encontradas, len(st.session_state.messages))

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "fuentes": fuentes_encontradas,
            "sugerencias": sugerencias,
        })

# Sugerencias de seguimiento: solo debajo de la ÚLTIMA respuesta del asistente.
# Se renderiza aquí (fuera del bloque de chat) para que aparezcan al pie tanto
# tras una respuesta nueva como al recargar el historial.
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    mostrar_sugerencias(
        st.session_state.messages[-1].get("sugerencias", []),
        len(st.session_state.messages) - 1,
    )