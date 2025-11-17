
import os, json, numpy as np
import streamlit as st
import sympy as sp
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

st.set_page_config(page_title="Formulas + IA", layout="centered", page_icon="🧠")
load_dotenv()

# ------------------------------
# Paths / Models
# ------------------------------
KB_PATH = Path(__file__).parent / "kb" / "casos4.json"
EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.header("⚙️ Configuracion")
api_key = st.sidebar.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY") or "")
model_name = st.sidebar.text_input("Modelo LLM", value=os.getenv("OPENAI_MODEL","gpt-4o-mini"))
use_llm = st.sidebar.checkbox("Usar LLM para explicacion", value=False)

c1,c2 = st.sidebar.columns(2)
if c1.button("Validar Key"):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
        _ = list(client.models.list())
        st.sidebar.success("✅ API Key valida.")
    except Exception as e:
        st.sidebar.error(f"❌ Error al validar: {e}")
if c2.button("Limpiar Key"):
    if "OPENAI_API_KEY" in os.environ: del os.environ["OPENAI_API_KEY"]
    st.sidebar.info("API Key eliminada")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_MODEL"] = model_name

# Normalizar etiquetas de industria (sinónimos)
IND_MAP = {
    "ia": {"ia","inteligencia artificial","ml","ai","machine learning","datos"},
    "biomedicina": {"biomedicina","salud","medicina","ecg","ecografía","bio"},
    "finanzas": {"finanzas","banca","riesgo","trading","mercados"},
    "energia": {"energia","energía","oil&gas","oil gas","electricidad"},
    "automotriz": {"automotriz","automotive","vehiculos","vehículos","manufactura"}
}

def norm_tag(t: str) -> str:
    t = (t or "").strip().lower()
    for k, vals in IND_MAP.items():
        if t in vals or t == k: return k
    return t

def normalize_industries(tags):
    return [norm_tag(t) for t in tags if t]

industria = st.sidebar.multiselect(
    "Filtrar por industria (opcional)",
    ["automotriz","finanzas","ia","energia","biomedicina"], default=[]
)

if industria:
    st.info(f"Industria seleccionada: {', '.join(industria)} — toca 'Analizar' para actualizar.")

# ------------------------------
# KB / Embeddings
# ------------------------------
def load_kb_robusto(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        st.warning("Error UTF-8; probando latin-1...")
    try:
        with open(path, "r", encoding="latin-1") as f:
            data = f.read()
        repl = {"“":'"',"”":'"',"„":'"',"‟":'"',"‘":"'","’":"'","‚":"'","‛":"'","–":"-","—":"-","−":"-","\u2028":" ","\u2029":" "}
        for k,v in repl.items():
            data = data.replace(k,v)
        return json.loads(data)
    except Exception as e:
        st.error(f"No pude leer kb/casos3.json: {e}")
        return []

@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer(EMBEDDER)

@st.cache_resource(show_spinner=False)
def load_index():
    kb = load_kb_robusto(KB_PATH)
    # normalizamos tags al cargar
    for x in kb:
        x["tags"] = normalize_industries(x.get("tags", []))
    texts = [f"{x.get('title','')} - {x.get('content','')} (tags: {', '.join(x.get('tags', []))})" for x in kb]
    model = load_embedder()
    X = model.encode(texts, normalize_embeddings=True)
    nn = NearestNeighbors(n_neighbors=min(10, len(kb)), metric="cosine")
    nn.fit(X)
    return kb, nn, X, model

kb, nn, X, embedder = load_index()

# ------------------------------
# Utils
# ------------------------------
def guess_topic(info):
    s = str(info["expr"])
    if any(fn in s for fn in ("sin","cos","tan")): return "trigonometria"
    if "exp" in s: return "exponencial"
    if "**" in s: return "polinomio"
    if info["int_latex"]: return "integral"
    if info["deriv_latex"]: return "derivada"
    return "algebra lineal"

def retrieve(query: str, k=10, must_all=None, any_tags=None, min_score=0.15):
    # must_all = lista de industrias normalizadas que deben estar presentes
    if must_all is None: must_all = []
    if any_tags is None: any_tags = []
    qv = embedder.encode([query], normalize_embeddings=True)
    dists, idxs = nn.kneighbors(qv, n_neighbors=min(k, len(kb)))
    cand = []
    for i, dist in zip(idxs[0], dists[0]):
        item = kb[i]; tags = item.get("tags", []); score = 1 - float(dist)
        if must_all and not all(t in tags for t in must_all):
            continue
        if any_tags and not any(t in tags for t in any_tags):
            continue
        if score >= min_score: cand.append({"score":score,"item":item})
    cand.sort(key=lambda r: r["score"], reverse=True)
    return cand[:3]

def strict_context(results, industrias):
    """Devuelve solo snippets cuyas tags intersecten la industria seleccionada.
       Si queda vacío, devolvemos lista vacía (preferible a mezclar industrias)."""
    if not industrias:
        return results
    keep = []
    industries_set = set(industrias)
    for r in results:
        tags = set(r["item"].get("tags", []))
        if tags & industries_set:
            keep.append(r)
    return keep

def call_llm(system_prompt, user_prompt):
    if not use_llm or not os.getenv("OPENAI_API_KEY"):
        return "Modo local: explicacion sin IA."
    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL","gpt-4o-mini"),
            messages=[{"role":"system","content":system_prompt},
                      {"role":"user","content":user_prompt}],
            temperature=0.3, max_tokens=450
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"[Error LLM] {e}"

# Parser seguro: admite 'y=W*x' y devuelve solo RHS para cálculo
def parse_formula_to_expr(s: str):
    s = (s or "").strip().replace("^","**")
    if "=" in s:
        try:
            lhs, rhs = [t.strip() for t in s.split("=",1)]
            # registrar símbolos (rápido: letras)
            names = sorted(set(ch for ch in (lhs+rhs) if ch.isalpha()))
            if names: sp.symbols(",".join(names))
            return sp.sympify(rhs, convert_xor=True)
        except Exception:
            pass
    return sp.sympify(s, convert_xor=True)

def analyze_formula(expr_str):
    x = sp.Symbol('x')
    expr = parse_formula_to_expr(expr_str)
    try: deriv = sp.diff(expr, x)
    except Exception: deriv = None
    try: integ = sp.integrate(expr, x)
    except Exception: integ = None
    return {"expr":expr,"x":x,"latex":sp.latex(expr),
            "deriv_latex": sp.latex(deriv) if deriv is not None else None,
            "int_latex": sp.latex(integ) if integ is not None else None}

# ------------------------------
# UI
# ------------------------------
st.title("📱 Formulas Matematicas e Industrias")
example = st.selectbox("Ejemplos", ["1/(1+exp(-x))","sin(x)","x**2 + 3*x + 2","exp(-x) * sin(x)","log(x)","1/(1+x)","y=W*x"], index=1)
formula = st.text_input("Escribi una formula (SymPy):", example)

if st.button("Analizar"):
    try:
        info = analyze_formula(formula.strip())
        topic = guess_topic(info)

        st.subheader("🧮 Interpretacion")
        st.latex(info["latex"])
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**Derivada**"); st.latex(info["deriv_latex"] or "N/D")
        with c2:
            st.markdown("**Integral**"); st.latex(info["int_latex"] or "N/D")

        query = f"{topic} {info['latex']}"
        # 1) Buscar SOLO en industria seleccionada
        results = retrieve(query, k=10, must_all=industria, any_tags=[topic], min_score=0.18)
        # 2) Si no hay nada, intentamos por tema pero mantenemos contexto estricto por industria (si hay)
        if not results:
            tmp = retrieve(query, k=10, must_all=[], any_tags=[topic], min_score=0.12)
            results = strict_context(tmp, industria)
        # 3) Si aún vacío y hay industria, preferimos NO mezclar industrias -> contexto vacío
        if not results and not industria:
            results = retrieve(query, k=10, must_all=[], any_tags=[], min_score=0.10)

        st.subheader("🏭 Casos sugeridos")
        if results:
            for r in results:
                st.markdown(f"- **{r['item']['title']}** ({r['score']:.2f}) — {r['item']['content']} *(tags: {', '.join(r['item'].get('tags', []))})*")
        else:
            st.info("No hay casos coincidentes para la industria seleccionada. Se generará explicación sin ejemplos del RAG para evitar mezclar industrias.")

        industria_str = ", ".join(industria) if industria else "sin preferencia"
        context = "\n".join([f"({i+1}) {r['item']['title']}: {r['item']['content']} [tags: {', '.join(r['item'].get('tags', []))}]"
                             for i,r in enumerate(results)])

        # ------------------------------
        # LLM Prompting: contexto limpio por industria
        # ------------------------------
        st.subheader("🧠 Explicacion IA")
        system_prompt = (
            "Sos un profesor de matematicas aplicadas por industria. "
            "Respetá estrictamente la industria objetivo; no menciones otras."
        )
        user_prompt = f"""
Formula (LaTeX): {info['latex']}
Tema aproximado: {topic}
Industria objetivo (EXCLUSIVA): {industria_str}
Resultados SymPy: derivada={info['deriv_latex']} ; integral={info['int_latex']}

Contexto (RAG solo de la industria): 
{context if results else "[sin contexto RAG para evitar mezclar industrias]"} 

INSTRUCCIONES ESTRICTAS:
- Restringí aplicaciones y ejemplos a la industria objetivo declarada arriba.
- Si el contexto RAG trajera otra industria, IGNORALO y NO LA MENCIONES.
- Si no hay contexto válido para esa industria, inventá un ejemplo PLAUSIBLE pero quedate en esa industria.
- Estructura obligatoria de salida:
  ## Explicacion
  ## Aplicacion (industria objetivo)
  ## Verificacion
  ## Fuentes
"""
        st.write(call_llm(system_prompt, user_prompt))

        # ------------------------------
        # Plot
        # ------------------------------
        st.subheader("📈 Grafico")
        x = info["x"]; expr = info["expr"]
        xs = np.linspace(-5,5,300); ys = [float(expr.subs(x, v)) for v in xs]
        fig = plt.figure()
        plt.plot(xs, ys)
        plt.title(f"f(x) = {sp.latex(expr)}"); plt.xlabel("x"); plt.ylabel("f(x)")
        st.pyplot(fig)

    except Exception as e:
        st.error(e)
