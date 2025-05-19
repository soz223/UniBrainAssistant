# UniBrain Streamlit App — Refactored & Simplified
# =================================================
# (Outputs now appear inside collapsible **expanders** so the page stays tidy.)
# Author: Songlin Zhao (2025‑05‑07)

"""
This is a *drop‑in* replacement for the original `app.py`.  

Major refactor goals ✨  
- **Single responsibility sections** – imports, constants, utils, model, UI, main.  
- **No duplicated code** – every function defined exactly once.  
- **Keep public API unchanged** – all externally‑called functions/classes retain name & signature.  
- **Easy navigation** – each section delimited by clear ASCII rulers.  

Functionality is *identical* to the pre‑refactor version.
"""

# ╭──────────────────────────────────────────────────────────────────────────╮
# 1. Imports
# ╰──────────────────────────────────────────────────────────────────────────╯
from __future__ import annotations

# ---- Standard -------------------------------------------------------------
import json, os, re, time, uuid, warnings
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

# ---- Third‑party ----------------------------------------------------------
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import nibabel as nib
import SimpleITK as sitk
import torch
import streamlit as st
from PIL import Image
from skimage.transform import resize

# ---- LangChain / OpenAI ---------------------------------------------------
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate,
)
from openai import OpenAI

from dotenv import load_dotenv
import os

load_dotenv()  # reads .env into os.environ


# ╭──────────────────────────────────────────────────────────────────────────╮
# 2. Configuration & Constants
# ╰──────────────────────────────────────────────────────────────────────────╯
IMG_SIZE  : int              = 96
DEVICE    : torch.device     = torch.device("cpu")
ROOT      : Path             = Path(__file__).resolve().parent
ASSETS    : Path             = ROOT / "assets"
PDF_PATH  : Path             = ROOT / "unibrain.pdf"
TXT_PATH  : Path             = ROOT / "extra_knowledge.txt"
VDB_PATH  : Path             = ROOT / "extra_knowledge.faiss"
PROMPT_PATH: Path            = ROOT / "prompts" / "unibrain_system_prompt.md"

NIIGZ     : str              = "application/gzip"  # universal mime for .nii.gz
DEFAULT_STEPS: List[str]     = [
    "Extraction", "Registration", "Segmentation",
    "Parcellation", "Network", "Classification",
]

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")  # env > hard‑code

# Global Streamlit page setup (once!)
st.set_page_config(
    page_title="UniBrain Assistant",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ╭──────────────────────────────────────────────────────────────────────────╮
# 3. Streamlit Session State Defaults
# ╰──────────────────────────────────────────────────────────────────────────╯
for k, v in {
    "selected_steps": DEFAULT_STEPS.copy(),
    "pipeline_done" : False,
    "messages"      : [],
}.items():
    st.session_state.setdefault(k, v)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Helper for cross‑version re‑run
def _rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


# ╭──────────────────────────────────────────────────────────────────────────╮
# 4. Utility Functions
# ╰──────────────────────────────────────────────────────────────────────────╯
## 4‑A. NIfTI ↔ torch helpers ------------------------------------------------

def nii_to_torch(path: Path) -> torch.Tensor:
    """Read .nii/.nii.gz → torch tensor NCDHW (1,1,D,H,W) on DEVICE."""
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
    if arr.shape != (IMG_SIZE,) * 3:
        arr = resize(arr, (IMG_SIZE,) * 3, order=1, preserve_range=True, anti_aliasing=True)
    return torch.from_numpy(arr)[None, None].to(DEVICE)


def save_tensor(t: torch.Tensor, out: Path) -> None:
    t = torch.nan_to_num(t).cpu().squeeze().float()
    sitk.WriteImage(sitk.GetImageFromArray(t.numpy()), str(out))


## 4‑B. Colour LUT & ROI utilities ------------------------------------------

def _make_cmap(n: int) -> np.ndarray:
    base = (
        plt.get_cmap("tab20").colors
        + plt.get_cmap("tab20b").colors
        + plt.get_cmap("tab20c").colors
    ) * 50
    return np.vstack([[0, 0, 0], np.array(base[: n - 1])]).astype(np.float32)


def roi_slice_to_rgb(label2d: np.ndarray) -> np.ndarray:
    max_id = int(label2d.max())
    lut = (_make_cmap(max_id + 1) * 255).astype(np.uint8)
    return lut[label2d]


## 4‑C. Adjacency helpers ----------------------------------------------------
@st.cache_data
def load_adj(path: str) -> np.ndarray:
    return torch.load(path, map_location="cpu").cpu().numpy()


def adj_to_graph_random(arr: np.ndarray, keep_ratio: float, seed: int = 42) -> nx.Graph:
    A = np.squeeze(arr)
    if A.ndim != 2:
        raise ValueError("Adj must be 2‑D — got %s" % (arr.shape,))

    rng = np.random.default_rng(seed)
    idx_u = np.triu_indices_from(A, k=1)
    ones  = np.flatnonzero(A[idx_u])
    mask_u = np.zeros_like(idx_u[0], dtype=bool)

    if keep_ratio < 1.0:
        n_keep = max(1, int(len(ones) * keep_ratio))
        keep_idx = rng.choice(ones, size=n_keep, replace=False)
        mask_u[keep_idx] = True
    else:
        mask_u[ones] = True

    mask = np.zeros_like(A, dtype=bool)
    mask[idx_u] = mask_u
    mask |= mask.T
    return nx.from_numpy_array(mask, create_using=nx.Graph)


def show_adj_heatmap(arr: np.ndarray) -> None:
    st.write("### Heatmap")
    arr2d = np.squeeze(arr)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(arr2d, aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.046)
    st.pyplot(fig, clear_figure=True)


def show_adj_graph(arr: np.ndarray, key_prefix: str) -> None:
    st.write("### Network graph (% edges)")
    keep = (
        st.slider("Keep top‑X % edges", 5, 100, 10, 5, key=f"{key_prefix}_keep")
        / 100.0
    )
    G = adj_to_graph_random(arr, keep)
    if G.number_of_edges() == 0:
        st.info("No edges retained at this percentage.")
        return

    pos = nx.spring_layout(G, seed=42, k=1 / np.sqrt(max(G.number_of_nodes(), 1)))
    fig, ax = plt.subplots(figsize=(5, 5))
    nx.draw_networkx_nodes(G, pos, node_size=20, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.25, width=1, ax=ax)
    ax.set_title(f"{G.number_of_nodes()} nodes | {G.number_of_edges()} edges")
    ax.axis("off")
    st.pyplot(fig, clear_figure=True)


# ╭──────────────────────────────────────────────────────────────────────────╮
# 5. Model & Cached Resources
# ╰──────────────────────────────────────────────────────────────────────────╯
## 5‑A. Dummy fallback -------------------------------------------------------
try:
    from model import UniBrain  # type: ignore
    MODEL_OK = True
except ImportError:

    class UniBrain(torch.nn.Module):
        def forward(self, *a, **k):
            b = a[1].shape[0]
            shp = (b, 1, IMG_SIZE, IMG_SIZE, IMG_SIZE)
            rand = lambda: torch.rand(shp, device=a[1].device)
            eye = lambda: torch.eye(4, device=a[1].device).unsqueeze(0)
            return (
                [rand()] * 3,
                [rand()] * 3,
                [rand()] * 3,
                [eye()] * 2,
                [eye()] * 2,
                rand(),
                rand(),
                rand(),
                rand(),
                torch.rand(b, 10, 10),
                torch.rand(b, 2),
            )

    MODEL_OK = False


## 5‑B. Cache wrappers -------------------------------------------------------
@st.cache_resource(show_spinner="⏳ Loading templates …")
def load_templates():
    def _load(name: str):
        arr = np.load(ASSETS / name).astype(np.float32)
        return torch.from_numpy(arr)[None, None].to(DEVICE)

    try:
        return tuple(_load(p) for p in ("tpl_img.npy", "tpl_gm.npy", "tpl_aal.npy"))
    except Exception as e:
        st.error(f"Template error: {e}")
        return (None, None, None)


@st.cache_resource(show_spinner="⏳ Loading UniBrain …")
def load_model():
    if not MODEL_OK:
        st.warning("Using dummy model (import failed)")
        return None
    try:
        m = UniBrain(img_size=IMG_SIZE, ext_stage=3, reg_stage=2, if_pred_aal=True).to(DEVICE)
        m.load_state_dict(torch.load(ASSETS / "unibrain.pth", map_location=DEVICE))
        m.eval()
        return m
    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None


REF_TPL, REF_GM, REF_AAL = load_templates()
MODEL = load_model()

# ╭──────────────────────────────────────────────────────────────────────────╮
# 6. Knowledge‑RAG Tool
# ╰──────────────────────────────────────────────────────────────────────────╯
@st.cache_resource(show_spinner="🔍 Building vector store …")
def vector_store():
    if not OPENAI_API_KEY:
        st.error("Missing OpenAI key")
        return None

    docs = []
    if PDF_PATH.exists():
        docs += PyPDFLoader(str(PDF_PATH)).load()
    if TXT_PATH.exists():
        docs += TextLoader(str(TXT_PATH), encoding="utf-8").load()
    if not docs:
        st.error("No knowledge docs found")
        return None

    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150).split_documents(docs)
    vdb = FAISS.from_documents(chunks, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
    vdb.save_local(str(VDB_PATH))
    return vdb


def ask_knowledge(q: str) -> str:
    vdb = vector_store()
    if vdb is None:
        return "Knowledge DB unavailable."

    retriever = vdb.as_retriever(search_kwargs={"k": 4})
    ctx = "\n\n".join(d.page_content for d in retriever.get_relevant_documents(q))
    prompt = (
        "You are UniBrain‑RAG assistant. Use the context only.\n\n"
        f"Context:\n{ctx}\n\nQ: {q}"
    )

    return ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-4o",
        temperature=0,
    ).invoke(prompt).content


KNOWLEDGE_TOOL = Tool(
    name="ask_unibrain_paper",
    func=ask_knowledge,
    description="Answer UniBrain paper questions",
    return_direct=True,
)

# ╭──────────────────────────────────────────────────────────────────────────╮
# 7. Inference Pipeline (public API function)
# ╰──────────────────────────────────────────────────────────────────────────╯
CARD = Dict[str, Any]  # convenience alias


def run_pipeline(inp: Path, work: Path, steps: List[str] | None = None) -> List[CARD]:
    """Run UniBrain end‑to‑end and write outputs under *work* directory."""
    if None in (MODEL, REF_TPL, REF_GM, REF_AAL):
        raise RuntimeError("Model / templates not ready")

    if steps is None:
        steps = DEFAULT_STEPS

    mov = nii_to_torch(inp)
    st.info("Running UniBrain …")
    t0 = time.time()
    with torch.no_grad():
        (
            striped,
            masks,
            warped,
            theta,
            theta_i,
            am_mov,
            am_ref2,
            aal_mov,
            aal_ref2,
            adj,
            logits,
        ) = MODEL(REF_TPL, mov, REF_GM, REF_AAL, if_train=False)
    st.success(f"Done in {time.time() - t0:.1f}s")

    nii_dir, pt_dir = work / "nii", work / "pt"
    nii_dir.mkdir(parents=True, exist_ok=True)
    pt_dir.mkdir(exist_ok=True)

    cards: List[CARD] = []

    def _nii_card(tensor: torch.Tensor, name: str, note: str = "") -> Path:
        out = nii_dir / f"{name}.nii.gz"
        save_tensor(tensor, out)
        cards.append({
            "step": name.replace("_", " ").title(),
            "nifti_path": out,
            "explanation": note,
        })
        return out

    # ---- Classification ---------------------------------------------------
    if "Classification" in steps:
        prob = torch.softmax(logits, 1)[0]
        cls, conf = int(prob.argmax()), float(prob.max())
        torch.save(logits.cpu(), pt_dir / "logits.pt")
        cards.append({
            "step": "Classification",
            "metrics": {"class": cls, "probability": conf},
            "explanation": f"Predicted **{cls}** (p={conf:.3f})",
            "file_path": pt_dir / "logits.pt",
        })

    # ---- Segmentation / Parcellation -------------------------------------
    if "Segmentation" in steps:
        _nii_card(torch.argmax(am_mov, 1), "am_seg", "Anatomical mask")
    if "Parcellation" in steps:
        _nii_card(torch.argmax(aal_mov, 1), "aal_seg", "AAL labels")

    # ---- Registration details ---------------------------------------------
    if "Registration" in steps:
        _nii_card(am_ref2, "am_ref2mov", "Template ANAT→input")
        _nii_card(aal_ref2, "aal_ref2mov", "Template AAL→input")
        for i, w in enumerate(warped, 1):
            _nii_card(w, f"warped{i}")

    # ---- Extraction masks --------------------------------------------------
    if "Extraction" in steps:
        for i, (m, s) in enumerate(zip(masks, striped), 1):
            _nii_card(m, f"mask{i}")
            _nii_card(s, f"strip{i}")

    # ---- Brain network (adjacency) ----------------------------------------
    if "Network" in steps:
        adj_path = pt_dir / "adj.pt"
        torch.save(adj, adj_path)
        cards.append({
            "step": "Network",
            "file_path": adj_path,
            "explanation": "Adjacency matrix",
        })

    return cards


# ╭──────────────────────────────────────────────────────────────────────────╮
# 8. Streamlit UI Components
# ╰──────────────────────────────────────────────────────────────────────────╯
## 8‑A. Slice + volume viewers ---------------------------------------------
@st.cache_data
def _slice_norm(vol: np.ndarray, z: int) -> np.ndarray:
    sl = vol[:, :, z].astype(np.float32)
    rng = sl.ptp() or 1
    return ((sl - sl.min()) / rng * 255).astype(np.uint8)


def viewer(nifti: Path, title: str, key: str, *, color: bool = False) -> None:
    vol = nib.load(str(nifti)).get_fdata()
    z_max = vol.shape[2] - 1
    z = st.slider(f"{title} – slice", 0, z_max, z_max // 2, key=key)

    if color:
        lab = vol[:, :, z].astype(np.int32)
        lab[lab < 1] = 0
        st.image(roi_slice_to_rgb(lab), caption=f"{title} (z={z})", use_container_width=True)
    else:
        st.image(_slice_norm(vol, z), caption=f"{title} (z={z})", use_container_width=True)


def vol_fig(arr: np.ndarray, ds: int, colourscale: str) -> go.Figure:
    arr_ds = arr[::ds, ::ds, ::ds]
    x, y, z = np.mgrid[: arr_ds.shape[0], : arr_ds.shape[1], : arr_ds.shape[2]]
    nonmin = arr_ds[arr_ds > arr_ds.min()]
    imin, imax = (np.percentile(nonmin, (5, 98)) if nonmin.size else (0, 1))
    fig = go.Figure(
        go.Volume(
            x=x.ravel(), y=y.ravel(), z=z.ravel(), value=arr_ds.ravel(),
            isomin=imin, isomax=imax,
            opacity=0.2,
            opacityscale=[[0, 0], [0.1, 0.05], [0.3, 0.1], [0.6, 0.3], [0.8, 0.5], [1, 0.7]],
            surface_count=17,
            colorscale=colourscale,
        )
    )
    fig.update_layout(scene=dict(aspectmode="data"), margin=dict(l=0, r=0, t=30, b=0), height=500,
                      title=dict(text="3‑D Volume Preview", x=0.5))
    return fig


## 8‑B. Upload & preview helpers -------------------------------------------

def handle_upload() -> Optional[Path]:
    upload = st.file_uploader("Upload NIfTI (.nii/.nii.gz)", ["nii", "nii.gz"])
    if not upload:
        return None

    if st.session_state.get("fname") != upload.name:
        job_id = uuid.uuid4().hex[:8]
        workdir = Path("uploads") / job_id
        workdir.mkdir(parents=True, exist_ok=True)
        up_path = workdir / upload.name
        up_path.write_bytes(upload.getbuffer())
        st.session_state.update(
            fname=upload.name,
            upload_path=str(up_path),
            work_dir=str(workdir),
            pipeline_done=False,
            cards=[],
        )
        st.success(f"Saved → {up_path}")
    return Path(st.session_state.upload_path)


def show_input_preview(in_path: Path) -> None:
    st.subheader("Input preview")
    viewer(in_path, "Input", "prev")

    # 3‑D preview controls
    ds = st.radio("Quality (↓ faster)", [4, 3, 2, 1], index=0, horizontal=True, key="quality_input")
    cmap = st.selectbox("Colormap", ["Greys", "Viridis", "Hot", "Rainbow", "Jet"], index=1, key="cmap_input")
    vol = nib.load(str(in_path)).get_fdata()
    st.plotly_chart(vol_fig(vol, ds, cmap), use_container_width=True, key="vol_input")


## 8‑C. Output cards --------------------------------------------------------

def is_parcellation(step: str) -> bool:
    step = step.lower()
    return any(k in step for k in ("aal_seg", "aal_ref2mov"))


def show_outputs() -> None:
    if not st.session_state.get("pipeline_done"):
        return

    st.header("Inference Outputs")
    for i, card in enumerate(st.session_state.cards):
        with st.expander(card["step"], expanded=False):
            st.write(card.get("explanation", ""))

            # ① classification metrics
            if m := card.get("metrics"):
                c1, c2 = st.columns(2)
                c1.metric("Class", m["class"])
                c2.metric("Confidence", f"{m['probability']:.3f}")

            # ② NIfTI outputs – 2D / 3D toggle
            if (p := card.get("nifti_path")) and Path(p).exists():
                view_mode = st.radio("Display mode", ["2D slice", "3D volume"], index=0, key=f"view_{i}")
                if view_mode == "2D slice":
                    viewer(Path(p), card["step"], f"out_{i}", color=is_parcellation(card["step"]))
                else:
                    ds = st.radio("Quality (↓ faster)", [4, 3, 2, 1], index=1, horizontal=True, key=f"quality_{i}")
                    cmap = st.selectbox("Colormap", ["Greys", "Viridis", "Hot", "Rainbow", "Jet"], index=1, key=f"cmap_{i}")
                    arr = nib.load(str(p)).get_fdata()
                    st.plotly_chart(vol_fig(arr, ds, cmap), use_container_width=True, key=f"vol_{i}")

                st.download_button("Download NIfTI", Path(p).read_bytes(), file_name=Path(p).name, mime=NIIGZ, key=f"dl_{i}")

            # ②‑b adj.pt visuals
            elif (fp := card.get("file_path")) and "adj" in Path(fp).stem:
                arr = load_adj(str(fp))
                mode = st.radio("Adjacency view", ["Heatmap", "Graph"], key=f"adj_view_{i}")
                if mode == "Heatmap":
                    show_adj_heatmap(arr)                          # no key_prefix
                else:
                    show_adj_graph(arr, key_prefix=f"adj_{i}")     # graph needs it
                st.download_button("Download adj.pt", Path(fp).read_bytes(), file_name=Path(fp).name, mime="application/octet‑stream", key=f"dl_adj_{i}")

            # ③ generic file download
            elif (fp := card.get("file_path")) and Path(fp).exists():
                st.download_button("Download data", Path(fp).read_bytes(), file_name=Path(fp).name, mime="application/octet‑stream", key=f"dlf_{i}")


# ╭──────────────────────────────────────────────────────────────────────────╮
# 9. Command Parser (regex + fallback LLM)
# ╰──────────────────────────────────────────────────────────────────────────╯
CMD_SYS_PROMPT = (
    "You are a controller for the UniBrain Streamlit app.\n"  # noqa: E501
    "Your ONLY job is to examine the user's sentence and decide\n"  # noqa: E501
    "• whether it requests skipping, enabling, or resetting pipeline steps.\n"  # noqa: E501
    "Valid step names: Extraction, Registration, Segmentation, Parcellation, Network, Classification.\n\n"  # noqa: E501
    "Treat these patterns all as “skip <Step>” commands:\n"  # noqa: E501
    "  - “skip segmentation”\n  - “without segmentation”\n  - “no segmentation”\n"  # noqa: E501
    "OUTPUT RULES:\n"  # noqa: E501
    "1. If the user wants to skip a step, output EXACT JSON: {\"action\":\"skip\",\"step\":\"<Step>\"}\n"  # noqa: E501
    "2. If the user wants to enable a step, output EXACT JSON: {\"action\":\"enable\",\"step\":\"<Step>\"}\n"  # noqa: E501
    "3. If the user wants to reset everything, output EXACT JSON: {\"action\":\"reset\"}\n"  # noqa: E501
    "4. Otherwise output the literal word: none"
)

client = OpenAI(api_key=OPENAI_API_KEY)


def _regex_detect(cmd: str) -> Optional[Tuple[str, str]]:
    cmd_l = cmd.lower().strip()
    if "reset steps" in cmd_l or "reset pipeline" in cmd_l:
        return ("reset", "")
    if m := re.match(r"(skip|without|no)\s+(\w+)", cmd_l):
        action, step = m.groups()
        return ("skip", step.capitalize())
    if m := re.match(r"(enable|turn on)\s+(\w+)", cmd_l):
        _, step = m.groups()
        return ("enable", step.capitalize())
    return None


def _llm_detect(text: str) -> Optional[Tuple[str, str]]:
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": CMD_SYS_PROMPT}, {"role": "user", "content": text}],
            max_tokens=20,
            temperature=0,
        )
        out = resp.choices[0].message.content.strip()
        if out == "none":
            return None
        data = json.loads(out)
        if data.get("action") == "reset":
            return ("reset", "")
        if data.get("action") in ("skip", "enable") and data.get("step"):
            return (data["action"], data["step"].capitalize())
    except Exception as e:
        st.warning(f"Cmd‑parser LLM error: {e}")
    return None


def detect_command(text: str) -> Optional[Tuple[str, str]]:
    return _regex_detect(text) or _llm_detect(text)


# ╭──────────────────────────────────────────────────────────────────────────╮
# 10. Agent Initialisation (RUN & RAG tools)
# ╰──────────────────────────────────────────────────────────────────────────╯

def _run_unibrain(_: str = "") -> str:  # dummy param for Tool signature
    if st.session_state.get("pipeline_done"):
        return "✅ Already done"
    try:
        in_path = Path(st.session_state.upload_path)
        st.session_state.cards = run_pipeline(in_path, in_path.parent, st.session_state.selected_steps)
        st.session_state.pipeline_done = True
        return "✅ Inference complete"
    except Exception as e:
        return f"🚨 Failure: {e}"


RUN_TOOL = Tool(
    name="run_unibrain_inference",
    description="Run/rerun UniBrain on uploaded file",
    func=_run_unibrain,
    return_direct=True,
)

agent: Optional[Any] = None
if OPENAI_API_KEY:
    try:
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4o-mini", temperature=0)
        memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True)
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(PROMPT_PATH.read_text(encoding="utf-8")),
            HumanMessagePromptTemplate.from_template("{input}"),
        ])
        agent = initialize_agent(
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            tools=[RUN_TOOL, KNOWLEDGE_TOOL],
            llm=llm,
            prompt=prompt_template,
            memory=memory,
            max_iterations=5,
            verbose=False,
            handle_parsing_errors=True,
        )
    except Exception as e:
        st.error(f"Agent init failed: {e}")

# ╭──────────────────────────────────────────────────────────────────────────╮
# 11. Main Streamlit Flow
# ╰──────────────────────────────────────────────────────────────────────────╯

st.title("🧠 UniBrain Assistant")

# -- Sidebar pipeline step selector & run button ----------------------------
with st.sidebar.expander("⚙️ Pipeline steps", expanded=False):
    st.write("Select which stages to run:")
    st.session_state.selected_steps = st.multiselect(
        "Stages", DEFAULT_STEPS, default=st.session_state.selected_steps, key="steps_select"
    )

if st.sidebar.button("Run UniBrain", disabled=st.session_state.pipeline_done, key="sidebar_run"):
    if "upload_path" not in st.session_state:
        st.error("❗ Please upload a file first.")
    else:
        in_path = Path(st.session_state.upload_path)
        st.session_state.cards = run_pipeline(in_path, in_path.parent, st.session_state.selected_steps)
        st.session_state.pipeline_done = True
        st.session_state.messages.append({"role": "assistant", "content": "✅ Finished running: " + ", ".join(st.session_state.selected_steps)})
        _rerun()

# -- File upload & preview --------------------------------------------------
if (in_path := handle_upload()):
    show_input_preview(in_path)
    show_outputs()

# ╭──────────────────────────── Chat Area ───────────────────────────────────╮
st.divider()
st.subheader("Chat with UniBrain Assistant")

# 1) render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# 2) input box (only if agent ready & not waiting)
waiting_reply = st.session_state.messages and st.session_state.messages[-1]["role"] == "user"
user_text = None
if not waiting_reply and agent:
    user_text = st.chat_input("Ask about UniBrain …", key="main_chat_input")

# 3) handle user input ------------------------------------------------------
if user_text:
    if cmd := detect_command(user_text):
        action, step = cmd
        if action == "reset":
            st.session_state.selected_steps = DEFAULT_STEPS.copy()
            ack = "🔄 Steps reset to all enabled."
        elif action == "skip":
            st.session_state.selected_steps = [s for s in st.session_state.selected_steps if s != step]
            ack = f"⏭️ **{step}** skipped."
        else:  # enable
            if step not in st.session_state.selected_steps:
                st.session_state.selected_steps.append(step)
            ack = f"✅ **{step}** enabled."

        # rerun pipeline
        st.session_state.pipeline_done = False
        in_path = Path(st.session_state.upload_path)
        st.session_state.cards = run_pipeline(in_path, in_path.parent, st.session_state.selected_steps)
        st.session_state.pipeline_done = True

        st.session_state.messages += [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": ack},
        ]
        _rerun()
    else:
        st.session_state.messages.append({"role": "user", "content": user_text})
        _rerun()

# 4) generate assistant reply ----------------------------------------------
if waiting_reply and agent:
    question = st.session_state.messages[-1]["content"]
    ctx = f"File: {st.session_state.upload_path}" if "upload_path" in st.session_state else "No file."

    with st.chat_message("assistant"), st.spinner("Thinking…"):
        try:
            res = agent.invoke({"input": question + "\n\nContext: " + ctx})
            answer = res.get("output", str(res))
        except Exception as e:
            answer = f"Agent error: {e}"

        st.session_state.messages.append({"role": "assistant", "content": answer})
        _rerun()

elif not agent:
    st.info("Agent disabled (no API key)")
