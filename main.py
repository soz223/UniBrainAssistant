"""
UniBrain Streamlit App — concise, maintainable refactor
======================================================
(Outputs appear inside collapsible expanders. Click to reveal/download.)

Author: Songlin Zhao (refactor by assistant)
"""
from __future__ import annotations

# ── Std / 3rd‑party ─────────────────────────────────────────────────────────
import os
from dotenv import load_dotenv
load_dotenv()
import uuid
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import streamlit as st
import SimpleITK as sitk
import nibabel as nib
import matplotlib.pyplot as plt
import networkx as nx
from skimage.transform import resize

# ── LangChain & tools ───────────────────────────────────────────────────────
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferWindowMemory

# ── Streamlit page config must be first Streamlit call ──────────────────────
st.set_page_config(page_title="UniBrain Assistant", layout="centered")

# ── Constants ───────────────────────────────────────────────────────────────
IMG_SIZE: int = 96
DEVICE = torch.device("cpu")
ROOT: Path = Path(__file__).resolve().parent
ASSETS: Path = ROOT / "assets"
PDF_PATH: Path = ROOT / "unibrain.pdf"
TXT_PATH: Path = ROOT / "extra_knowledge.txt"
VDB_PATH: Path = ROOT / "extra_knowledge.faiss"  # directory name for FAISS
NIIGZ_MIME = "application/gzip"

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ╭────────────────────── Utils: adjacency & plotting ───────────────────────╮
@st.cache_data
def load_adj(path: str) -> np.ndarray:
    """CPU‑safe load of a torch‑saved adjacency matrix → numpy array."""
    return torch.load(path, map_location="cpu").cpu().numpy()

def adj_to_graph_random(arr: np.ndarray, keep_ratio: float, seed: int = 42) -> nx.Graph:
    """Randomly retain keep_ratio∈(0,1] of 1‑edges in a binary adjacency matrix."""
    A = np.squeeze(arr)
    if A.ndim != 2:
        raise ValueError(f"Adj must be 2‑D, got shape {A.shape}")

    rng = np.random.default_rng(seed)
    idx_u = np.triu_indices_from(A, k=1)
    ones = np.flatnonzero(A[idx_u])  # indices within the 1‑D upper‑tri slice

    mask_u = np.zeros_like(idx_u[0], dtype=bool)
    if keep_ratio < 1.0:
        n_keep = max(1, int(len(ones) * keep_ratio)) if len(ones) else 0
        if n_keep:
            keep = rng.choice(ones, size=n_keep, replace=False)
            mask_u[keep] = True
    else:
        mask_u[ones] = True

    mask = np.zeros_like(A, dtype=bool)
    mask[idx_u] = mask_u
    mask = mask | mask.T
    return nx.from_numpy_array(mask, create_using=nx.Graph)

def show_adj_heatmap(arr: np.ndarray) -> None:
    st.write("### Heatmap")
    arr2d = np.squeeze(arr)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(arr2d, aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.046)
    st.pyplot(fig, clear_figure=True)

def show_adj_graph(arr: np.ndarray, key_prefix: str) -> None:
    st.write("### Network graph (X % of edges)")
    keep = (
        st.slider(
            "Keep top‑X % edges", 5, 100, 20, 5, key=f"{key_prefix}_keep"
        )
        / 100.0
    )
    G = adj_to_graph_random(arr, keep_ratio=keep)
    if G.number_of_edges() == 0:
        st.info("No edges retained at this percentage.")
        return
    pos = nx.spring_layout(G, seed=42, k=1 / np.sqrt(max(G.number_of_nodes(), 1)))
    fig, ax = plt.subplots(figsize=(5, 5))
    nx.draw_networkx_nodes(G, pos, node_size=20, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.25, width=1, ax=ax)
    ax.set_title(f"{G.number_of_nodes()} nodes  |  {G.number_of_edges()} edges")
    ax.axis("off")
    st.pyplot(fig, clear_figure=True)

# ╭────────────────────────── Model import (fallback) ───────────────────────╮
try:
    from model import UniBrain  # type: ignore
    MODEL_OK = True
except ImportError:  # minimal stub
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

# ╭──────────────────────────── Cached loaders ──────────────────────────────╮
@st.cache_resource(show_spinner="⏳ Loading templates …")
def load_templates():
    def load(name: str):
        arr = np.load(ASSETS / name).astype(np.float32)
        return torch.from_numpy(arr)[None, None].to(DEVICE)

    try:
        return tuple(load(p) for p in ("tpl_img.npy", "tpl_gm.npy", "tpl_aal.npy"))
    except Exception as e:
        st.error(f"Template error: {e}")
        return (None, None, None)

@st.cache_resource(show_spinner="⏳ Loading UniBrain …")
def load_model():
    if not MODEL_OK:
        st.warning("Using dummy model (import failed)")
        return None
    try:
        m = UniBrain(
            img_size=IMG_SIZE, ext_stage=3, reg_stage=2, if_pred_aal=True
        ).to(DEVICE)
        m.load_state_dict(torch.load(ASSETS / "unibrain.pth", map_location=DEVICE))
        m.eval()
        return m
    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None

REF_TPL, REF_GM, REF_AAL = load_templates()
MODEL = load_model()

# ╭──────────────────────────── Utility helpers ─────────────────────────────╮
@st.cache_data
def load_vol(path_str: str) -> np.ndarray:
    return nib.load(path_str).get_fdata()

def nii_to_torch(p: Path) -> torch.Tensor:
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.float32)
    if arr.shape != (IMG_SIZE,) * 3:
        arr = resize(
            arr, (IMG_SIZE,) * 3, order=1, preserve_range=True, anti_aliasing=True
        )
    return torch.from_numpy(arr)[None, None].to(DEVICE)

def save_tensor(t: torch.Tensor, out: Path) -> None:
    t = torch.nan_to_num(t).cpu().squeeze().float()
    sitk.WriteImage(sitk.GetImageFromArray(t.numpy()), str(out))

@st.cache_data
def slice_vol(vol: np.ndarray, z: int) -> np.ndarray:
    sl = vol[:, :, z].astype(np.float32)
    rng = sl.ptp() or 1
    return ((sl - sl.min()) / rng * 255).astype(np.uint8)

# ╭────────────────────────────── Knowledge RAG ─────────────────────────────╮
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
        st.error("No knowledge docs")
        return None
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150
    ).split_documents(docs)
    vdb = FAISS.from_documents( # FAISS stores document information as memory.
        chunks, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY) # OpenAIEmbeddings converts text to vector embeddings using OpenAI's API, which are then stored in the FAISS vector store for efficient similarity search.
    )
    # save_local expects a directory; VDB_PATH is used as a dir name
    vdb.save_local(str(VDB_PATH))
    return vdb

def ask_knowledge(q: str) -> str:
    vdb = vector_store()
    if vdb is None:
        return "Knowledge DB unavailable."

    retriever = vdb.as_retriever(search_kwargs={"k": 4})  # retrieve top 4 relevant chunks from the vector store based on the query
    docs = retriever.get_relevant_documents(q)  # 
    ctx = "\n\n".join(d.page_content for d in docs)

    prompt = ( 
        "You are UniBrain‑RAG assistant. Use the context only.\n\n"
        f"Context:\n{ctx}\n\n"
        f"Q: {q}"
    )
    return ChatOpenAI(
        openai_api_key=OPENAI_API_KEY, model_name="gpt-4o", temperature=0
    ).invoke(prompt).content

KNOWLEDGE_TOOL = Tool(
    name="ask_unibrain_paper",
    func=ask_knowledge,
    description="Answer UniBrain paper questions",
    return_direct=True,
)

# ╭──────────────────────────── Inference pipeline ──────────────────────────╮
CARD = Dict  # type alias for cards

def run_pipeline(inp: Path, work: Path) -> List[CARD]:
    if None in (MODEL, REF_TPL, REF_GM, REF_AAL):
        raise RuntimeError("Model/templates not ready")

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

    nii_dir, pt_dir = (work / "nii", work / "pt")
    nii_dir.mkdir(parents=True, exist_ok=True)
    pt_dir.mkdir(parents=True, exist_ok=True)

    cards: List[CARD] = []

    def nii_card(t: torch.Tensor, name: str, note: str = "") -> Path:
        out = nii_dir / f"{name}.nii.gz"
        save_tensor(t, out)
        cards.append(
            {"step": name.replace("_", " ").title(), "nifti_path": out, "explanation": note}
        )
        return out

    prob = torch.softmax(logits, 1)[0]
    cls, conf = int(prob.argmax()), float(prob.max())

    cls_name = {0: "Healthy", 1: "AD"}.get(cls, f"Class {cls}")
    torch.save(logits.cpu(), pt_dir / "logits.pt")
    cards.append(
        {
            "step": "Prediction",
            "metrics": {"class": cls, "probability": conf},
            "explanation": f"Predicted **{cls_name}** (p={conf:.3f})",
            "file_path": pt_dir / "logits.pt",
        }
    )

    nii_card(torch.argmax(am_mov, 1), "am_seg", "Anatomical mask")
    nii_card(torch.argmax(aal_mov, 1), "aal_seg", "AAL labels")
    nii_card(am_ref2, "am_ref2mov", "Template ANAT→input")
    nii_card(aal_ref2, "aal_ref2mov", "Template AAL→input")

    for i, (m, s) in enumerate(zip(masks, striped), 1):
        nii_card(m, f"mask{i}")
        nii_card(s, f"strip{i}")
    for i, w in enumerate(warped, 1):
        nii_card(w, f"warped{i}")

    for lbl, obj in {"theta": theta, "theta_inv": theta_i, "adj": adj}.items():
        p = pt_dir / f"{lbl}.pt"
        torch.save(obj, p)
        cards.append({"step": lbl, "file_path": p})

    return cards

# Tool to trigger pipeline from the agent

def _run_unibrain() -> str:
    if st.session_state.get("pipeline_done"):
        return "✅ Already done"
    try:
        st.session_state.cards = run_pipeline(
            Path(st.session_state.upload_path), Path(st.session_state.upload_path).parent
        )
        st.session_state.pipeline_done = True
        return "✅ Inference complete"
    except Exception as e:
        return f"🚨 Failure: {e}"

RUN_TOOL = Tool(
    name="run_unibrain_inference",
    description="Run/rerun UniBrain on uploaded file",
    func=lambda _: "❗ No file" if "upload_path" not in st.session_state else _run_unibrain(),
)

# ╭──────────────────────────── Agent initialization ────────────────────────╮
PROMPT_PATH = ROOT / "prompts" / "unibrain_system_prompt.md"
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8") if PROMPT_PATH.exists() else (
    "You are UniBrain assistant."
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)




agent = None
if OPENAI_API_KEY:
    try:
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY, model_name="gpt-4o-mini", temperature=0
        )
        memory = ConversationBufferWindowMemory(  # this part trunk instead of compact history. 
            memory_key="chat_history", return_messages=True
        )
        agent = initialize_agent(
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            tools=[RUN_TOOL, KNOWLEDGE_TOOL],
            llm=llm,
            prompt=prompt_template,  # keep original pattern
            memory=memory,
            max_iterations=5,
            verbose=False,
            handle_parsing_errors=True,
        )
    except Exception as e:
        st.error(f"Agent init failed: {e}")
        st.exception(e)

# ╭────────────────────────────── UI: viewers ───────────────────────────────╮

def viewer(nifti: Path, title: str, key: str, *, color: bool = False) -> None:
    vol = load_vol(str(nifti))
    z_max = vol.shape[2] - 1
    z = st.slider(f"{title} – slice", 0, z_max, z_max // 2, key=key)
    st.image(slice_vol(vol, z), caption=f"{title} (z={z})", use_container_width=True, clamp=True)



def show_input_preview(in_path: Path) -> None:
    st.subheader("Input preview")
    try:
        from input_preview import real_nifti_viewer
        real_nifti_viewer(in_path, "Input Brain Volume", "input_preview", color=False)
    except ImportError:
        viewer(in_path, "Input", "prev")

# ╭────────────────────────────── File handling ─────────────────────────────╮

def is_parcellation(step: str) -> bool:
    step_l = step.lower()
    return any(k in step_l for k in ("aal_seg", "aal_ref2mov"))


def handle_upload() -> Optional[Path]:
    """Return local Path of uploaded NIfTI, or None if nothing yet."""
    upload = st.file_uploader("Upload NIfTI (.nii/.nii.gz)", ["nii", "nii.gz"])
    if not upload:
        return None

    # If a new file is uploaded, make a fresh job dir and save it
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


def run_inference_button(in_path: Path) -> None:
    if st.button("Run UniBrain", disabled=st.session_state.pipeline_done):
        st.session_state.cards = run_pipeline(in_path, Path(st.session_state.work_dir))
        st.session_state.pipeline_done = True
        st.rerun()


def show_outputs() -> None:
    if not st.session_state.get("pipeline_done"):
        return

    st.header("Inference Outputs")
    for i, card in enumerate(st.session_state.cards):
        with st.expander(card["step"], expanded=False):
            st.write(card.get("explanation", ""))

            # ① Classification metrics
            if m := card.get("metrics"):
                c1, c2 = st.columns(2)
                c1.metric("Class", m["class"])
                c2.metric("Confidence", f"{m['probability']:.3f}")

            # ② NIfTI display with Real Data Slices
            p = card.get("nifti_path")
            if p and Path(p).exists():
                try:
                    from input_preview import real_nifti_viewer
                    real_nifti_viewer(Path(p), card["step"], f"out_{i}", color=is_parcellation(card["step"]))
                except ImportError:
                    viewer(Path(p), card["step"], f"out_{i}", color=is_parcellation(card["step"]))

                st.download_button(
                    "Download NIfTI",
                    Path(p).read_bytes(),
                    file_name=Path(p).name,
                    mime=NIIGZ_MIME,
                    key=f"dl_{i}",
                )

            # ②‑b Adjacency heatmap/graph toggle
            elif (fp := card.get("file_path")) and Path(fp).exists() and Path(fp).stem == "adj":
                arr_adj = load_adj(str(fp))
                view_mode = st.radio(
                    "Adjacency view", ["Heatmap", "Graph"], index=0, key=f"adj_view_{i}"
                )
                if view_mode == "Heatmap":
                    show_adj_heatmap(arr_adj)
                else:
                    show_adj_graph(arr_adj, key_prefix=f"adj_{i}")

                st.download_button(
                    "Download adj.pt",
                    Path(fp).read_bytes(),
                    file_name=Path(fp).name,
                    mime="application/octet-stream",
                    key=f"dl_adj_{i}",
                )

            # ③ Other binary outputs
            if (fp := card.get("file_path")) and Path(fp).exists():
                st.download_button(
                    "Download data",
                    Path(fp).read_bytes(),
                    file_name=Path(fp).name,
                    mime="application/octet-stream",
                    key=f"dlf_{i}",
                )


# main program starts here

# ╭────────────────────────────── Main app ──────────────────────────────────╮
st.title("🧠 UniBrain Assistant")
for k, v in {"messages": [], "pipeline_done": False}.items():
    st.session_state.setdefault(k, v)

in_path = handle_upload()
if in_path:
    show_input_preview(in_path) # visualzation image with notes
    run_inference_button(in_path) # inference button
    show_outputs() # outputs with inference




# ╭────────────────────────────── Chat area ─────────────────────────────────╮
st.divider()
st.subheader("Chat with UniBrain Assistant")
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


# if agent is defined 
if agent:
    # 1st if: If the user submits a new chat input, append it to the chat history and rerun the app.
    if q := st.chat_input("Ask about UniBrain …"):
        st.session_state.messages.append({"role": "user", "content": q})
        st.rerun() # add history along with current question to agent input

    # 2nd if: If the last message is from the user, process it with the agent and display the assistant's response.
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user": # expalin: check if there are any messages and if the last one is from the user
        q = st.session_state.messages[-1]["content"]
        ctx = f"File: {st.session_state.upload_path}" if "upload_path" in st.session_state else "No file loaded."

        # Rebuild a compact chat history
        hist: List[HumanMessage | AIMessage] = []
        msgs = st.session_state.messages[:-1]
        for i in range(0, len(msgs) - 1, 2):
            if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
                hist += [
                    HumanMessage(content=msgs[i]["content"]),
                    AIMessage(content=msgs[i + 1]["content"]),
                ]

        full_history: List[SystemMessage | HumanMessage | AIMessage] = [
            SystemMessage(content=SYSTEM_PROMPT)
        ] + hist
        full_history.append(HumanMessage(content=f"{q}\n\nContext: {ctx}"))
        # Insert the assistant's reply into the chat history before displaying
        with st.chat_message("assistant"), st.spinner("Thinking…"):
            try:
                result = agent.invoke({"input": full_history})  # keep original behavior
                out = result.get("output", str(result))
            except Exception as e:
                out = f"Agent error: {e}"
            st.session_state.messages.append({"role": "assistant", "content": out})
            st.rerun()
else:
    st.info("Agent disabled (no API key)")