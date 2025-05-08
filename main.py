"""
UniBrain Streamlit App — concise, maintainable refactor
======================================================
(Outputs now appear inside collapsible **expanders** so the page stays tidy. Click to reveal/download.)

Author: Songlin Zhao (2025‑05‑01)
"""
from __future__ import annotations

# ── Std / 3rd‑party ─────────────────────────────────────────────────────────
import os, uuid, time, warnings
from pathlib import Path

import numpy as np, torch, streamlit as st, SimpleITK as sitk, nibabel as nib
from PIL import Image
from skimage.transform import resize
import plotly.graph_objects as go

# ── LangChain & tools ───────────────────────────────────────────────────────
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# ── Constants ───────────────────────────────────────────────────────────────
IMG_SIZE = 96
DEVICE   = torch.device("cpu")
ROOT     = Path(__file__).resolve().parent
ASSETS   = ROOT / "assets"
PDF_PATH = ROOT / "unibrain.pdf"
TXT_PATH = ROOT / "extra_knowledge.txt"
VDB_PATH = ROOT / "extra_knowledge.faiss"

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ------------------------------------------------------------------#
# Session‑state defaults                                            #
# ------------------------------------------------------------------#
DEFAULT_STEPS = [
    "Extraction", "Registration", "Segmentation",
    "Parcellation", "Network", "Classification",
]
if "selected_steps" not in st.session_state:
    st.session_state.selected_steps = DEFAULT_STEPS.copy()
if "pipeline_done" not in st.session_state:
    st.session_state.pipeline_done = False
if "messages" not in st.session_state:
    st.session_state.messages = []


# cross-version rerun helper
def _rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

# ── utils_adj.py (或直接放到现有文件上方) ─────────────────────────────
import torch, networkx as nx
@st.cache_data
def load_adj(path: str) -> np.ndarray:
    """CPU-safe加载 torch 存的邻接矩阵并返回 numpy array"""
    return torch.load(path, map_location="cpu").cpu().numpy()

def adj_to_graph(arr: np.ndarray, thresh: float) -> nx.Graph:
    """把邻接矩阵转成无向图；自动 squeeze 到 2D"""
    arr2d = np.squeeze(arr)                # (1,N,N) -> (N,N)
    if arr2d.ndim != 2:
        raise ValueError(f"Adj must be 2-D, got shape {arr.shape}")
    mask = arr2d > thresh
    return nx.from_numpy_array(mask, create_using=nx.Graph)


def show_adj_heatmap(arr: np.ndarray):
    st.write("### Heatmap")
    arr2d = np.squeeze(arr)           # ← 去掉多余维度
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(arr2d, aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.046)
    st.pyplot(fig, clear_figure=True)

import numpy as np
import networkx as nx

def adj_to_graph_random(arr: np.ndarray, keep_ratio: float, seed: int = 42) -> nx.Graph:
    """
    Randomly retain keep_ratio ∈ (0,1] of the 1-edges in a binary adjacency matrix.
    """
    A = np.squeeze(arr)                 # (1,N,N) → (N,N)
    if A.ndim != 2:
        raise ValueError(f"Adj must be 2-D, got shape {arr.shape}")

    # --- pick edges from upper-triangular to avoid duplicates --------------
    rng   = np.random.default_rng(seed)
    idx_u = np.triu_indices_from(A, k=1)          # upper-tri indices
    ones  = np.flatnonzero(A[idx_u])              # positions of 1-edges
    if keep_ratio < 1.0:
        n_keep = max(1, int(len(ones) * keep_ratio))
        keep_idx = rng.choice(ones, size=n_keep, replace=False)
        mask_u   = np.zeros_like(idx_u[0], dtype=bool)
        mask_u[keep_idx] = True
    else:
        mask_u = np.zeros_like(idx_u[0], dtype=bool)
        mask_u[ones] = True

    # --- symmetric binary mask --------------------------------------------
    mask = np.zeros_like(A, dtype=bool)
    mask[idx_u] = mask_u
    mask = mask | mask.T               # make symmetric

    return nx.from_numpy_array(mask, create_using=nx.Graph)


def show_adj_graph(arr: np.ndarray, key_prefix: str):
    st.write("### Network graph ( X % of edges)")

    keep = st.slider(
        "Keep top-X % edges",
        5, 100, 20, 5,
        key=f"{key_prefix}_keep"
    ) / 100.0

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





# ── Model import (dummy fallback) ───────────────────────────────────────────
try:
    from model import UniBrain  # type: ignore
    MODEL_OK = True
except ImportError:  # minimal stub
    class UniBrain(torch.nn.Module):
        def forward(self, *a, **k):
            b = a[1].shape[0]; shp = (b,1,IMG_SIZE,IMG_SIZE,IMG_SIZE)
            rand = lambda: torch.rand(shp, device=a[1].device)
            eye  = lambda: torch.eye(4, device=a[1].device).unsqueeze(0)
            return ([rand()]*3, [rand()]*3, [rand()]*3, [eye()]*2, [eye()]*2,
                    rand(), rand(), rand(), rand(), torch.rand(b,10,10), torch.rand(b,2))
    MODEL_OK = False

# ╭────────────────── Cached loaders ───────────────────────────────────────╮
@st.cache_resource(show_spinner="⏳ Loading templates …")
def load_templates():
    def load(name: str):
        arr = np.load(ASSETS / name).astype(np.float32)
        return torch.from_numpy(arr)[None, None].to(DEVICE)
    try:
        return tuple(load(p) for p in ("tpl_img.npy","tpl_gm.npy","tpl_aal.npy"))
    except Exception as e:
        st.error(f"Template error: {e}"); return (None,None,None)

@st.cache_resource(show_spinner="⏳ Loading UniBrain …")
def load_model():
    if not MODEL_OK:
        st.warning("Using dummy model (import failed)"); return None
    try:
        m = UniBrain(img_size=IMG_SIZE, ext_stage=3, reg_stage=2, if_pred_aal=True).to(DEVICE)
        m.load_state_dict(torch.load(ASSETS/"unibrain.pth", map_location=DEVICE)); m.eval(); return m
    except Exception as e:
        st.error(f"Model load failed: {e}"); return None

REF_TPL, REF_GM, REF_AAL = load_templates(); MODEL = load_model()

# ╭────────────────── Utility helpers ───────────────────────────────────────╮
NIIGZ = "application/gzip"

def nii_to_torch(p: Path):
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.float32)
    if arr.shape != (IMG_SIZE,)*3:
        arr = resize(arr,(IMG_SIZE,)*3,order=1,preserve_range=True,anti_aliasing=True)
    return torch.from_numpy(arr)[None,None].to(DEVICE)

def save_tensor(t: torch.Tensor, out: Path):
    t = torch.nan_to_num(t).cpu().squeeze().float()
    sitk.WriteImage(sitk.GetImageFromArray(t.numpy()), str(out))

def slice_png(t: torch.Tensor, out: Path):
    arr = t.squeeze().cpu().numpy()[t.shape[-1]//2]
    norm=((arr-arr.min())/(arr.ptp()+1e-6)*255).astype(np.uint8)
    Image.fromarray(norm).save(out)

# ╭────────────────── Knowledge RAG ─────────────────────────────────────────╮
OPENAI_API_KEY = "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"

# ── colour utils ──────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

def _make_cmap(n: int) -> np.ndarray:
    """
    生成 (n,3) float32 LUT，索引 0 为黑，其余按 tab20/20b/20c 轮流。
    """
    base = (plt.get_cmap('tab20').colors +
            plt.get_cmap('tab20b').colors +
            plt.get_cmap('tab20c').colors) * 50        # 上千色足够
    return np.vstack([[0, 0, 0], np.array(base[:n-1])]).astype(np.float32)

def roi_slice_to_rgb(label2d: np.ndarray) -> np.ndarray:
    """
    label2d: H×W int32, ROI id；背景=0
    返回  H×W×3 uint8 彩色图（背景黑）
    """
    max_id = int(label2d.max())
    lut = (_make_cmap(max_id + 1) * 255).astype(np.uint8)
    return lut[label2d]






@st.cache_resource(show_spinner="🔍 Building vector store …")
def vector_store():
    if not OPENAI_API_KEY:
        st.error("Missing OpenAI key"); return None
    docs=[]
    if PDF_PATH.exists(): docs+=PyPDFLoader(str(PDF_PATH)).load()
    if TXT_PATH.exists(): docs+=TextLoader(str(TXT_PATH),encoding="utf-8").load()
    if not docs: st.error("No knowledge docs"); return None
    chunks=RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=150).split_documents(docs)
    vdb=FAISS.from_documents(chunks,OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)); vdb.save_local(str(VDB_PATH)); return vdb
def ask_knowledge(q: str) -> str:
    vdb = vector_store()
    if vdb is None:
        return "Knowledge DB unavailable."

    # 正确调用 as_retriever
    retriever = vdb.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(q)
    ctx = "\n\n".join(d.page_content for d in docs)

    prompt = (
        "You are UniBrain-RAG assistant. Use the context only.\n\n"
        f"Context:\n{ctx}\n\n"
        f"Q: {q}"
    )
    return ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-4o",
        temperature=0
    ).invoke(prompt).content



KNOWLEDGE_TOOL=Tool(name="ask_unibrain_paper",func=ask_knowledge,description="Answer UniBrain paper questions",return_direct=True)

# ╭────────────────── Inference pipeline ────────────────────────────────────╮
CARD=dict

from pathlib import Path
import time
import torch
import streamlit as st

# CARD alias already defined elsewhere
# MODEL, REF_TPL, REF_GM, REF_AAL, nii_to_torch, save_tensor all assumed imported

def run_pipeline(
    inp: Path,
    work: Path,
    steps: list[str] | None = None,    # <-- new param
) -> list[CARD]:
    if None in (MODEL, REF_TPL, REF_GM, REF_AAL):
        raise RuntimeError("Model/templates not ready")

    # default: run all steps
    all_steps = ["Extraction", "Registration", "Segmentation", "Parcellation", "Network", "Classification"]
    if steps is None:
        steps = all_steps

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
    pt_dir.mkdir(exist_ok=True)

    cards: list[CARD] = []

    def nii_card(tensor, name, note=""):
        out = nii_dir / f"{name}.nii.gz"
        save_tensor(tensor, out)
        cards.append({
            "step": name.replace("_", " ").title(),
            "nifti_path": out,
            "explanation": note,
        })
        return out

    # Classification
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

    # Segmentation / Parcellation
    if "Segmentation" in steps:
        nii_card(torch.argmax(am_mov, 1), "am_seg", "Anatomical mask")
    if "Parcellation" in steps:
        nii_card(torch.argmax(aal_mov, 1), "aal_seg", "AAL labels")

    # Registration details
    if "Registration" in steps:
        nii_card(am_ref2, "am_ref2mov", "Template ANAT→input")
        nii_card(aal_ref2, "aal_ref2mov", "Template AAL→input")
        for i, w in enumerate(warped, 1):
            nii_card(w, f"warped{i}")

    # Extraction masks
    if "Extraction" in steps:
        for i, (m, s) in enumerate(zip(masks, striped), 1):
            nii_card(m, f"mask{i}")
            nii_card(s, f"strip{i}")

    # Brain network (adjacency)
    if "Network" in steps:
        adj_path = pt_dir / "adj.pt"
        torch.save(adj, adj_path)
        cards.append({
            "step": "Network",
            "file_path": adj_path,
            "explanation": "Adjacency matrix",
        })

    return cards


# def run_pipeline(inp:Path,work:Path)->list[CARD]:
#     if None in (MODEL,REF_TPL,REF_GM,REF_AAL): raise RuntimeError("Model/templates not ready")
#     mov=nii_to_torch(inp)
#     st.info("Running UniBrain …"); t0=time.time()
#     with torch.no_grad():
#         (striped,masks,warped,theta,theta_i,am_mov,am_ref2,aal_mov,aal_ref2,adj,logits)=MODEL(REF_TPL,mov,REF_GM,REF_AAL,if_train=False)
#     st.success(f"Done in {time.time()-t0:.1f}s")

#     nii_dir,pt_dir=(work/"nii",work/"pt"); nii_dir.mkdir(parents=True,exist_ok=True); pt_dir.mkdir(exist_ok=True)
#     cards: list[CARD]=[]
#     def nii_card(t,name,note=""):
#         out=nii_dir/f"{name}.nii.gz"; save_tensor(t,out)
#         cards.append({"step":name.replace("_"," ").title(),"nifti_path":out,"explanation":note}); return out

#     prob=torch.softmax(logits,1)[0]; cls,conf=int(prob.argmax()),float(prob.max())
#     torch.save(logits.cpu(),pt_dir/"logits.pt")
#     cards.append({"step":"Classification","metrics":{"class":cls,"probability":conf},"explanation":f"Predicted **{cls}** (p={conf:.3f})","file_path":pt_dir/"logits.pt"})

#     nii_card(torch.argmax(am_mov,1),"am_seg","Anatomical mask")
#     nii_card(torch.argmax(aal_mov,1),"aal_seg","AAL labels")
#     nii_card(am_ref2,"am_ref2mov","Template ANAT→input")
#     nii_card(aal_ref2,"aal_ref2mov","Template AAL→input")

#     for i,(m,s) in enumerate(zip(masks,striped),1):
#         nii_card(m,f"mask{i}"); nii_card(s,f"strip{i}")
#     for i,w in enumerate(warped,1):
#         nii_card(w,f"warped{i}")

#     for lbl,obj in {"theta":theta,"theta_inv":theta_i,"adj":adj}.items():
#         p=pt_dir/f"{lbl}.pt"; torch.save(obj,p); cards.append({"step":lbl,"file_path":p})

#     return cards

RUN_TOOL=Tool(name="run_unibrain_inference",description="Run/rerun UniBrain on uploaded file",func=lambda _: "❗ No file" if "upload_path" not in st.session_state else _run_unibrain())

def _run_unibrain():
    if st.session_state.get("pipeline_done"): return "✅ Already done"
    try:
        st.session_state.cards=run_pipeline(Path(st.session_state.upload_path),Path(st.session_state.upload_path).parent)
        st.session_state.pipeline_done=True; return "✅ Inference complete"
    except Exception as e: return f"🚨 Failure: {e}"

# # ╭────────────────── Agent init ─────────────────────────────────────────────╮
# agent=None
# if OPENAI_API_KEY:
#     try:
#         agent=initialize_agent(agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,tools=[RUN_TOOL,KNOWLEDGE_TOOL],llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY,model_name="gpt-4o",temperature=0),max_iterations=5,verbose=False)
#     except Exception as e: st.error(f"Agent init failed: {e}")
from pathlib import Path
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferWindowMemory

# ——— 1. 读取外部 Prompt 文件 —————————————————————————————
PROMPT_PATH = ROOT / "prompts" / "unibrain_system_prompt.md"
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")

# ——— 2. 构造带 system + human 的 PromptTemplate —————————————————
prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template("{input}")
])

# ——— 3. 初始化 Agent 时注入 prompt 和 memory —————————————————
agent = None
if OPENAI_API_KEY:
    try:
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name="gpt-4o-mini",
            temperature=0
        )
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True
        )
        agent = initialize_agent(
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            tools=[RUN_TOOL, KNOWLEDGE_TOOL],
            llm=llm,
            prompt=prompt_template,           
            memory=memory,                    
            max_iterations=5,
            verbose=False,
            handle_parsing_errors=True
        )
        # st.write("🔍 Final agent.prompt:", agent.prompt or agent.agent.prompt_template)

    except Exception as e:
        st.error(f"Agent init failed: {e}")
        st.exception(e)





# ╭────────────────── UI helpers ─────────────────────────────────────────────╮
@st.cache_data
def slice_vol(vol,z): sl=vol[:,:,z].astype(np.float32); rng=sl.ptp() or 1; return ((sl-sl.min())/rng*255).astype(np.uint8)

def viewer(nifti: Path, title: str, key: str, *, color: bool = False):
    vol = nib.load(str(nifti)).get_fdata()
    z_max = vol.shape[2] - 1
    z = st.slider(f"{title} – slice", 0, z_max, z_max // 2, key=key)

    slice2d = vol[:, :, z]
    if color:  # AAL segmentation ➜ 彩色
        lab = slice2d.astype(np.int32)
        lab[lab < 1] = 0          # 强制背景=0
        st.image(
            roi_slice_to_rgb(lab),   # LUT[0] 已是纯黑
            caption=f"{title} (z={z})",
            use_container_width=True,
            clamp=True
        )

    else:
        st.image(slice_vol(vol, z), caption=f"{title} (z={z})",
                 use_container_width=True, clamp=True)


# ── 1. 重写 vol_fig：只接受参数，不调 st ────────────────────────────
def vol_fig(arr: np.ndarray, ds: int, colourscale: str) -> go.Figure:
    # 降采样
    arr_ds = arr[::ds, ::ds, ::ds]
    # 坐标
    x, y, z = np.mgrid[
        :arr_ds.shape[0],
        :arr_ds.shape[1],
        :arr_ds.shape[2],
    ]
    # 阈值
    nonmin = arr_ds[arr_ds > arr_ds.min()]
    imin, imax = (np.percentile(nonmin, (5, 98))
                  if nonmin.size else (0, 1))

    fig = go.Figure(
        go.Volume(
            x=x.ravel(),
            y=y.ravel(),
            z=z.ravel(),
            value=arr_ds.ravel(),
            isomin=imin,
            isomax=imax,
            opacity=0.2,
            opacityscale=[
                [0, 0],
                [0.1, 0.05],
                [0.3, 0.1],
                [0.6, 0.3],
                [0.8, 0.5],
                [1, 0.7],
            ],
            surface_count=17,
            colorscale=colourscale,
        )
    )
    fig.update_layout(
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=30, b=0),
        height=500,
        title=dict(text="3-D Volume Preview", x=0.5),
    )
    return fig


# ── 2. 更新输入预览：给 input 也加唯一 key ────────────────────────
def show_input_preview(in_path: Path) -> None:
    st.subheader("Input preview")
    viewer(in_path, "Input", "prev")  # 保持原来的 2D viewer

    # 3D 渲染参数：唯一 key
    ds_key   = "quality_input"
    cmap_key = "cmap_input"
    ds_in = st.radio(
        "Quality (↓ faster)",
        [4, 3, 2, 1],
        index=0,
        horizontal=True,
        key=ds_key,
    )
    cmap_in = st.selectbox(
        "Colormap",
        ["Greys", "Viridis", "Hot", "Rainbow", "Jet"],
        index=1,
        key=cmap_key,
    )

    arr = nib.load(str(in_path)).get_fdata()
    fig = vol_fig(arr, ds=ds_in, colourscale=cmap_in)
    st.plotly_chart(fig, use_container_width=True, key="vol_input")




# def vol_fig(arr:np.ndarray):
#     ds=st.radio("Quality (↓ faster)",[4,3,2,1],index=1,horizontal=True)
#     arr=arr[::ds,::ds,::ds]
#     x,y,z=np.mgrid[:arr.shape[0],:arr.shape[1],:arr.shape[2]]
#     nonmin=arr[arr>arr.min()]; imin,imax=np.percentile(nonmin,(5,98)) if nonmin.size else (0,1)
#     return go.Figure(go.Volume(x=x.ravel(),y=y.ravel(),z=z.ravel(),value=arr.ravel(),isomin=imin,isomax=imax,opacity=0.2,opacityscale=[[0,0],[0.1,0.05],[0.3,0.1],[0.6,0.3],[0.8,0.5],[1,0.7]],surface_count=17,colorscale="Greys"))

# ╭────────────────── Streamlit main ────────────────────────────────────────╮
st.set_page_config(page_title="UniBrain Agent",layout="centered", initial_sidebar_state="collapsed")

st.title("🧠 UniBrain Assistant")
for k,v in {"messages":[],"pipeline_done":False}.items(): st.session_state.setdefault(k,v)

# ── constants you already have ──────────────────────────────────────────────
NIIGZ = "application/gzip"

# ---------------------------------------------------------------------------#
# 0. 工具：判定是否为 parcellation（决定彩色 / 灰阶）                        #
# ---------------------------------------------------------------------------#
def is_parcellation(step: str) -> bool:
    step = step.lower()
    return any(k in step for k in ("aal_seg", "aal_ref2mov"))

# ---------------------------------------------------------------------------#
# 1. 文件上传 & 落盘                                                          #
# ---------------------------------------------------------------------------#
def handle_upload() -> Path | None:
    """Return local Path of uploaded NIfTI, or None if nothing yet."""
    upload = st.file_uploader("Upload NIfTI (.nii/.nii.gz)", ["nii", "nii.gz"])
    if not upload:
        # st.info("Upload a NIfTI file to begin.")
        return None

    # 新文件 → 生成 job 目录并保存
    if st.session_state.get("fname") != upload.name:
        job_id  = uuid.uuid4().hex[:8]
        workdir = Path("uploads") / job_id
        workdir.mkdir(parents=True, exist_ok=True)

        up_path = workdir / upload.name
        up_path.write_bytes(upload.getbuffer())

        st.session_state.update(
            fname          = upload.name,
            upload_path    = str(up_path),
            work_dir       = str(workdir),
            pipeline_done  = False,
            cards          = []
        )
        st.success(f"Saved → {up_path}")

    return Path(st.session_state.upload_path)

# ---------------------------------------------------------------------------#
# 2. 输入预览（中间切片 + 3-D 体渲染）                                        #
# ---------------------------------------------------------------------------#
def show_input_preview(in_path: Path) -> None:
    st.subheader("Input preview")
    viewer(in_path, "Input", "prev")                      # 2-D slice viewer
    vol = nib.load(str(in_path)).get_fdata()
    # st.plotly_chart(vol_fig(vol), use_container_width=True)
    st.plotly_chart(vol_fig(vol, ds=2, colourscale="Greys"), use_container_width=True)



# ---------------------------------------------------------------------------#
# 3. “Run UniBrain” 按钮                                                      #
# ---------------------------------------------------------------------------#
def run_inference_button(in_path: Path) -> None:
    if st.button("Run UniBrain", disabled=st.session_state.pipeline_done, key="run_unibrain"):
        st.session_state.cards = run_pipeline(
            in_path, Path(st.session_state.work_dir)
        )
        st.session_state.pipeline_done = True
        st.rerun()


# 1) Let user select which steps to run
all_steps = ["Extraction", "Registration", "Segmentation",
             "Parcellation", "Network", "Classification"]

# sidebar default close

with st.sidebar.expander("⚙️ Pipeline steps", expanded=False):
    st.write("Select which stages to run:")
    st.session_state.selected_steps = st.multiselect(
        "Stages",
        DEFAULT_STEPS,
        default=st.session_state.selected_steps,
        key="steps_select",
    )

selected_steps = st.session_state.selected_steps

# # 2) Run button
# if st.sidebar.button("Run UniBrain", disabled=st.session_state.pipeline_done):
#     # call with selected_steps
#     in_path = Path(st.session_state.upload_path)
#     work_dir = in_path.parent
#     st.session_state.cards = run_pipeline(in_path, work_dir, selected_steps)
#     st.session_state.pipeline_done = True
#     st.rerun()

# ───────────────── Sidebar run button ─────────────────────────
if st.sidebar.button(
        "Run UniBrain",
        disabled=st.session_state.pipeline_done,
        key="sidebar_run"):
    if "upload_path" not in st.session_state:
        st.error("❗ Please upload a file first.")
    else:
        in_path  = Path(st.session_state.upload_path)
        work_dir = in_path.parent

        st.session_state.cards = run_pipeline(
            in_path,
            work_dir,
            st.session_state.selected_steps        # ← pass steps
        )
        st.session_state.pipeline_done = True

        # 1️⃣ append friendly summary *before* rerun
        summary = "✅ Finished running: " + ", ".join(st.session_state.selected_steps)
        st.session_state.messages.append({"role": "assistant", "content": summary})

        _rerun()   # ← do this last



# ---------------------------------------------------------------------------#
# 4. 结果展示：折叠卡片 + 下载                                                #
# ---------------------------------------------------------------------------#
def show_outputs() -> None:
    if not st.session_state.get("pipeline_done"):
        return

    st.header("Inference Outputs")
    for i, card in enumerate(st.session_state.cards):
        with st.expander(card["step"], expanded=False):
            st.write(card.get("explanation", ""))

            # ── ① 分类指标 ───────────────────────────────────────────────
            if m := card.get("metrics"):
                c1, c2 = st.columns(2)
                c1.metric("Class",       m["class"])
                c2.metric("Confidence", f"{m['probability']:.3f}")

            # ── ② 如果有 NIfTI，就给个显示模式切换 ──────────────────────

            if (p := card.get("nifti_path")) and Path(p).exists():
                arr = nib.load(str(p)).get_fdata()

                view_mode = st.radio(
                    "Display mode",
                    ["2D slice", "3D volume"],
                    index=0,
                    key=f"view_{i}",
                )

                if view_mode == "2D slice":
                    viewer(
                        Path(p),
                        card["step"],
                        f"out_{i}",
                        color=is_parcellation(card["step"]),
                    )
                else:
                    # 先选质量下采样
                    ds_i = st.radio(
                        "Quality (↓ faster)",
                        [4, 3, 2, 1],
                        index=1,
                        horizontal=True,
                        key=f"quality_{i}",
                    )
                    # 再选调色板
                    cmap_i = st.selectbox(
                        "Colormap",
                        ["Greys", "Viridis", "Hot", "Rainbow", "Jet"],
                        index=1,
                        key=f"cmap_{i}",
                    )
                    fig = vol_fig(arr, ds=ds_i, colourscale=cmap_i)
                    st.plotly_chart(fig, use_container_width=True, key=f"vol_{i}")

                # 下载按钮
                st.download_button(
                    "Download NIfTI",
                    Path(p).read_bytes(),
                    file_name=Path(p).name,
                    mime=NIIGZ,
                    key=f"dl_{i}"
                )
            # ── ②-b  若是 adj.pt 就显示 Heatmap / Graph ──────────────────
            elif (fp := card.get("file_path")) and "adj" in fp.stem:
                arr = load_adj(str(fp))
                view_mode = st.radio(
                    "Adjacency view",
                    ["Heatmap", "Graph"],
                    index=0,
                    key=f"adj_view_{i}"
                )
                if view_mode == "Heatmap":
                    show_adj_heatmap(arr)
                else:
                    show_adj_graph(arr, key_prefix=f"adj_{i}")

                # 下载按钮
                st.download_button(
                    "Download adj.pt",
                    Path(fp).read_bytes(),
                    file_name=Path(fp).name,
                    mime="application/octet-stream",
                    key=f"dl_adj_{i}"
                )

            # ── ③ 其它二进制输出 ───────────────────────────────────────
            if (fp := card.get("file_path")) and Path(fp).exists():
                st.download_button(
                    "Download data",
                    Path(fp).read_bytes(),
                    file_name=Path(fp).name,
                    mime="application/octet-stream",
                    key=f"dlf_{i}"
                )

            

CMD_SYS_PROMPT = """
You are a controller for the UniBrain Streamlit app. 
Your ONLY job is to examine the user's sentence and decide
• whether it requests skipping, enabling, or resetting pipeline steps.
Valid step names: Extraction, Registration, Segmentation, Parcellation, Network, Classification.

Treat these patterns all as “skip <Step>” commands:
  - “skip segmentation”
  - “without segmentation”
  - “no segmentation”
  - “preprocessing without segmentation”
  - “do the preprocessing without segmentation”
  - “run pipeline without segmentation”

OUTPUT RULES:
1. If the user wants to skip a step, output EXACT JSON: {"action":"skip","step":"<Step>"}
2. If the user wants to enable a step, output EXACT JSON: {"action":"enable","step":"<Step>"}
3. If the user wants to reset everything, output EXACT JSON: {"action":"reset"}
4. Otherwise output the literal word: none

Do NOT wrap JSON in markdown; no extra text.
""".strip()



from openai import OpenAI
import re, json, streamlit as st
from typing import Optional, Tuple

client = OpenAI(api_key=OPENAI_API_KEY)


DEFAULT_STEPS = [
    "Extraction","Registration","Segmentation",
    "Parcellation","Network","Classification"
]

def regex_detect(cmd: str) -> Optional[Tuple[str,str]]:
    cmd = cmd.lower().strip()
    # reset
    if "reset steps" in cmd or "reset pipeline" in cmd:
        return ("reset","")
    # skip patterns
    m = re.match(r"(skip|without|no)\s+(\w+)", cmd)
    if m:
        action, step = m.groups()
        if action in ("without","no"):
            action = "skip"
        return (action, step.capitalize())
    # enable
    m = re.match(r"(enable|turn on)\s+(\w+)", cmd)
    if m:
        action, step = m.groups()
        return ("enable", step.capitalize())
    return None

def llm_detect_command(text: str) -> Optional[Tuple[str,str]]:
    try:
        resp = client.chat.completions.create(
            model       = "gpt-3.5-turbo-0125",
            messages    = [
                {"role":"system","content":CMD_SYS_PROMPT},
                {"role":"user",  "content":text},
            ],
            max_tokens  = 20,
            temperature = 0,
        )
        out = resp.choices[0].message.content.strip()
        if out == "none":
            return None
        data = json.loads(out)
        action = data.get("action")
        if action == "reset":
            return ("reset","")
        if action in ("skip","enable") and data.get("step"):
            return (action, data["step"].capitalize())
    except Exception as e:
        st.warning(f"Cmd-parser LLM error: {e}")
    return None

def detect_command(text: str) -> Optional[Tuple[str,str]]:
    # 1) try regex
    cmd = regex_detect(text)
    if cmd:
        return cmd
    # 2) fallback to LLM
    return llm_detect_command(text)




# ---------------------------------------------------------------------------#
# 🏃 主流程：一步一函数                                                       #
# ---------------------------------------------------------------------------#
in_path = handle_upload()
if in_path:
    show_input_preview(in_path)
    # run_inference_button(in_path)
    show_outputs()











# ╭────────────────── Chat area ─────────────────────────────────────────╮
st.divider()
st.subheader("Chat with UniBrain Assistant")

# 0) Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# 1) Show single input box only when no pending reply
waiting_reply = (
    st.session_state.messages
    and st.session_state.messages[-1]["role"] == "user"
)
if not waiting_reply and agent:
    user_text = st.chat_input(
        "Ask about UniBrain …",
        key="main_chat_input"
    )
else:
    user_text = None

# 2) Detect & handle commands vs normal chat
if user_text:
    cmd = detect_command(user_text)
    # st.write(f"🔍 Debug: detect_command -> {cmd}")
    if cmd:
        action, step = cmd
        # ── Update the selected_steps ─────────────────────────────
        if action == "reset":
            st.session_state.selected_steps = DEFAULT_STEPS.copy()
            ack = "🔄 Steps reset to all enabled."
        elif action == "skip":
            st.session_state.selected_steps = [
                s for s in st.session_state.selected_steps if s != step
            ]
            ack = f"⏭️ **{step}** skipped."
        else:  # enable
            if step not in st.session_state.selected_steps:
                st.session_state.selected_steps.append(step)
            ack = f"✅ **{step}** enabled."

        # ── Immediately re-run the pipeline ─────────────────────────
        st.session_state.pipeline_done = False
        # you will need `Path` imported at top: from pathlib import Path
        in_path  = Path(st.session_state.upload_path)
        work_dir = in_path.parent
        st.session_state.cards = run_pipeline(
            in_path, work_dir, st.session_state.selected_steps
        )
        st.session_state.pipeline_done = True

        # ── Echo back to the chat UI ───────────────────────────────
        st.session_state.messages.append({"role":"user",      "content":user_text})
        st.session_state.messages.append({"role":"assistant", "content":ack})


        _rerun()    
    else:
        # normal chat → queue LLM, *do not* add summary yet
        st.session_state.messages.append({"role": "user", "content": user_text})
        _rerun()


# 3) If waiting_reply, call agent exactly once
if waiting_reply and agent:
    question = st.session_state.messages[-1]["content"]
    ctx      = f"File: {st.session_state.upload_path}" if "upload_path" in st.session_state else "No file."

    # build history for agent
    hist = []
    for m in st.session_state.messages[:-1]:
        if m["role"] == "user":
            hist.append(HumanMessage(content=m["content"]))
        else:
            hist.append(AIMessage   (content=m["content"]))

    full_history = [SystemMessage(content=SYSTEM_PROMPT)] + hist
    full_history.append(HumanMessage(content=f"{question}\n\nContext: {ctx}"))

    with st.chat_message("assistant"), st.spinner("Thinking…"):
        try:
            res = agent.invoke({"input": full_history})
            answer = res.get("output", str(res))
        except Exception as e:
            answer = f"Agent error: {e}"

        st.session_state.messages.append({"role":"assistant","content":answer})
        _rerun()

elif not agent:
    st.info("Agent disabled (no API key)")
