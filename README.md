# 🧠 UniBrain‑Assistant


**UniBrain‑Assistant** is an open-source, browser-based platform that integrates end-to-end deep learning into a fully conversational workflow for structural brain MRI analysis.

### :rocket: Try Me - One Click, Done Quick!

You can drop in a NIfTI file, watch every preprocessing step unfold in real time, explore the resulting connectome interactively, and ask questions in plain English or any natural languages — all without leaving your web browser.

It pairs Streamlit's reactive UI with LangChain's tool‑calling so you can **see**, **tweak**, and **interrogate** each stage of the pipeline:

* skull‑stripping → affine registration → tissue segmentation → AAL parcellation → graph construction → disease classification
* fully **interactive**: 3-view NIfTI slice viewer (Axial / Sagittal / Coronal) with cross-linked navigation, heat‑map / graph visualisations, one‑click downloads
* **pipeline orchestration by natural‑language** – e.g. `run the pipeline`, `what does AAL parcellation do?`
* **RAG‑powered Q & A** over both your outputs **and** the UniBrain paper itself

### ⏳ We are working hard to enhance the tool, and a new version will be released soon.

---

## ✨ Key Features

| UI / UX                                     | Details                                                                                                     |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Drag‑&‑drop NIfTI** (`.nii` / `.nii.gz`)  | Files stored under `uploads/<8‑char‑id>/` for easy cleanup                                                  |
| **Smart reruns**                            | Upload survives every Streamlit rerun – viewers and cards never disappear                                   |
| **Collapsible output cards**                | Keep the page tidy; expand only what you need                                                               |
| **Interactive 3-view viewer**               | Axial / Sagittal / Coronal sliders — click any view to jump the other two                                  |
| **Adjacency exploration**                   | Toggle *heat‑map* or *interactive network graph* (edge‑density slider)                                      |
| **Download buttons everywhere**             | NIfTI (`.nii.gz`) or raw PyTorch (`.pt`)                                                                    |
| **Chat assistant** (GPT‑4o‑mini by default) | • answers neuroscience questions via the RAG tool<br>• can call `run_unibrain_inference` tool automatically |

---

## 🏗️ Project Layout

```text
├─ main.py                 ← Streamlit app (single entry point)
├─ input_preview.py        ← Interactive 3-view NIfTI viewer
├─ model.py                ← UniBrain network (or drop‑in stub)
├─ assets/
│  ├─ tpl_img.npy          ← template volume
│  ├─ tpl_gm.npy           ← template GM mask
│  ├─ tpl_aal.npy          ← template AAL labels
│  └─ unibrain.pth         ← model weights
├─ prompts/
│  └─ unibrain_system_prompt.md
├─ unibrain.pdf            ← paper for RAG
└─ extra_knowledge.faiss/  ← pre‑built FAISS vector store
```

> **No UniBrain weights?**
> If `assets/unibrain.pth` is missing the app loads a **dummy stub** so you can
> still explore the UI.

---

## 🚀 Quick Start

```bash
git clone https://github.com/<your‑handle>/unibrain-assistant.git
cd unibrain-assistant
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt

# Set your API key (see below)
cp .env.example .env
# then edit .env and fill in your key

streamlit run main.py
```

Open [http://localhost:8501](http://localhost:8501) → upload a NIfTI → **Run UniBrain**.
Then talk to your data:

```
❯ run the pipeline
❯ what does a high AAL parcellation score mean?
❯ what is the predicted diagnosis?
```

---

## 🔑 API Key Setup

The app reads your key from a `.env` file (never commit this file — it is already in `.gitignore`).

```bash
cp .env.example .env
```

Then open `.env` and set:

```
OPENAI_API_KEY=sk-your-real-key-here
```

Alternatively export it in your shell before running:

```bash
export OPENAI_API_KEY="sk-..."
streamlit run main.py
```

| Var              | Purpose                                        |
| ---------------- | ---------------------------------------------- |
| `OPENAI_API_KEY` | Required for chat, RAG, and knowledge Q&A      |
| `IMG_SIZE`       | (optional) override default 96³ voxel size     |

---

## 📦 Core Dependencies

* `streamlit ≥1.32`
* `torch`, `numpy`, `nibabel`, `SimpleITK`, `networkx`, `matplotlib`
* `langchain`, `langchain‑openai`, `faiss‑cpu`
* `openai` (≥1.0 python SDK)

See `requirements.txt` for exact versions.

---

## 🤖 Command Grammar (for reference)

| Intent          | Examples (case‑insensitive)                               |
| --------------- | --------------------------------------------------------- |
| **Run**         | `run the pipeline`, `start inference`                     |
| **Ask paper**   | `what is skull stripping?`, `explain AAL parcellation`    |
| Anything else   | routed to the regular chat assistant                      |

---

## 🔬 Method Structure

<p align="center">
  <img src="./figures/structure.png" alt="End‑to‑end processing pipeline" width="100%"/>
</p>

---

## 🖼️ Demo

<p align="center">
  <img src="./figures/demo1.png" alt="Upload & preprocessing" width="100%"/>
  <img src="./figures/demo2.png" alt="Interactive slice viewer" width="100%"/>
  <img src="./figures/demo3.png" alt="3‑D volumetric viewer" width="100%"/>
</p>
<p align="center">
  <img src="./figures/demo4.png" alt="Graph visualisation" width="100%"/>
  <img src="./figures/demo5.png" alt="Chat‑driven control" width="100%"/>
</p>

---

## 📝 Contributing

PRs are welcome! Interesting directions:

* plug‑in **non‑rigid** registration back‑ends
* support **multi‑modal** inputs (fMRI + DTI)
* add **batch mode** & progress bars

---

## 📚 Citation

Please cite the following work if UniBrain‑Assistant contributes to your research:

```bibtex
@article{su2025end,
  title={End-to-End Deep Learning for Structural Brain Imaging: A Unified Framework},
  author={Su, Yao and Han, Keqi and Zeng, Mingjie and Sun, Lichao and Zhan, Liang and Yang, Carl and He, Lifang and Kong, Xiangnan},
  journal={arXiv preprint arXiv:2502.18523},
  year={2025}
}
```

---

## 📄 License

MIT – do whatever you want, but please cite the UniBrain paper if you use the
model for research.
