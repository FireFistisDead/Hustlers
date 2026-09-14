# 🛡️ PharmaGuard AI

**Regulatory Intelligence Control Tower**
Drug Safety Signal Detector & Regulatory Submission Readiness Checker

---

## 👥 Team

| Field | Value |
|---|---|
| **Team Name** | Hustlers |
| **Track** | AI |
| **Team Lead** | Yash Khare — 23aiml027@charusat.edu.in |
| **Members** | Puja Rachchh, Krishna Chodvadiya, Harsh Shah |

---

## 🎯 Problem Statement

FDA's FAERS database contains 20M+ adverse event reports. The Vioxx signal was missed for years — causing 27,000+ heart attacks before regulatory action. Separately, a drug approval CTD dossier spans 100,000+ pages across 5 modules — one missing section triggers rejection, costing 6–12 months and $50–100M. Both problems share the same root cause: too much complex data for manual review.

---

## 💡 Solution

PharmaGuard AI is a Regulatory Intelligence Control Tower that automates the end-to-end pipeline from raw adverse event reports to regulatory action briefs:

**Adverse Event Data → Safety Evidence → Signal Detection → Regulatory Impact → CTD Submission Readiness → Regulatory Action Brief → Human Review**

It applies deterministic disproportionality analysis (PRR, ROR, IC) on FDA FAERS data, maps detected signals to regulatory impact categories, evaluates CTD dossier completeness against ICH guidelines, and surfaces concise AI-generated narratives for human reviewers — collapsing weeks of manual pharmacovigilance work into minutes.

---

## ✨ Key Features

- **FAERS Signal Detection:** Automated disproportionality analysis (PRR, ROR, IC/IC025) across 20M+ adverse event reports using DuckDB for in-process analytics at speed.
- **Regulatory Impact Scoring:** Deterministic rule engine that maps detected signals to regulatory action categories (label update, REMS, market withdrawal).
- **CTD Submission Readiness Checker:** ICH M2/M4 module completeness engine that flags missing sections before submission.
- **Regulatory Action Brief:** Structured narrative report combining signal evidence + impact assessment, ready for human reviewer sign-off.
- **Human-in-the-Loop Review:** Lightweight workflow for reviewer assignment, audit trail, and decision capture — powered by IBM Bob as the primary development environment.

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **Languages** | Python 3.11, TypeScript |
| **Frameworks** | FastAPI, Pydantic v2, React (planned) |
| **IBM Technologies** | watsonx.ai (Granite), IBM Bob |
| **Databases** | DuckDB |
| **Other** | pandas, numpy, scipy, pytest, httpx, uvicorn |

---

## 📁 Repository Structure

```
src/
├── backend/
│   ├── app/
│   │   ├── main.py           ← FastAPI app factory
│   │   ├── api/              ← HTTP routers
│   │   ├── ingest/           ← FAERS file loading & validation
│   │   ├── model/            ← Normalized domain models
│   │   ├── signal/           ← Disproportionality calculations
│   │   ├── impact/           ← Signal → regulatory impact rules
│   │   ├── ctd/              ← ICH CTD readiness engine
│   │   ├── review/           ← Human-in-the-loop workflow
│   │   └── narrative/        ← Granite/watsonx narrative layer
│   └── tests/
├── config/
│   └── settings.yaml.example
├── data/
│   ├── raw/                  ← FAERS quarterly ASCII exports (gitignored)
│   └── demo/                 ← Curated demo slices for dev/CI
├── frontend/                 ← React UI (planned)
├── pyproject.toml
└── .env.example
docs/
├── problem-statement.md
├── solution-overview.md
├── architecture.md
└── setup-guide.md
demo/
presentation/
submission.yaml
```

---

## ⚡ How to Run

```bash
# 1. Clone the repo
git clone https://github.com/your-org/pharmaguard-ai.git
cd pharmaguard-ai/Hustlers/src

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Configure environment
cp .env.example .env
# Edit .env with your values

# 5. Start the backend
uvicorn backend.app.main:app --reload --port 8000

# 6. Run tests
pytest
```

The API will be available at `http://localhost:8000`
Interactive docs: `http://localhost:8000/docs`

---

## 🖥️ Demo

| Artifact | Link |
|---|---|
| 📹 Demo Video | [See demo/demo-video-link.txt](demo/demo-video-link.txt) |
| 🌐 Live Demo | [See demo/live-demo-url.txt](demo/live-demo-url.txt) |
| 🖼️ Screenshots | [See demo/screenshots/](demo/screenshots/) |
| 📊 Presentation | [See presentation/](presentation/) |

---

## ⚠️ Known Limitations

- FAERS ingest, signal calculation, CTD engine, and narrative modules are scaffolded but not yet implemented — schema confirmation from real FAERS files is pending.
- No authentication on API endpoints (prototype scope).
- Frontend is a placeholder; React implementation is a future sprint.
- watsonx.ai / Granite SDK not yet installed — narrative layer is deferred.

---

## 🏅 What We're Most Proud Of

The end-to-end pipeline architecture that connects raw adverse event data all the way to a human-reviewable regulatory brief — a flow that typically requires weeks of manual pharmacovigilance work. The deterministic signal engine and CTD readiness checker are the technical centrepieces worth closest scrutiny.
