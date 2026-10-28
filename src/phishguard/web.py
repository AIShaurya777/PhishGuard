"""The project website (Milestone 8): serves the page in web/ and two JSON endpoints.

Run:  .venv/bin/python -m phishguard.web     then open http://127.0.0.1:8000

POST /api/check    {"url": "..."} -> what the models read, their scores and warnings (serving.Checker)
GET  /api/numbers  accuracy over 5 splits (results/) and the robustness numbers (models/site_numbers.json)
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from phishguard.data import REPO_ROOT
from phishguard.evaluate import load_results
from phishguard.serving import Checker

SITE_DIR = REPO_ROOT / "web"

# (results group, saved model name, label on the website, input)
ACCURACY_ROWS = [
    ("m2", "XGBoost (all columns)", "XGBoost on all 50 dataset columns", "full link"),
    ("m3_views", "Char n-gram LogReg | raw URL", "Character n-grams", "full link"),
    ("m4_deep", "Char-CNN | raw URL", "Char-CNN", "full link"),
    ("m5_hybrid", "Hybrid CNN + features | raw URL", "Hybrid: CNN + link features", "full link"),
    ("m3_views", "XGBoost URL features | domain only", "XGBoost on 40 link features", "domain only"),
    ("m4_deep", "BiLSTM | domain only", "BiLSTM", "domain only"),
    ("m3_views", "Char n-gram LogReg | domain only", "Character n-grams", "domain only"),
    ("m4_deep", "Char-CNN | domain only", "Char-CNN", "domain only"),
    ("m5_hybrid", "Hybrid CNN + features | domain only", "Hybrid: CNN + link features", "domain only"),
]

checker: Checker | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global checker
    checker = Checker.load()
    yield


app = FastAPI(title="PhishGuard", lifespan=lifespan, docs_url=None, redoc_url=None)


class CheckRequest(BaseModel):
    url: str


@app.post("/api/check")
def check(request: CheckRequest) -> dict:
    try:
        return checker.check(request.url)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))


@app.get("/api/numbers")
def numbers() -> dict:
    return {"accuracy": accuracy_table(),
            **{key: value for key, value in checker.numbers.items() if key != "suffix_priors"}}


def accuracy_table() -> list[dict]:
    """Test accuracy, mean ± sd over 5 site-owner splits, read from results/ (never typed by hand)."""
    tables, rows = {}, []
    for group, name, label, view in ACCURACY_ROWS:
        if group not in tables:
            tables[group] = load_results(group).set_index("model")
        r = tables[group].loc[name]
        rows.append({"model": label, "input": view, "accuracy": float(r["accuracy"]),
                     "sd": float(r["accuracy_sd"]), "roc_auc": float(r["roc_auc"])})
    return rows


app.mount("/", StaticFiles(directory=SITE_DIR, html=True), name="site")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
