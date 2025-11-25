import os
import base64
from collections import Counter

import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

# ---------------------------------------------------------
# Imports de dependências de visão computacional (robustos)
# ---------------------------------------------------------

try:
    import cv2
except ImportError as e:
    print(f"[BOOT] ERRO importando cv2: {e}")
    cv2 = None

try:
    from ultralytics import YOLO
except ImportError as e:
    print(f"[BOOT] ERRO importando ultralytics.YOLO: {e}")
    YOLO = None

# ---------------------------------------------------------
# Carregamento do modelo (robusto para Cloud Run)
# ---------------------------------------------------------

MODEL_PATH = os.getenv("MODEL_PATH", "models/yolo_seg_best.pt")

model = None

if YOLO is None:
    print(
        "[BOOT] AVISO: ultralytics não está disponível. "
        "O servidor vai subir, mas as análises de QA não vão funcionar."
    )
else:
    try:
        if os.path.exists(MODEL_PATH):
            print(f"[BOOT] Carregando modelo YOLO de: {MODEL_PATH}")
            model = YOLO(MODEL_PATH)
            print("[BOOT] Modelo carregado com sucesso.")
        else:
            print(
                f"[BOOT] AVISO: modelo não encontrado em {MODEL_PATH}. "
                f"O servidor irá subir, mas as requisições de análise vão falhar até o modelo ser configurado."
            )
    except Exception as e:
        print(f"[BOOT] ERRO ao carregar modelo em {MODEL_PATH}: {e}")
        model = None

# nomes de classe (cuvette, liquid, bubble, etc.)
raw_names = getattr(model.model, "names", None) if model is not None else None
if isinstance(raw_names, dict):
    CLASS_NAMES = {int(k): v for k, v in raw_names.items()}
elif isinstance(raw_names, (list, tuple)):
    CLASS_NAMES = {i: n for i, n in enumerate(raw_names)}
else:
    # fallback simples se não conseguir ler do modelo
    CLASS_NAMES = {0: "cuvette", 1: "liquid", 2: "bubble"}


def _find_class_id(label: str):
    label = label.lower()
    for cid, name in CLASS_NAMES.items():
        if str(name).lower() == label:
            return cid
    return None


ID_CUVETTE = _find_class_id("cuvette")
ID_LIQUID = _find_class_id("liquid")
ID_BUBBLE = _find_class_id("bubble")
ID_GLARE = _find_class_id("glare")
ID_SMUDGE = _find_class_id("smudge")


def build_qa_analysis(counts: Counter) -> dict:
    """
    Gera um 'score' de qualidade, resumo em texto, lista de issues
    e uma recomendação baseada nas classes detectadas.
    """
    issues = []
    score = "ok"  # ok | attention | reject

    # Atalhos pra ler as contagens (se a classe existir no modelo)
    cuv = counts.get(ID_CUVETTE, 0) if ID_CUVETTE is not None else 0
    liq = counts.get(ID_LIQUID, 0) if ID_LIQUID is not None else 0
    bub = counts.get(ID_BUBBLE, 0) if ID_BUBBLE is not None else 0
    gla = counts.get(ID_GLARE, 0) if ID_GLARE is not None else 0
    smu = counts.get(ID_SMUDGE, 0) if ID_SMUDGE is not None else 0

    # Regras bem simples (pode ir refinando depois)
    if cuv == 0:
        issues.append("Nenhuma cubeta (cuvette) foi detectada.")
        score = "reject"

    if liq == 0:
        issues.append("Região de líquido não foi detectada com clareza.")
        if score != "reject":
            score = "attention"

    if bub > 0:
        issues.append(f"Foram detectadas {bub} região(ões) com bolhas (bubble).")
        if score == "ok":
            score = "attention"

    if gla > 0:
        issues.append(
            "Foi detectado brilho/reflexo (glare) que pode distorcer a leitura da absorbância."
        )
        if score == "ok":
            score = "attention"

    if smu > 0:
        issues.append(
            "Foram detectidas manchas/contaminação (smudge) na área da cubeta."
        )
        if score == "ok":
            score = "attention"

    # Mensagens finais
    if score == "ok":
        summary = (
            "Imagem adequada: cubeta e líquido detectados, sem bolhas ou artefatos relevantes."
        )
        recommendation = "Pode seguir com a análise espectrofotométrica usando esta imagem."
    elif score == "attention":
        summary = "Imagem com atenção: há fatores que podem prejudicar a leitura."
        recommendation = (
            "Considere repetir a foto após eliminar bolhas, reduzir reflexos ou limpar a cubeta. "
            "Se não for possível repetir, prossiga com cautela e valide o resultado."
        )
    else:  # reject
        summary = "Imagem crítica: provavelmente inadequada para uma análise confiável."
        recommendation = (
            "Recomenda-se descartar esta imagem e capturar uma nova, verificando alinhamento da cubeta, "
            "presença de líquido e ausência de bolhas/reflexos fortes."
        )

    return {
        "score": score,
        "summary": summary,
        "issues": issues,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------
# FastAPI + Templates
# ---------------------------------------------------------

app = FastAPI(
    title="IFOTOM QA – YOLO Cuvette Segmentation",
    description="Interface web para avaliar imagens de cubetas (cuvette/liquid/bubble) com YOLO.",
    version="1.0.0",
)

templates = Jinja2Templates(directory="templates")


def run_segmentation(image_bytes: bytes) -> dict:
    """Roda a IA de segmentação na imagem e monta os dados para o front."""

    if cv2 is None or YOLO is None:
        raise ValueError(
            "Dependências de visão computacional não estão disponíveis no servidor "
            "(cv2/ultralytics). Verifique as dependências e o ambiente de deploy."
        )

    if model is None:
        raise ValueError(
            "O modelo de QA não está carregado no servidor. "
            "Verifique a configuração do modelo (MODEL_PATH) ou contate o suporte."
        )

    # Primeiro: trata caso de arquivo vazio
    if not image_bytes:
        raise ValueError(
            "Nenhuma imagem foi recebida. "
            "Selecione um arquivo de imagem antes de clicar em 'Analisar imagem'."
        )

    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError(
            "Não foi possível decodificar a imagem. "
            "Verifique se o arquivo é uma imagem válida (JPG, PNG, etc.)."
        )

    # Inference
    results = model.predict(
        img_bgr,
        imgsz=640,
        conf=0.25,
        iou=0.7,
        verbose=False,
    )
    if not results:
        raise ValueError("O modelo não retornou resultados para essa imagem.")

    r = results[0]

    # Imagem anotada
    viz_bgr = r.plot()  # BGR com caixas/máscaras
    viz_rgb = cv2.cvtColor(viz_bgr, cv2.COLOR_BGR2RGB)
    ok, png_bytes = cv2.imencode(".png", viz_rgb)
    if not ok:
        raise ValueError("Falha ao gerar imagem anotada.")

    plot_base64 = base64.b64encode(png_bytes.tobytes()).decode("utf-8")

    # Contagem de classes detectadas
    counts = Counter()
    if r.boxes is not None and r.boxes.cls is not None:
        for c in r.boxes.cls.tolist():
            counts[int(c)] += 1

    per_class = []
    for cid, name in CLASS_NAMES.items():
        per_class.append(
            {
                "id": cid,
                "name": name,
                "count": counts.get(cid, 0),
            }
        )

    qa = build_qa_analysis(counts)

    return {
        "plot_base64": plot_base64,
        "per_class": per_class,
        "qa": qa,
    }


@app.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    """Página principal com formulário de upload."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": None,
            "error": None,
        },
    )


@app.post("/", response_class=HTMLResponse)
async def upload_page(request: Request, file: UploadFile = File(...)):
    """Recebe a imagem via formulário e renderiza o resultado no HTML."""
    if file is None:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": None,
                "error": "Nenhum arquivo enviado. Selecione uma imagem.",
            },
        )

    try:
        contents = await file.read()
        result = run_segmentation(contents)
        result["filename"] = file.filename

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": result,
                "error": None,
            },
        )
    except ValueError as e:
        # Erros "esperados" de entrada do usuário (arquivo vazio, formato inválido, modelo ausente, etc.)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": None,
                "error": str(e),
            },
        )
    except Exception as e:
        # Erros inesperados
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": None,
                "error": f"Erro ao processar imagem: {e}",
            },
        )


@app.post("/predict", response_class=JSONResponse)
async def predict_api(file: UploadFile = File(...)):
    """Endpoint JSON para uso via API (POST /predict com campo 'file')."""
    if file is None:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")

    try:
        contents = await file.read()
        result = run_segmentation(contents)
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {e}")
