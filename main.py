import os
import base64
from collections import Counter

import numpy as np
import cv2

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from ultralytics import YOLO


MODEL_PATH = os.getenv("MODEL_PATH", "models/yolo_seg_best.pt")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"Modelo YOLO não encontrado em {MODEL_PATH}. "
        f"Coloque seu yolo_seg_best.pt em 'models/' ou defina MODEL_PATH."
    )

model = YOLO(MODEL_PATH)

# nomes de classe (cuvette, liquid, bubble, etc.)
raw_names = getattr(model.model, "names", None)
if isinstance(raw_names, dict):
    CLASS_NAMES = {int(k): v for k, v in raw_names.items()}
elif isinstance(raw_names, (list, tuple)):
    CLASS_NAMES = {i: n for i, n in enumerate(raw_names)}
else:
    CLASS_NAMES = {0: "cuvette", 1: "liquid", 2: "bubble"}



app = FastAPI(
    title="IFOTOM QA – YOLO Cuvette Segmentation",
    description="Interface web para avaliar imagens de cubetas (cuvette/liquid/bubble) com YOLO.",
    version="1.0.0",
)

templates = Jinja2Templates(directory="templates")



def run_segmentation(image_bytes: bytes) -> dict:
    """Roda a IA de segmentação na imagem e monta os dados para o front."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError("Não foi possível decodificar a imagem. Formato inválido ou arquivo corrompido.")

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

    viz_bgr = r.plot()  
    viz_rgb = cv2.cvtColor(viz_bgr, cv2.COLOR_BGR2RGB)
    ok, png_bytes = cv2.imencode(".png", viz_rgb)
    if not ok:
        raise ValueError("Falha ao gerar imagem anotada.")

    plot_base64 = base64.b64encode(png_bytes.tobytes()).decode("utf-8")

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

    qa_flags = []
    if counts.get(0, 0) == 0:
        qa_flags.append(
            {"level": "danger", "message": "Cubeta (cuvette) não detectada na imagem."}
        )
    if counts.get(1, 0) == 0:
        qa_flags.append(
            {
                "level": "warning",
                "message": "Região de líquido não detectada com clareza.",
            }
        )
    if counts.get(2, 0) > 0:
        qa_flags.append(
            {
                "level": "warning",
                "message": f"Foram detectadas {counts[2]} região(ões) com bolhas.",
            }
        )

    if not qa_flags:
        qa_flags.append(
            {
                "level": "success",
                "message": "Imagem adequada: cubeta e líquido detectados, sem bolhas visíveis.",
            }
        )

    return {
        "plot_base64": plot_base64,
        "per_class": per_class,
        "qa_flags": qa_flags,
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
    except Exception as e:
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
