
# IFOTOM QA — Passo a passo completo

Este pacote é um **guia executável** para você montar o fluxo de QA por Visão Computacional do IFOTOM,
desde os dados até o servidor de API. Siga as etapas na ordem.

> **Resumo rápido**: você vai baixar pequenos datasets públicos (3–4 GB), converter rótulos para YOLO-Seg,
unificar *splits*, treinar no Colab (YOLO-Seg nano) e rodar um servidor FastAPI com os endpoints `/qa/framecheck` e `/qa/yolo`.
O app manda miniaturas/ROI e recebe `issues` + `overlays`/métricas para decidir repetir ou prosseguir.

---

## 0) Pré-requisitos

- **Git + VS Code** (organização do repo e edição de código).
- **Python 3.10+** (local) para conversões e o servidor FastAPI.
- **Colab (ou Kaggle)** para treinar o YOLO-Seg *nano* e validar.
- (Opcional) **DVC** para versionar dados grandes.

Estrutura deste pacote:
```
ifotom-qa-starter/
  datasets/ifotom/yolo_seg/         # destino final (YOLO-Seg) após conversões
  framecheck/                        # trainer multi-rótulo (FrameCheck)
  scripts/                           # conversores e utilitários
  api/fastapi_app/                   # servidor FastAPI (stub funcional)
  train_seg.yaml                     # config de treino YOLO-Seg (Ultralytics)
  datasets/ifotom/yolo_seg_cuvette.yaml   # YAML do dataset (nomes e splits)
  requirements.txt                   # deps para scripts/servidor
  README.md                          # este guia
```

---

## 1) Baixar os datasets públicos (mínimo viável)

Você **não** precisa baixar coleções enormes. Sugerido:
- **Vector-LabPics V1 (~3 GB)** — aprender *vidraria + líquido* (pré-treino).
- **Bubbles (Heßenkemper, ~500–600 MB)** — máscaras de bolhas.
- **BubbleID (Dryad, ~1 GB)** — bolhas / instâncias para robustez.

> Baixe **apenas** os zips principais de cada dataset. Consulte as páginas oficiais para os links.
> Salve tudo em `data/raw/` (crie a pasta na raiz deste projeto).

**Exemplo de estrutura após baixar/extrair:**
```
data/
  raw/
    LabPicsV1/ ... (conteúdo extraído)
    bubbles_hzdr/ ... (conteúdo extraído)
    bubbleid/ ... (conteúdo extraído)
```

---

## 2) Converter rótulos para **seu esquema YOLO-Seg**

Esquema de classes:
```
0: cuvette
1: liquid
2: bubble
3: glare
4: smudge
```

- **LabPics → cuvette/liquid**: use `scripts/convert_labpics.py` apontando para as anotações do LabPics.
  Ele mapeia `vessel`→`cuvette` e `liquid`→`liquid` (ajuste se o dataset usar outro nome).
- **Bubbles (Heßenkemper) e BubbleID → bubble**: use `scripts/convert_bubbles.py` para vetorizar máscaras (contornos) em polígonos YOLO-Seg.
- (Opcional) **glare** e **smudge**: rotule **20–50** imagens do seu domínio (com qualquer anotador de polígonos) e salve direto no formato YOLO-Seg.

**Comandos (exemplos):**
```bash
# Ambiente Python para scripts
python -m venv .venv && source .venv/bin/activate  # (Linux/macOS) ou .venv\Scripts\activate (Windows)
pip install -r requirements.txt

# Converter LabPics (ajuste --labpics-dir com o caminho real)
python scripts/convert_labpics.py   --labpics-dir data/raw/LabPicsV1   --out-dir datasets/ifotom/yolo_seg   --split all

# Converter bolhas (ajuste --images-dir/--masks-dir conforme dataset baixado)
python scripts/convert_bubbles.py   --images-dir data/raw/bubbles_hzdr/images   --masks-dir data/raw/bubbles_hzdr/masks   --out-dir datasets/ifotom/yolo_seg
```

Ao final, você terá **imagens** e **labels .txt** no formato YOLO-Seg dentro de `datasets/ifotom/yolo_seg/`.

---

## 3) Unificar *splits* (train/val/test) de forma estratificada

Rode:
```bash
python scripts/split_stratified.py   --yolo-root datasets/ifotom/yolo_seg   --train 0.80 --val 0.10 --test 0.10   --seed 42
```

Isto embaralha por **origem** e cria os diretórios `images/{train,val,test}` e `labels/{train,val,test}`.

---

## 4) Treinar o **YOLO-Seg** no Colab

### 4.1 Suba este diretório para o seu Drive (ou dê `git clone` dentro do Colab).

### 4.2 Notebook Colab (copie/cole)
```python
!pip install -q ultralytics==8.3.0 opencv-python pillow onnx onnxruntime
from ultralytics import YOLO

# Caminhos: ajuste se necessário (se montou Drive, aponte para sua pasta)
data_yaml = "datasets/ifotom/yolo_seg_cuvette.yaml"

# 1) Treino (modelo nano)
model = YOLO("yolov8n-seg.pt")     # ou yolo11n-seg.pt se disponível
model.train(data=data_yaml, epochs=200, imgsz=640, batch=16,
            project="runs/ifotom-seg", name="cuvette-qa-seg-nano")

# 2) Validação
model.val()

# 3) Exportar (ONNX/TFLite) — útil para on-device no futuro
model.export(format="onnx", dynamic=True)
model.export(format="tflite")
```

> Se der *OOM*, reduza `batch` para 8/4 ou `imgsz` para 512. Se quiser acelerar o *convergir*, congele camadas do backbone nas primeiras 20–40 épocas.

---

## 5) Treinar o **FrameCheck** (classificação multi-rótulo)

Monte `framecheck/train.csv` e `framecheck/val.csv` assim:
```
filename,bubble,tilt,low_fill,glare,smudge,no_cuvette
path/to/img_0001.jpg,1,0,1,0,0,0
path/to/img_0002.jpg,0,0,0,1,1,0
...
```

Rode o trainer:
```bash
python framecheck/trainer.py   --train-csv framecheck/train.csv   --val-csv framecheck/val.csv   --epochs 60 --img-size 224   --out-dir framecheck/runs/exp1
```

Ele salva um `.pt` leve (MobileNet/EfficientNet-lite). Depois é só carregar no servidor (`/qa/framecheck`).

---

## 6) Servidor FastAPI (stub funcional)

Instalar deps (local):
```bash
pip install -r requirements.txt
```

Rodar servidor:
```bash
uvicorn api.fastapi_app.main:app --reload --port 8000
```

- `POST /qa/framecheck` — recebe miniaturas/ROI e retorna `issues` com *scores* (no stub, valores simulados).
- `POST /qa/yolo` — recebe imagem e retorna `bbox_cuvette`, `metrics` (fill/tilt/bolha/glare) e máscaras (simuladas no stub).
- **Substitua os stubs** carregando seus pesos: YOLO (Ultralytics) para `/qa/yolo` e o classificador `.pt` para `/qa/framecheck`.

Acesse `http://127.0.0.1:8000/docs` para a UI Swagger (OpenAPI).

---

## 7) Integração no App (resumo)

- **Após burst**: envie 1–3 miniaturas/ROI para `/qa/framecheck`.
  - Se `action=="repeat"`, **repita a captura** (e mostre o Marcinho com a dica).
  - Se `proceed`, continua o fluxo normal.
- (Opcional) Para *explicabilidade*, chame `/qa/yolo` em 1 frame do burst para mostrar overlay (bbox/máscaras) e métricas (fill%, tilt).

**Overlay**: o servidor já retorna `polygons`/`bbox` no pixel-space. Desenhe em cima da preview.

---

## 8) Roadmap rápido (após MVP)

1. Afinar thresholds no servidor (sem atualizar o app).
2. Adicionar classe `bubble` no YOLO-Seg (se começar só com cuvette+liquid).
3. Treinar `glare`/`smudge` com poucas anotações do seu ambiente.
4. Medir impacto: redução de CV da absorbância e de medidas descartadas.

---

## 9) Licenças e citações

- Cheque os termos de uso dos datasets (LabPics, Heßenkemper, BubbleID) e **cite os autores** quando publicar/demonstrar.
- Este starter é MIT (veja cabeçalhos dos arquivos).

Bom trabalho! 🚀
# QA_IA
