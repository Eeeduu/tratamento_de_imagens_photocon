"""
servidor_ia.py
Servidor local de remocao de fundo.
- Fundo branco/claro: remocao por cor (preserva elementos finos como correntes)
- Fundo complexo/lifestyle: IA rembg
Rode com: python servidor_ia.py
Acesse o editor em: http://localhost:5001
"""

import importlib, subprocess, sys

def _instalar_se_ausente(pacotes):
    for modulo, pacote in pacotes.items():
        try:
            importlib.import_module(modulo)
        except ImportError:
            print(f"[bootstrap] Instalando '{pacote}'...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pacote])

_instalar_se_ausente({
    "flask": "flask",
    "flask_cors": "flask-cors",
    "rembg": "rembg[cpu]",
    "PIL": "Pillow",
    "numpy": "numpy",
})

from flask import Flask, request, send_file, send_from_directory
from flask_cors import CORS
from rembg import remove, new_session
from PIL import Image, ImageFilter
import io, os, webbrowser, threading, time
import numpy as np

app = Flask(__name__)
CORS(app)

PASTA_ATUAL = os.path.dirname(os.path.abspath(__file__))

# Carrega modelo IA para imagens de fundo complexo
print("[IA] Carregando modelo de segmentacao...")
try:
    SESSAO_IA = new_session("isnet-general-use")
    print("[IA] Modelo carregado com sucesso!")
except Exception as e:
    print(f"[IA] Aviso: usando u2net ({e})")
    SESSAO_IA = new_session("u2net")


# ---------------------------------------------------------------------------
# Funcoes de processamento
# ---------------------------------------------------------------------------

def detectar_fundo_branco(img_rgb, limiar=230, min_pontos=5):
    """
    Amostra 8 pontos nas bordas da imagem.
    Se >= min_pontos forem pixels claros (acima do limiar), o fundo e branco.
    """
    w, h = img_rgb.size
    margem = max(5, min(w, h) // 25)
    pontos = [
        (margem, margem), (w - margem, margem),
        (margem, h - margem), (w - margem, h - margem),
        (w // 2, margem), (w // 2, h - margem),
        (margem, h // 2), (w - margem, h // 2),
    ]
    brancos = sum(
        1 for px, py in pontos
        if all(c >= limiar for c in img_rgb.getpixel((px, py))[:3])
    )
    return brancos >= min_pontos


def remover_fundo_branco(img_pil):
    """
    Remove fundo branco usando flood-fill a partir das bordas.
    
    Diferente de remover TODOS os pixels brancos, esta funcao
    remove apenas os pixels brancos que estao CONECTADOS a borda
    da imagem (ou seja, o fundo real). Pixels brancos dentro do
    produto (reflexos, brilhos, metal) sao preservados.
    """
    img_rgba = img_pil.convert("RGBA")
    img_hsv = img_pil.convert("HSV")

    dados_rgba = np.array(img_rgba, dtype=np.uint8)
    dados_hsv = np.array(img_hsv, dtype=np.uint8)
    h, w = dados_hsv.shape[:2]

    s = dados_hsv[:, :, 1]
    v = dados_hsv[:, :, 2]

    # Mascara de pixels "brancos" candidatos (MUITO mais rígida)
    # Background de estúdio costuma ser 255 ou muito perto disso.
    # A alça da bolsa, embora clara, tem textura e cor que a impedem de ser 255 puro.
    candidatos = (v >= 252) & (s <= 8)

    # Flood-fill: marca apenas os pixels brancos conectados as bordas
    from collections import deque

    visitado = np.zeros((h, w), dtype=bool)
    fundo = np.zeros((h, w), dtype=bool)
    fila = deque()

    # Semeia a fila com todos os pixels das 4 bordas que sao brancos
    for x in range(w):
        if candidatos[0, x] and not visitado[0, x]:
            fila.append((0, x))
            visitado[0, x] = True
        if candidatos[h-1, x] and not visitado[h-1, x]:
            fila.append((h-1, x))
            visitado[h-1, x] = True

    for y in range(h):
        if candidatos[y, 0] and not visitado[y, 0]:
            fila.append((y, 0))
            visitado[y, 0] = True
        if candidatos[y, w-1] and not visitado[y, w-1]:
            fila.append((y, w-1))
            visitado[y, w-1] = True

    # BFS: expande para pixels brancos vizinhos conectados
    while fila:
        cy, cx = fila.popleft()
        fundo[cy, cx] = True

        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and not visitado[ny, nx] and candidatos[ny, nx]:
                visitado[ny, nx] = True
                fila.append((ny, nx))

    # Aplica transparencia apenas no fundo conectado as bordas
    dados_rgba[fundo, 3] = 0

    return Image.fromarray(dados_rgba, "RGBA")


def remover_fundo_ia(dados_entrada):
    """Remove fundo complexo usando IA (rembg) com alpha matting."""
    return remove(
        dados_entrada,
        session=SESSAO_IA,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10,
    )


# ---------------------------------------------------------------------------
# Rotas Flask
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(PASTA_ATUAL, "index.html")


@app.route("/remover-fundo", methods=["POST"])
def remover_fundo():
    if "imagem" not in request.files:
        return {"erro": "Nenhuma imagem enviada."}, 400

    arquivo = request.files["imagem"]
    dados_entrada = arquivo.read()

    try:
        img = Image.open(io.BytesIO(dados_entrada)).convert("RGB")
        eh_fundo_branco = detectar_fundo_branco(img)

        if eh_fundo_branco:
            # Fundo branco: remocao por cor (preserva correntes e elementos finos)
            print(f"[PROC] Fundo branco detectado -> usando remocao por cor")
            resultado = remover_fundo_branco(img)
        else:
            # Fundo complexo: usa IA
            print(f"[PROC] Fundo complexo detectado -> usando IA (rembg)")
            dados_saida = remover_fundo_ia(dados_entrada)
            resultado = Image.open(io.BytesIO(dados_saida))

        buffer = io.BytesIO()
        resultado.save(buffer, format="PNG")
        buffer.seek(0)
        return send_file(buffer, mimetype="image/png", download_name="sem_fundo.png")

    except Exception as e:
        print(f"[ERRO] {e}")
        return {"erro": str(e)}, 500


def abrir_navegador():
    time.sleep(2)
    webbrowser.open("http://localhost:5001")


if __name__ == "__main__":
    print("=" * 55)
    print("  [IA] Servidor Local - Remocao de Fundo")
    print("  Modos: cor (fundo branco) | IA (fundo complexo)")
    print("=" * 55)
    print("  Acesse: http://localhost:5001")
    print("  Pressione Ctrl+C para encerrar.")
    print("=" * 55)
    threading.Thread(target=abrir_navegador, daemon=True).start()
    app.run(host="0.0.0.0", port=5001, debug=False)
