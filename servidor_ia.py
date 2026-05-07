"""
servidor_ia.py
Servidor local de remocao de fundo com suporte a download em lote (ZIP).
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

from flask import Flask, request, send_file, send_from_directory, jsonify
from flask_cors import CORS
from rembg import remove, new_session
from PIL import Image, ImageFilter
import io, os, webbrowser, threading, time, zipfile, shutil
import numpy as np

app = Flask(__name__)
CORS(app)

PASTA_ATUAL = os.path.dirname(os.path.abspath(__file__))
PASTA_OUTPUT = os.path.join(PASTA_ATUAL, "output_ia")

# Garante que a pasta de output existe e esta limpa ao iniciar
if os.path.exists(PASTA_OUTPUT):
    shutil.rmtree(PASTA_OUTPUT)
os.makedirs(PASTA_OUTPUT)

# Carrega modelo IA
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
    w, h = img_rgb.size
    margem = max(5, min(w, h) // 25)
    pontos = [(margem, margem), (w - margem, margem), (margem, h - margem), (w - margem, h - margem), (w // 2, margem), (w // 2, h - margem), (margem, h // 2), (w - margem, h // 2)]
    brancos = sum(1 for px, py in pontos if all(c >= limiar for c in img_rgb.getpixel((px, py))[:3]))
    return brancos >= min_pontos

def remover_fundo_branco(img_pil):
    img_rgba = img_pil.convert("RGBA")
    img_hsv = img_pil.convert("HSV")
    dados_rgba = np.array(img_rgba, dtype=np.uint8)
    dados_hsv = np.array(img_hsv, dtype=np.uint8)
    h, w = dados_hsv.shape[:2]
    s, v = dados_hsv[:, :, 1], dados_hsv[:, :, 2]
    candidatos = (v >= 252) & (s <= 8)
    from collections import deque
    visitado, fundo, fila = np.zeros((h, w), dtype=bool), np.zeros((h, w), dtype=bool), deque()
    for x in range(w):
        if candidatos[0, x] and not visitado[0, x]: fila.append((0, x)); visitado[0, x] = True
        if candidatos[h-1, x] and not visitado[h-1, x]: fila.append((h-1, x)); visitado[h-1, x] = True
    for y in range(h):
        if candidatos[y, 0] and not visitado[y, 0]: fila.append((y, 0)); visitado[y, 0] = True
        if candidatos[y, w-1] and not visitado[y, w-1]: fila.append((y, w-1)); visitado[y, w-1] = True
    while fila:
        cy, cx = fila.popleft(); fundo[cy, cx] = True
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and not visitado[ny, nx] and candidatos[ny, nx]: visitado[ny, nx] = True; fila.append((ny, nx))
    dados_rgba[fundo, 3] = 0
    return Image.fromarray(dados_rgba, "RGBA")

def remover_fundo_ia(dados_entrada):
    return remove(dados_entrada, session=SESSAO_IA, alpha_matting=True, alpha_matting_foreground_threshold=240, alpha_matting_background_threshold=10, alpha_matting_erode_size=10)


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
    nome_original = request.form.get("nome", "imagem.png")
    dados_entrada = arquivo.read()

    try:
        img = Image.open(io.BytesIO(dados_entrada)).convert("RGB")
        eh_fundo_branco = detectar_fundo_branco(img)

        if eh_fundo_branco:
            resultado = remover_fundo_branco(img)
        else:
            dados_saida = remover_fundo_ia(dados_entrada)
            resultado = Image.open(io.BytesIO(dados_saida))

        # Salva na pasta de output para o ZIP
        nome_limpo = os.path.splitext(nome_original)[0] + "_sem_fundo.png"
        caminho_salvamento = os.path.join(PASTA_OUTPUT, nome_limpo)
        resultado.save(caminho_salvamento, format="PNG")

        # Retorna a imagem individual normalmente
        buffer = io.BytesIO()
        resultado.save(buffer, format="PNG")
        buffer.seek(0)
        return send_file(buffer, mimetype="image/png")

    except Exception as e:
        print(f"[ERRO] {e}")
        return {"erro": str(e)}, 500

@app.route("/baixar_zip")
def baixar_zip():
    """Compacta a pasta de output e envia o ZIP."""
    arquivos = [f for f in os.listdir(PASTA_OUTPUT) if f.endswith('.png')]
    if not arquivos:
        return {"erro": "Nenhuma imagem processada ainda."}, 400

    memoria_zip = io.BytesIO()
    with zipfile.ZipFile(memoria_zip, 'w') as zf:
        for f in arquivos:
            zf.write(os.path.join(PASTA_OUTPUT, f), f)
    
    memoria_zip.seek(0)
    return send_file(memoria_zip, mimetype="application/zip", as_attachment=True, download_name="imagens_sem_fundo.zip")

@app.route("/limpar", methods=["POST"])
def limpar_pasta():
    """Limpa a pasta de processados."""
    for f in os.listdir(PASTA_OUTPUT):
        os.remove(os.path.join(PASTA_OUTPUT, f))
    return jsonify({"status": "limpo"})

def abrir_navegador():
    time.sleep(2)
    webbrowser.open("http://localhost:5001")

if __name__ == "__main__":
    print("=" * 55)
    print("  [IA] Servidor Local - Remocao de Fundo (Com suporte a ZIP)")
    print("=" * 55)
    print("  Acesse: http://localhost:5001")
    print("=" * 55)
    threading.Thread(target=abrir_navegador, daemon=True).start()
    app.run(host="0.0.0.0", port=5001, debug=False)
