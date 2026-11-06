"""
Exporta lo necesario para la página web (carpeta docs/):

  docs/modelo/pesos.js     Pesos ENTRENADOS de la red Generadora y la Discriminadora
                           (float16 en base64) para ejecutarlas en el navegador.
  docs/modelo/datos.js     Historial de entrenamiento por época, dibujos reales de
                           Quick Draw y los dibujos del generador en cada época.
  docs/modelo/codigo.js    El archivo gan_quickdraw.py completo, para mostrarlo en la
                           página con números de línea.

Se ejecuta DESPUÉS de entrenar con gan_quickdraw.py:
    python exportar_web.py

Detalle técnico: en modo evaluación, una BatchNorm es solo una multiplicación y una
suma por canal, así que se "funde" (fold) con la capa anterior. Así el navegador solo
tiene que calcular capas Linear / Conv / ConvTranspose con su sesgo.
"""
import base64
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BASE = Path(__file__).resolve().parent
RESULTADOS = BASE / "resultados"
DOCS_MODELO = BASE / "docs" / "modelo"
DOCS_MODELO.mkdir(parents=True, exist_ok=True)

CATEGORIA = "cat"
NUM_IMAGENES = 30_000
SEMILLA = 42
EPS_BN = 1e-5  # epsilon por defecto de BatchNorm en PyTorch


def fundir_bn(sd, prefijo_bn):
    """Devuelve (escala, desplazamiento) equivalentes a una BatchNorm en modo evaluación."""
    gamma = sd[f"{prefijo_bn}.weight"]
    beta = sd[f"{prefijo_bn}.bias"]
    media = sd[f"{prefijo_bn}.running_mean"]
    var = sd[f"{prefijo_bn}.running_var"]
    escala = gamma / torch.sqrt(var + EPS_BN)
    return escala, beta - media * escala


# ---------------------------------------------------------------------------
# 1. Pesos del GENERADOR (con las BatchNorm fundidas)
# ---------------------------------------------------------------------------
g = torch.load(RESULTADOS / "generador.pth", map_location="cpu", weights_only=True)
g = {k.replace("red.", ""): v.float() for k, v in g.items()}

s1, b1 = fundir_bn(g, "capa1_batchnorm")
s2, b2 = fundir_bn(g, "capa2_batchnorm")
pesos_g = {
    # Linear 100 -> 6272: forma [6272, 100]
    "g1_w": g["capa1_lineal.weight"] * s1[:, None],
    "g1_b": b1,
    # ConvTranspose2d 128 -> 64: forma [128 (entrada), 64 (salida), 4, 4]
    "g2_w": g["capa2_deconv.weight"] * s2[None, :, None, None],
    "g2_b": b2,
    # ConvTranspose2d 64 -> 1: forma [64, 1, 4, 4]
    "g3_w": g["capa3_deconv.weight"],
    "g3_b": g["capa3_deconv.bias"],
}

# ---------------------------------------------------------------------------
# 2. Pesos del DISCRIMINADOR (con la BatchNorm fundida)
# ---------------------------------------------------------------------------
d = torch.load(RESULTADOS / "discriminador.pth", map_location="cpu", weights_only=True)
d = {k.replace("red.", ""): v.float() for k, v in d.items()}

s3, b3 = fundir_bn(d, "capa2_batchnorm")
pesos_d = {
    # Conv2d 1 -> 64: forma [64, 1, 4, 4]
    "d1_w": d["capa1_conv.weight"],
    "d1_b": d["capa1_conv.bias"],
    # Conv2d 64 -> 128: forma [128, 64, 4, 4]
    "d2_w": d["capa2_conv.weight"] * s3[:, None, None, None],
    "d2_b": b3,
    # Linear 6272 -> 1: forma [1, 6272]
    "d3_w": d["capa3_lineal.weight"],
    "d3_b": d["capa3_lineal.bias"],
}

# Empaqueta todo en un solo bloque float16
indice, trozos, desplazamiento = {}, [], 0
for nombre, tensor in {**pesos_g, **pesos_d}.items():
    plano = tensor.detach().numpy().astype(np.float16).ravel()
    indice[nombre] = {"inicio": desplazamiento, "n": int(plano.size), "forma": list(tensor.shape)}
    trozos.append(plano)
    desplazamiento += plano.size
bloque = np.concatenate(trozos).astype("<f2").tobytes()

(DOCS_MODELO / "pesos.js").write_text(
    "// Pesos entrenados de la GAN (generado por exportar_web.py, no editar a mano)\n"
    f"self.PESOS_GAN = {{indice: {json.dumps(indice)}, "
    f"base64: \"{base64.b64encode(bloque).decode()}\"}};\n",
    encoding="utf-8",
)
print(f"pesos.js: {desplazamiento:,} valores ({len(bloque) / 1e6:.2f} MB en float16)")

# ---------------------------------------------------------------------------
# 3. Datos para las gráficas y las cuadrículas de la página
# ---------------------------------------------------------------------------
# 3a. Historial por época
with open(RESULTADOS / "historial_entrenamiento.csv", encoding="utf-8") as f:
    historial = [{k: float(v) for k, v in fila.items()} for fila in csv.DictReader(f)]
for fila in historial:
    fila["epoca"] = int(fila["epoca"])

# 3b. Dibujos reales: la misma muestra aleatoria que usó el entrenamiento
todos = np.load(BASE / "datos" / f"{CATEGORIA}.npy", mmap_mode="r", allow_pickle=False)
rng = np.random.default_rng(SEMILLA)
indices = np.sort(rng.choice(todos.shape[0], NUM_IMAGENES, replace=False))
reales = np.asarray(todos[indices[64:112]], dtype=np.uint8)  # 48 dibujos de 28x28

# 3c. Dibujos del generador en cada época (ruido fijo), sacados de evolucion.gif.
#     Cada cuadro es una cuadrícula 8x8 (celdas de 28 px, margen 2) ampliada x3.
POR_EPOCA = 24
gif = Image.open(RESULTADOS / "evolucion.gif")
epocas = []
for i in range(gif.n_frames):
    gif.seek(i)
    cuadro = np.asarray(gif.convert("L"))[::3, ::3]   # deshace la ampliación x3
    celdas = []
    for k in range(POR_EPOCA):
        f, c = divmod(k, 8)
        y, x = 2 + f * 30, 2 + c * 30
        celdas.append(255 - cuadro[y:y + 28, x:x + 28])  # trazo = 255, fondo = 0
    epocas.append(np.stack(celdas).astype(np.uint8))
epocas = np.stack(epocas)  # (25, 24, 28, 28)

resumen = json.loads((RESULTADOS / "resumen.json").read_text(encoding="utf-8"))
datos = {
    "categoria": CATEGORIA,
    "historial": historial,
    "resumen": resumen,
    "reales_b64": base64.b64encode(reales.tobytes()).decode(),
    "epocas_b64": base64.b64encode(epocas.tobytes()).decode(),
    "por_epoca": POR_EPOCA,
}
(DOCS_MODELO / "datos.js").write_text(
    "// Datos del entrenamiento (generado por exportar_web.py, no editar a mano)\n"
    f"window.DATOS_GAN = {json.dumps(datos, ensure_ascii=False)};\n",
    encoding="utf-8",
)
print(f"datos.js: {len(historial)} épocas, {len(reales)} dibujos reales, "
      f"{epocas.shape[0]}x{epocas.shape[1]} dibujos generados por época")

# ---------------------------------------------------------------------------
# 4. Código fuente completo, para el visor de código de la página
# ---------------------------------------------------------------------------
fuente = (BASE / "gan_quickdraw.py").read_text(encoding="utf-8")
(DOCS_MODELO / "codigo.js").write_text(
    "// Copia de gan_quickdraw.py (generado por exportar_web.py, no editar a mano)\n"
    f"window.CODIGO_GAN = {json.dumps(fuente, ensure_ascii=False)};\n",
    encoding="utf-8",
)
print(f"codigo.js: gan_quickdraw.py con {fuente.count(chr(10))} líneas")
