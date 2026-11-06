/*
 * red.js: ejecuta en el navegador la red GENERADORA y la red DISCRIMINADORA
 * entrenadas en gan_quickdraw.py (mismos pesos, exportados con exportar_web.py).
 *
 * Las BatchNorm ya vienen "fundidas" dentro de los pesos (ver exportar_web.py),
 * así que aquí solo hay capas Linear, Conv2d y ConvTranspose2d con su sesgo.
 */
(function (global) {
  "use strict";

  let P = null; // pesos decodificados (Float32Array por capa)

  // ---- float16 -> float32 (los pesos se guardaron en media precisión) ----------
  function tablaFloat16() {
    const t = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) {
      const signo = h & 0x8000 ? -1 : 1;
      const exp = (h >> 10) & 0x1f;
      const frac = h & 0x3ff;
      if (exp === 0) t[h] = signo * Math.pow(2, -14) * (frac / 1024);
      else if (exp === 31) t[h] = frac ? NaN : signo * Infinity;
      else t[h] = signo * Math.pow(2, exp - 15) * (1 + frac / 1024);
    }
    return t;
  }

  function base64ABytes(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return bytes;
  }

  function cargar() {
    if (P) return P;
    const fuente = global.PESOS_GAN;
    if (!fuente) throw new Error("No se encontraron los pesos (modelo/pesos.js)");
    const u16 = new Uint16Array(base64ABytes(fuente.base64).buffer);
    const tabla = tablaFloat16();
    P = {};
    for (const [nombre, info] of Object.entries(fuente.indice)) {
      const arr = new Float32Array(info.n);
      for (let i = 0; i < info.n; i++) arr[i] = tabla[u16[info.inicio + i]];
      P[nombre] = arr;
    }
    return P;
  }

  // ---- Capas -------------------------------------------------------------------
  // ConvTranspose2d(kernel=4, stride=2, padding=1): duplica el tamaño (n -> 2n)
  // pesos con forma [entrada][salida][4][4], como en PyTorch
  function convTranspuesta(x, cin, n, w, b, cout) {
    const m = n * 2, area = m * m;
    const y = new Float32Array(cout * area);
    for (let o = 0; o < cout; o++) y.fill(b[o], o * area, (o + 1) * area);
    for (let i = 0; i < cin; i++) {
      for (let iy = 0; iy < n; iy++) {
        for (let ix = 0; ix < n; ix++) {
          const v = x[(i * n + iy) * n + ix];
          if (v === 0) continue; // tras ReLU muchos valores son 0
          for (let ky = 0; ky < 4; ky++) {
            const oy = iy * 2 - 1 + ky;
            if (oy < 0 || oy >= m) continue;
            for (let kx = 0; kx < 4; kx++) {
              const ox = ix * 2 - 1 + kx;
              if (ox < 0 || ox >= m) continue;
              const wBase = i * cout * 16 + ky * 4 + kx;
              const yBase = oy * m + ox;
              for (let o = 0; o < cout; o++) y[o * area + yBase] += v * w[wBase + o * 16];
            }
          }
        }
      }
    }
    return y;
  }

  // Conv2d(kernel=4, stride=2, padding=1): reduce a la mitad (n -> n/2)
  // pesos con forma [salida][entrada][4][4]
  function convolucion(x, cin, n, w, b, cout) {
    const m = n / 2;
    const y = new Float32Array(cout * m * m);
    for (let o = 0; o < cout; o++) {
      for (let oy = 0; oy < m; oy++) {
        for (let ox = 0; ox < m; ox++) {
          let s = b[o];
          for (let i = 0; i < cin; i++) {
            const xBase = i * n * n, wBase = (o * cin + i) * 16;
            for (let ky = 0; ky < 4; ky++) {
              const iy = oy * 2 - 1 + ky;
              if (iy < 0 || iy >= n) continue;
              for (let kx = 0; kx < 4; kx++) {
                const ix = ox * 2 - 1 + kx;
                if (ix < 0 || ix >= n) continue;
                s += x[xBase + iy * n + ix] * w[wBase + ky * 4 + kx];
              }
            }
          }
          y[(o * m + oy) * m + ox] = s;
        }
      }
    }
    return y;
  }

  const relu = (a) => { for (let i = 0; i < a.length; i++) if (a[i] < 0) a[i] = 0; return a; };
  const leaky = (a, p) => { for (let i = 0; i < a.length; i++) if (a[i] < 0) a[i] *= p; return a; };

  // ---- RED GENERADORA: z (100) -> imagen 28x28 en [-1, 1] ------------------------
  function generar(z, conIntermedios) {
    cargar();
    // Capa 1: Linear 100 -> 6272 (+ BatchNorm fundida) + ReLU, vista como 128x7x7
    const h1 = new Float32Array(6272);
    for (let j = 0; j < 6272; j++) {
      let s = P.g1_b[j];
      const off = j * 100;
      for (let i = 0; i < 100; i++) s += P.g1_w[off + i] * z[i];
      h1[j] = s > 0 ? s : 0;
    }
    // Capa 2: ConvTranspose 128x7x7 -> 64x14x14 (+ BatchNorm fundida) + ReLU
    const h2 = relu(convTranspuesta(h1, 128, 7, P.g2_w, P.g2_b, 64));
    // Capa 3: ConvTranspose 64x14x14 -> 1x28x28 + Tanh
    const salida = convTranspuesta(h2, 64, 14, P.g3_w, P.g3_b, 1);
    for (let i = 0; i < salida.length; i++) salida[i] = Math.tanh(salida[i]);
    return conIntermedios ? { z, capa1: h1, capa2: h2, salida } : salida;
  }

  // ---- RED DISCRIMINADORA: imagen 28x28 en [-1, 1] -> probabilidad de "real" ----
  function discriminar(imagen, conIntermedios) {
    cargar();
    const h1 = leaky(convolucion(imagen, 1, 28, P.d1_w, P.d1_b, 64), 0.2);   // 64x14x14
    const h2 = leaky(convolucion(h1, 64, 14, P.d2_w, P.d2_b, 128), 0.2);     // 128x7x7
    let s = P.d3_b[0];
    for (let i = 0; i < 6272; i++) s += P.d3_w[i] * h2[i];                   // Linear 6272 -> 1
    const prob = 1 / (1 + Math.exp(-s));                                      // Sigmoid
    return conIntermedios ? { capa1: h1, capa2: h2, puntaje: s, prob } : prob;
  }

  // ---- Ruido z reproducible: PRNG con semilla + Box-Muller -------------------------
  function aleatorio(semilla) {
    let a = semilla >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function ruido(rnd, dim = 100) {
    const z = new Float32Array(dim);
    for (let i = 0; i < dim; i += 2) {
      const u = Math.max(rnd(), 1e-12), v = rnd();
      const r = Math.sqrt(-2 * Math.log(u));
      z[i] = r * Math.cos(2 * Math.PI * v);
      if (i + 1 < dim) z[i + 1] = r * Math.sin(2 * Math.PI * v);
    }
    return z;
  }

  // ---- Atiende un pedido (lo usa el Web Worker y también el modo sin worker) ----
  function ejecutar(m) {
    if (m.tipo === "generar") {
      const r = generar(m.z, m.intermedios);
      const img = m.intermedios ? r.salida : r;
      const res = { id: m.id, img };
      if (m.intermedios) { res.capa1 = r.capa1; res.capa2 = r.capa2; }
      if (m.juzgar) res.prob = discriminar(img);
      return res;
    }
    if (m.tipo === "juzgar") {
      const d = discriminar(m.img, m.intermedios);
      return m.intermedios ? { id: m.id, prob: d.prob, capa1: d.capa1, capa2: d.capa2 } : { id: m.id, prob: d };
    }
    return { id: m.id };
  }

  global.GAN = { cargar, generar, discriminar, aleatorio, ruido, ejecutar };
})(typeof self !== "undefined" ? self : globalThis);
