/*
 * app.js: contenido vivo de la página.
 * - Portada: campo de puntos (semitono) hecho con gatos que dibuja la red Generadora.
 * - 01 y 02: los valores reales de cada capa de G y de D, en vivo.
 * - 03 y 05: gráficas de las pérdidas y de los valores por época.
 * - 04: el archivo gan_quickdraw.py completo, con las líneas resaltadas.
 * - 05: la escena de las épocas, la única parte que sigue al scroll.
 * - 06: dibujos reales y generados en la época 25.
 */
(function () {
  "use strict";
  var reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return [].slice.call((r || document).querySelectorAll(s)); };
  var f = function (v, d) { return v.toFixed(d == null ? 2 : d); };
  var DATOS = window.DATOS_GAN, HIST = DATOS.historial;
  var TINTA = [242, 242, 238], NARANJA = [254, 133, 73];

  // ------------------------------------------------------------------ datos
  function desdeBase64(s) {
    var bin = atob(s), u = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    return u;
  }
  var REALES = desdeBase64(DATOS.reales_b64);   // 48 dibujos reales de 28x28 (0..255)
  var EPOCAS = desdeBase64(DATOS.epocas_b64);   // 25 épocas x 24 dibujos del generador
  var POR_EPOCA = DATOS.por_epoca, N_REALES = REALES.length / 784;
  function real(k) { return REALES.subarray(k * 784, (k + 1) * 784); }
  function realTanh(k) {
    var u = real(k), o = new Float32Array(784);
    for (var i = 0; i < 784; i++) o[i] = u[i] / 127.5 - 1;   // mismo escalado que en el entrenamiento
    return o;
  }
  function deEpoca(e, k) { var i = (e - 1) * POR_EPOCA + k; return EPOCAS.subarray(i * 784, (i + 1) * 784); }

  // ------------------------------------------------- motor: Web Worker, o el hilo principal si no hay
  var Motor = (function () {
    var w = null, sig = 0, pend = {}, cola = [], local = false, cargando = null, activo = false;
    function cargarLocal() {
      if (!cargando) cargando = new Promise(function (ok, mal) {
        var s = document.createElement("script");
        s.src = "modelo/pesos.js"; s.onload = ok; s.onerror = mal;
        document.head.appendChild(s);
      });
      return cargando;
    }
    function bombear() {
      if (activo) return;
      activo = true;
      cargarLocal().then(function paso() {
        var t = cola.shift();
        if (t) t.res(GAN.ejecutar(t.msg));
        if (cola.length) setTimeout(paso, 0); else activo = false;
      });
    }
    function pasarALocal() {
      local = true; w = null;
      Object.keys(pend).forEach(function (k) { cola.push(pend[k]); });
      pend = {};
      if (cola.length) bombear();
    }
    try {
      if (location.protocol === "file:") throw new Error("sin worker en file://");
      w = new Worker("js/trabajador.js");
      w.onmessage = function (ev) {
        var r = ev.data, t = pend[r.id];
        if (!t) return;
        delete pend[r.id];
        if (!r.cancelado) t.res(r);
      };
      w.onerror = function (e) { if (e.preventDefault) e.preventDefault(); pasarALocal(); };
    } catch (e) { pasarALocal(); }
    function pedir(msg, alta) {
      msg.id = ++sig; msg.alta = !!alta;
      return new Promise(function (res) {
        var t = { msg: msg, res: res };
        if (w && !local) { pend[msg.id] = t; w.postMessage(msg); }
        else { if (alta) cola.unshift(t); else cola.push(t); bombear(); }
      });
    }
    return {
      generar: function (z, o) { o = o || {}; return pedir({ tipo: "generar", z: z, intermedios: !!o.intermedios, juzgar: !!o.juzgar }, o.alta); },
      juzgar: function (img, o) { o = o || {}; return pedir({ tipo: "juzgar", img: img, intermedios: !!o.intermedios }, o.alta); },
    };
  })();
  var azar = GAN.aleatorio((Date.now() ^ 0x9e3779b9) >>> 0);
  function nuevoZ() { return GAN.ruido(azar); }

  // ------------------------------------------------------------- pintar
  // Trazo en la tinta de la página sobre transparente: el fondo lo pone cada lienzo
  var c28 = document.createElement("canvas"); c28.width = c28.height = 28;
  var x28 = c28.getContext("2d"), datos28 = x28.createImageData(28, 28);
  function valor01(img, i) { var t = img instanceof Uint8Array ? img[i] / 255 : (img[i] + 1) / 2; return t < 0 ? 0 : t > 1 ? 1 : t; }
  function pintar(lienzo, img) {
    var p = datos28.data;
    for (var i = 0; i < 784; i++) {
      p[4 * i] = TINTA[0]; p[4 * i + 1] = TINTA[1]; p[4 * i + 2] = TINTA[2]; p[4 * i + 3] = valor01(img, i) * 255;
    }
    x28.putImageData(datos28, 0, 0);
    var ctx = lienzo.getContext("2d");
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
    ctx.clearRect(0, 0, lienzo.width, lienzo.height);
    ctx.drawImage(c28, 0, 0, lienzo.width, lienzo.height);
  }
  function lienzos(cont, n, tam) {
    var lista = [];
    for (var i = 0; i < n; i++) { var c = document.createElement("canvas"); c.width = c.height = tam; cont.appendChild(c); lista.push(c); }
    return lista;
  }
  // Mapas de activación: de negro a la tinta (G) o al naranja (D)
  var cm = document.createElement("canvas"), xm = cm.getContext("2d");
  function pintarMapa(lienzo, datos, inicio, n, max, color) {
    cm.width = cm.height = n;
    var id = xm.createImageData(n, n), p = id.data;
    for (var i = 0; i < n * n; i++) {
      var t = Math.pow(Math.min(1, Math.max(0, datos[inicio + i] / max)), 0.8);
      for (var c = 0; c < 3; c++) p[4 * i + c] = 6 + (color[c] - 6) * t;
      p[4 * i + 3] = 255;
    }
    xm.putImageData(id, 0, 0);
    var ctx = lienzo.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(cm, 0, 0, lienzo.width, lienzo.height);
  }
  function masActivos(datos, canales, area, k) {
    var medias = [];
    for (var c = 0; c < canales; c++) { var s = 0; for (var i = 0; i < area; i++) s += datos[c * area + i]; medias.push([s / area, c]); }
    return medias.sort(function (a, b) { return b[0] - a[0]; }).slice(0, k).map(function (m) { return m[1]; });
  }
  function pintarCapa(lista, datos, sel, n, color) {
    var area = n * n, max = 1e-6;
    sel.forEach(function (c) { for (var i = 0; i < area; i++) if (datos[c * area + i] > max) max = datos[c * area + i]; });
    sel.forEach(function (c, k) { pintarMapa(lista[k], datos, c * area, n, max, color); });
  }
  function dibujarZ(svg, z) {
    var s = '<line x1="0" x2="100" y1="50" y2="50"/>';
    for (var i = 0; i < z.length; i++) {
      var h = Math.min(50, Math.abs(z[i]) / 3 * 50);
      s += '<rect x="' + (i + 0.15) + '" width="0.7" y="' + (z[i] > 0 ? 50 - h : 50) + '" height="' + Math.max(h, 0.4) + '"/>';
    }
    svg.innerHTML = s;
  }
  // Un ciclo de 3.5 s mientras el elemento está en pantalla (con movimiento reducido: una sola vez)
  function mientrasVisible(el, ms, fn) {
    var t = null, hecho = false;
    new IntersectionObserver(function (es) {
      if (es[0].isIntersecting) {
        if (!hecho || !reduce) fn();
        hecho = true;
        if (!reduce && !t) t = setInterval(function () { if (document.visibilityState === "visible") fn(); }, ms);
      } else if (t) { clearInterval(t); t = null; }
    }, { rootMargin: "100px 0px" }).observe(el);
  }

  // ------------------------------------------------- portada: pared de gatos generados, en puntos
  // El campo es una pared completa de gatos de la red Generadora. animaciones.js lo recorta como una
  // tarjeta a la derecha del texto y, al bajar, la agranda hasta la pantalla completa (primera página).
  (function () {
    var lienzo = $("#campo"), hero = $(".hero"), media = $("#hero-media"), copy = $("#hero-copy"), ctx = lienzo.getContext("2d");
    var BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];
    var paso = 4, tam = 144, campo = null, W = 0, H = 0, cols = 0, filas = 0, gatos = [];
    function mascara(x, y) {
      // viñeta suave hacia los bordes
      var bx = Math.min(x, W - x) / (W * 0.1), by = Math.min(y, H - y) / (H * 0.12);
      return Math.min(1, Math.max(0.3, Math.min(bx, by)));
    }
    // la tarjeta inicial: a la derecha del título (escritorio) o entre la nota y el título (pantallas angostas)
    function tarjeta() {
      var m = media.getBoundingClientRect(), guardado = copy.style.transform;
      copy.style.transform = "none";
      var c = copy.getBoundingClientRect(), gut = c.left - m.left, der = 0, rango = document.createRange();
      [$(".hero h1"), $(".hero__lead"), $(".hero__note")].forEach(function (el) {
        rango.selectNodeContents(el);
        [].forEach.call(rango.getClientRects(), function (q) { der = Math.max(der, q.right - m.left); });
      });
      var botones = $(".hero__foot .btn-row").getBoundingClientRect(), texto = $(".hero__text").getBoundingClientRect(), nota = $(".hero__note").getBoundingClientRect();
      copy.style.transform = guardado;
      window.HERO_CARD = m.width >= 1024
        ? { t: 40, r: gut, b: m.height - (botones.top - m.top - 28), l: Math.min(der + 56, m.width * 0.62) }
        : { t: nota.bottom - m.top + 20, r: gut, b: m.height - (texto.top - m.top - 24), l: gut };
    }
    function preparar() {
      var r = media.getBoundingClientRect();
      W = Math.round(r.width); H = Math.round(r.height);
      if (!W || !H) return;
      var dpr = Math.min(2, devicePixelRatio || 1);
      lienzo.width = W * dpr; lienzo.height = H * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      paso = W < 640 ? 3 : 4; tam = W < 640 ? 96 : 144;
      cols = Math.ceil(W / tam); filas = Math.ceil(H / tam);
      campo = new Float32Array(cols * tam * filas * tam);
      for (var k = 0; k < gatos.length; k++) if (gatos[k] && k < cols * filas) ponerGato(k, gatos[k]);
      tarjeta();
      dibujar(0, 0, W, H);
      if (window.redibujarAnimaciones) window.redibujarAnimaciones();
    }
    function ponerGato(k, img) {
      var cx = (k % cols) * tam, cy = Math.floor(k / cols) * tam, ancho = cols * tam;
      for (var y = 0; y < tam; y++) for (var x = 0; x < tam; x++) {
        // interpolación bilineal del dibujo de 28x28
        var u = x / tam * 27, v = y / tam * 27, i0 = Math.floor(u), j0 = Math.floor(v), du = u - i0, dv = v - j0;
        var a = valor01(img, j0 * 28 + i0), b = valor01(img, j0 * 28 + Math.min(27, i0 + 1));
        var c = valor01(img, Math.min(27, j0 + 1) * 28 + i0), d = valor01(img, Math.min(27, j0 + 1) * 28 + Math.min(27, i0 + 1));
        campo[(cy + y) * ancho + cx + x] = (a * (1 - du) + b * du) * (1 - dv) + (c * (1 - du) + d * du) * dv;
      }
    }
    // dibuja los puntos de un rectángulo (todo el campo, o solo la celda de un gato que cambió)
    function dibujar(x0, y0, w, h) {
      if (!campo) return;
      ctx.clearRect(x0, y0, w, h);
      var ancho = cols * tam, s = paso === 4 ? 2.2 : 1.7;
      var xi = Math.ceil(x0 / paso) * paso, yi = Math.ceil(y0 / paso) * paso;
      for (var y = yi; y < Math.min(H, y0 + h); y += paso) for (var x = xi; x < Math.min(W, x0 + w); x += paso) {
        var m = mascara(x, y), trazo = campo[y * ancho + x];
        var v = trazo * (0.4 + 0.7 * m) + 0.1 * m;
        var umbral = (BAYER[((y / paso) % 4) * 4 + ((x / paso) % 4)] + 0.5) / 16;
        if (v > umbral) {
          ctx.fillStyle = "rgba(242,242,238," + (0.25 + 0.75 * Math.min(1, trazo * 0.95 + m * 0.2)).toFixed(2) + ")";
          ctx.fillRect(x, y, s, s);
        }
      }
    }
    preparar();
    // los gatos llegan de a uno desde el worker; se redibuja en lotes
    var total = cols * filas, pendiente = false, llenos = 0;
    for (var k = 0; k < total; k++) (function (k) {
      Motor.generar(nuevoZ()).then(function (r) {
        gatos[k] = r.img; ponerGato(k, r.img); llenos++;
        if (!pendiente) { pendiente = true; requestAnimationFrame(function () { pendiente = false; dibujar(0, 0, W, H); }); }
      });
    })(k);
    // muro vivo (primera página): cada tanto un gato se reemplaza por otro nuevo
    if (!reduce) setInterval(function () {
      if (llenos < total || document.visibilityState !== "visible" || hero.getBoundingClientRect().bottom < 0) return;
      var k = Math.floor(Math.random() * cols * filas);
      Motor.generar(nuevoZ()).then(function (r) {
        gatos[k] = r.img; ponerGato(k, r.img);
        dibujar((k % cols) * tam, Math.floor(k / cols) * tam, tam, tam);
      });
    }, 900);
    var t;
    addEventListener("resize", function () { clearTimeout(t); t = setTimeout(preparar, 200); });
  })();

  // ------------------------------------------------- 01. dentro de la red Generadora
  (function () {
    var zsvg = $("#vg-z"), c1 = lienzos($("#vg-c1"), 16, 28), c2 = lienzos($("#vg-c2"), 16, 56), out = $("#vg-out"), estado = $("#vg-estado");
    var sel = null;
    mientrasVisible($("#generadora"), 3500, function () {
      var z = nuevoZ();
      Motor.generar(z, { intermedios: true, juzgar: true, alta: true }).then(function (r) {
        if (!sel) sel = { c1: masActivos(r.capa1, 128, 49, 16), c2: masActivos(r.capa2, 64, 196, 16) };
        dibujarZ(zsvg, z);
        pintarCapa(c1, r.capa1, sel.c1, 7, TINTA);
        pintarCapa(c2, r.capa2, sel.c2, 14, TINTA);
        pintar(out, r.img);
        estado.innerHTML = "D(G(z)) <b>" + f(r.prob) + "</b>";
      });
    });
  })();

  // ------------------------------------------------- 02. dentro de la red Discriminadora
  (function () {
    var cin = $("#vd-in"), tipo = $("#vd-tipo"), c1 = lienzos($("#vd-c1"), 16, 56), c2 = lienzos($("#vd-c2"), 16, 28);
    var marca = $("#vd-marca"), prob = $("#vd-prob"), veredicto = $("#vd-veredicto"), sel = null, turnoReal = true;
    mientrasVisible($("#discriminadora"), 3500, function () {
      var esReal = turnoReal; turnoReal = !turnoReal;
      var k = Math.floor(Math.random() * N_REALES);
      var entrada = esReal ? Promise.resolve(realTanh(k)) : Motor.generar(nuevoZ(), { alta: true }).then(function (r) { return r.img; });
      entrada.then(function (img) {
        return Motor.juzgar(img, { intermedios: true, alta: true }).then(function (r) {
          if (!sel) sel = { c1: masActivos(r.capa1, 64, 196, 16), c2: masActivos(r.capa2, 128, 49, 16) };
          pintar(cin, esReal ? real(k) : img);
          tipo.innerHTML = esReal ? "entrada <b>real</b> · Quick Draw" : "entrada <b>generada</b> por G";
          pintarCapa(c1, r.capa1, sel.c1, 14, NARANJA);
          pintarCapa(c2, r.capa2, sel.c2, 7, NARANJA);
          marca.style.left = (r.prob * 100) + "%";
          prob.textContent = f(r.prob);
          veredicto.innerHTML = r.prob >= 0.5 ? 'D dice: <b class="txt-d">real</b>' : "D dice: <b>falso</b>";
        });
      });
    });
  })();

  // ------------------------------------------------------------------ gráficas (SVG)
  var tooltip = $("#tooltip"), idGraf = 0;
  function grafica(cont, cfg) {
    var clip = "rec" + (++idGraf);
    function dibujar() {
      var W = cont.clientWidth, H = cont.clientHeight, m = { l: 44, r: cfg.derecha || 56, t: 12, b: 32 };
      if (!W) return;
      var X = function (v) { return m.l + (v - cfg.x[0]) / (cfg.x[1] - cfg.x[0]) * (W - m.l - m.r); };
      var Y = function (v) { return m.t + (1 - (v - cfg.y[0]) / (cfg.y[1] - cfg.y[0])) * (H - m.t - m.b); };
      var s = '<defs><clipPath id="' + clip + '"><rect x="' + m.l + '" y="' + m.t + '" width="' + (W - m.l - m.r) + '" height="' + (H - m.t - m.b) + '"/></clipPath></defs>';
      cfg.yTicks.forEach(function (v) {
        s += '<line class="rej" x1="' + m.l + '" x2="' + (W - m.r) + '" y1="' + Y(v) + '" y2="' + Y(v) + '"/>' +
          '<text class="eje" x="' + (m.l - 10) + '" y="' + (Y(v) + 4) + '" text-anchor="end">' + cfg.fmtY(v) + "</text>";
      });
      cfg.xTicks.forEach(function (v) { s += '<text class="eje" x="' + X(v) + '" y="' + (H - 6) + '" text-anchor="middle">' + cfg.fmtX(v) + "</text>"; });
      if (cfg.ref != null) {
        s += '<line class="ref" x1="' + m.l + '" x2="' + (W - m.r) + '" y1="' + Y(cfg.ref) + '" y2="' + Y(cfg.ref) + '"/>';
        if (cfg.refTexto) s += '<text class="ref-t" x="' + (W - m.r) + '" y="' + (Y(cfg.ref) - 8) + '" text-anchor="end">' + cfg.refTexto + "</text>";
      }
      s += '<g clip-path="url(#' + clip + ')">';
      cfg.series.forEach(function (se) {
        s += '<path class="s ' + se.color + (se.punteada ? " pt" : "") + '" d="M' + se.puntos.map(function (p) {
          return X(p[0]).toFixed(1) + "," + Y(Math.max(cfg.y[0] - 1, Math.min(cfg.y[1] + 1, p[1]))).toFixed(1);
        }).join("L") + '"/>';
      });
      s += "</g>";
      if (cfg.finales) cfg.series.forEach(function (se) {
        var u = se.puntos[se.puntos.length - 1];
        s += '<text class="fin" x="' + (X(u[0]) + 8) + '" y="' + (Y(u[1]) + 4) + '">' + f(u[1]) + "</text>";
      });
      if (cfg.hover) {
        s += '<g class="hover" style="display:none"><line class="cruz" y1="' + m.t + '" y2="' + (H - m.b) + '"/>' +
          cfg.series.map(function (se) { return '<circle class="pto ' + se.color + '" r="5"/>'; }).join("") + "</g>" +
          '<rect class="zona" x="' + m.l + '" y="' + m.t + '" width="' + (W - m.l - m.r) + '" height="' + (H - m.t - m.b) + '" fill="transparent"/>';
      }
      cont.innerHTML = '<svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="' + cfg.titulo + '">' + s + "</svg>";
      if (cfg.hover) conectarHover(cont.querySelector("svg"), X, Y);
    }
    function conectarHover(svg, X, Y) {
      var zona = svg.querySelector(".zona"), g = svg.querySelector(".hover");
      var linea = g.querySelector("line"), puntos = [].slice.call(g.querySelectorAll("circle"));
      zona.addEventListener("pointermove", function (ev) {
        var x = ev.clientX - svg.getBoundingClientRect().left;
        var mejor = cfg.series[0].puntos.map(function (p) { return p[0]; }).reduce(function (a, b) { return Math.abs(X(b) - x) < Math.abs(X(a) - x) ? b : a; });
        g.style.display = "";
        linea.setAttribute("x1", X(mejor)); linea.setAttribute("x2", X(mejor));
        tooltip.innerHTML = "<b>época " + mejor + "</b>" + cfg.series.map(function (se, k) {
          var v = se.valor(mejor);
          puntos[k].setAttribute("cx", X(mejor)); puntos[k].setAttribute("cy", Y(v));
          return '<div><span><i style="background:var(--' + se.color + ')"></i>' + se.nombre + "</span><em>" + f(v, 3) + "</em></div>";
        }).join("");
        tooltip.hidden = false;
        var tx = ev.clientX + 16;
        if (tx + tooltip.offsetWidth > innerWidth - 12) tx = ev.clientX - tooltip.offsetWidth - 16;
        tooltip.style.left = tx + "px"; tooltip.style.top = (ev.clientY - 20) + "px";
      });
      zona.addEventListener("pointerleave", function () { g.style.display = "none"; tooltip.hidden = true; });
    }
    new ResizeObserver(dibujar).observe(cont);
    dibujar();
  }

  // 03. Las funciones de pérdida según la salida del discriminador
  (function () {
    var ps = [];
    for (var i = 0; i <= 200; i++) ps.push(0.004 + i * (0.992 / 200));
    var curva = function (fn) { return ps.map(function (p) { return [p, fn(p)]; }); };
    var ticksP = [0, 0.25, 0.5, 0.75, 1], fmtP = function (v) { return f(v); };
    grafica($("#graf-ld"), {
      titulo: "Pérdida del discriminador", x: [0, 1], y: [0, 5.5], xTicks: ticksP, yTicks: [0, 1, 2, 3, 4, 5], fmtX: fmtP, fmtY: String, derecha: 16,
      series: [{ color: "d", puntos: curva(function (p) { return -Math.log(p); }) }, { color: "d", punteada: true, puntos: curva(function (p) { return -Math.log(1 - p); }) }],
    });
    grafica($("#graf-lg"), {
      titulo: "Pérdida del generador", x: [0, 1], y: [-5.5, 5.5], xTicks: ticksP, yTicks: [-4, -2, 0, 2, 4], fmtX: fmtP, fmtY: String, derecha: 16,
      series: [{ color: "g", puntos: curva(function (p) { return -Math.log(p); }) }, { color: "g", punteada: true, puntos: curva(function (p) { return Math.log(1 - p); }) }],
    });
  })();

  // 05. Valores de G y D por época
  function serieEpocas(nombre, clave, color) {
    return { nombre: nombre, color: color, puntos: HIST.map(function (h) { return [h.epoca, h[clave]]; }), valor: function (e) { return HIST[e - 1][clave]; } };
  }
  var ticksEp = [1, 5, 10, 15, 20, 25];
  grafica($("#graf-salida"), {
    titulo: "Salida del discriminador por época", x: [1, 25], y: [0, 1], xTicks: ticksEp, yTicks: [0, 0.25, 0.5, 0.75, 1],
    fmtX: String, fmtY: function (v) { return f(v); }, ref: 0.5, refTexto: "equilibrio 0.5", finales: true, hover: true,
    series: [serieEpocas("D(x) reales", "D_x", "d"), serieEpocas("D(G(z)) falsos", "D_G_z", "g")],
  });
  grafica($("#graf-perdidas"), {
    titulo: "Pérdidas por época", x: [1, 25], y: [0, 5], xTicks: ticksEp, yTicks: [0, 1, 2, 3, 4, 5],
    fmtX: String, fmtY: String, finales: true, hover: true,
    series: [serieEpocas("Pérdida D", "perdida_D", "d"), serieEpocas("Pérdida G", "perdida_G", "g")],
  });
  $("#tabla-epocas").innerHTML = [1, 5, 10, 15, 20, 25].map(function (e) {
    var h = HIST[e - 1];
    return "<tr><td>" + e + '</td><td class="n">' + f(h.perdida_D, 3) + '</td><td class="n">' + f(h.perdida_G, 3) +
      '</td><td class="n">' + f(h.D_x, 3) + '</td><td class="n">' + f(h.D_G_z, 3) + "</td></tr>";
  }).join("");

  // ------------------------------------------------- 05. escena de las épocas: animaciones.js la mueve con el scroll
  (function () {
    var celdas = lienzos($("#tira-celdas"), POR_EPOCA, 112), actual = 0;
    var num = $("#tira-ep"), barra = $("#tira-barra"), epNum = $("#ep-num"), epDx = $("#ep-dx"), epDgz = $("#ep-dgz"), epPerd = $("#ep-perd");
    window.mostrarEpoca = function (e) {
      if (e === actual) return;
      actual = e;
      celdas.forEach(function (c, k) { pintar(c, deEpoca(e, k)); });
      var h = HIST[e - 1];
      num.textContent = e; epNum.textContent = e;
      barra.style.width = (e / 25 * 100) + "%";
      epDx.textContent = f(h.D_x); epDgz.textContent = f(h.D_G_z);
      epPerd.textContent = f(h.perdida_D) + " / " + f(h.perdida_G);
    };
    window.mostrarEpoca(25);   // la composición final, hasta que la escena tome el control
  })();

  // ------------------------------------------------- 06. resultado: reales y generados en la época 25
  lienzos($("#rej-reales"), 24, 112).forEach(function (c, k) { pintar(c, real(k)); });
  lienzos($("#rej-generados"), 24, 112).forEach(function (c, k) { pintar(c, deEpoca(25, k)); });

  // ------------------------------------------------- 04. código: gan_quickdraw.py completo con números de línea
  function resaltar(src) {
    var esc = function (s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); };
    var re = /(#[^\n]*)|("(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')|\b(class|def|return|for|in|if|else|import|from|super|self|None|True|False|not|and|or)\b|\b(\d+(?:\.\d+)?(?:e-?\d+)?)\b|\b(nn|torch)\b|\b([A-Za-z_]\w*)(?=\()/g;
    var out = "", ult = 0, m;
    while ((m = re.exec(src))) {
      var cls = m[1] ? "c" : m[2] ? "s" : m[3] ? "k" : m[4] ? "n" : m[5] ? "t" : "f";
      out += esc(src.slice(ult, m.index)) + '<span class="tk-' + cls + '">' + esc(m[0]) + "</span>";
      ult = re.lastIndex;
    }
    return out + esc(src.slice(ult));
  }
  (function () {
    var ver = $("#cod-ver"), pre = $("#cod-pre"), visor = $("#visor"), rango = $("#cod-rango");
    var tabs = $$(".tabs button"), ind = $(".tabs__ind");
    var lineas = (window.CODIGO_GAN || "").replace(/\n$/, "").split("\n");
    ver.innerHTML = lineas.map(function (l, i) { return '<span class="ln" id="L' + (i + 1) + '"><i>' + (i + 1) + "</i>" + resaltar(l) + "</span>"; }).join("");
    function moverIndicador() {
      var t = tabs.filter(function (b) { return b.getAttribute("aria-selected") === "true"; })[0];
      if (!t) { ind.style.width = "0"; return; }
      ind.style.width = t.offsetWidth + "px";
      ind.style.transform = "translateX(" + t.offsetLeft + "px)";
    }
    function seleccionar(desde, hasta, bajarPagina) {
      $$(".ln.sel", ver).forEach(function (e) { e.classList.remove("sel"); });
      for (var i = desde; i <= hasta; i++) { var l = $("#L" + i); if (l) l.classList.add("sel"); }
      rango.textContent = desde === hasta ? "línea " + desde : "líneas " + desde + "–" + hasta;
      tabs.forEach(function (t) { t.setAttribute("aria-selected", String(+t.dataset.desde === desde)); });
      moverIndicador();
      var primera = $("#L" + Math.max(1, desde - 2));
      if (primera) pre.scrollTo({ top: primera.offsetTop - 16, behavior: bajarPagina && !reduce ? "smooth" : "auto" });
      if (bajarPagina) visor.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "center" });
    }
    $$('a[href="#visor"][data-desde], .tabs button').forEach(function (el) {
      el.addEventListener("click", function (ev) { ev.preventDefault(); seleccionar(+el.dataset.desde, +el.dataset.hasta, el.tagName === "A"); });
    });
    addEventListener("resize", moverIndicador);
    if (document.fonts) document.fonts.ready.then(moverIndicador);
    seleccionar(242, 274, false);
  })();

  document.documentElement.dataset.ready = "1";
})();
