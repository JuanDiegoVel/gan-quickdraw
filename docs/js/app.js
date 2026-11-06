/*
 * app.js: contenido vivo de la página.
 * El sistema de animaciones por scroll está en index.html (plantilla scroll-cinematic-landing);
 * aquí se conectan los dibujos, las gráficas y el código con el modelo entrenado.
 */
(function () {
  "use strict";
  var reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return [].slice.call((r || document).querySelectorAll(s)); };
  var f = function (v, d) { return v.toFixed(d == null ? 2 : d); };
  var DATOS = window.DATOS_GAN;
  var HIST = DATOS.historial;

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

  // ------------------------------------------------- motor (Web Worker o hilo principal)
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
      generar: function (z, o) {
        o = o || {};
        return pedir({ tipo: "generar", z: z, intermedios: !!o.intermedios, juzgar: !!o.juzgar }, o.alta);
      },
      juzgar: function (img, o) {
        o = o || {};
        return pedir({ tipo: "juzgar", img: img, intermedios: !!o.intermedios }, o.alta);
      },
    };
  })();

  var azar = GAN.aleatorio((Date.now() ^ 0x9e3779b9) >>> 0);
  function nuevoZ() { return GAN.ruido(azar); }

  // ------------------------------------------------------------- pintar dibujos
  // Trazo claro sobre fondo transparente: el fondo lo pone el CSS de cada lienzo
  var c28 = document.createElement("canvas"); c28.width = c28.height = 28;
  var x28 = c28.getContext("2d"), datos28 = x28.createImageData(28, 28);
  function pintar(lienzo, img) {
    var u8 = img instanceof Uint8Array, p = datos28.data;
    for (var i = 0; i < 784; i++) {
      var t = u8 ? img[i] / 255 : (img[i] + 1) / 2;
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      p[4 * i] = 245; p[4 * i + 1] = 245; p[4 * i + 2] = 247; p[4 * i + 3] = t * 255;
    }
    x28.putImageData(datos28, 0, 0);
    var ctx = lienzo.getContext("2d");
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
    ctx.clearRect(0, 0, lienzo.width, lienzo.height);
    ctx.drawImage(c28, 0, 0, lienzo.width, lienzo.height);
  }
  function lienzos(cont, n, tam, clase) {
    var lista = [];
    for (var i = 0; i < n; i++) {
      var c = document.createElement("canvas");
      c.width = c.height = tam;
      if (clase) c.className = clase;
      cont.appendChild(c); lista.push(c);
    }
    return lista;
  }

  // Mapas de activación: de casi negro al color de la red (azul G, naranja D)
  var COLOR = { g: [41, 151, 255], d: [255, 159, 10] };
  var cm = document.createElement("canvas"), xm = cm.getContext("2d");
  function pintarMapa(lienzo, datos, inicio, n, max, color) {
    cm.width = cm.height = n;
    var id = xm.createImageData(n, n), p = id.data;
    for (var i = 0; i < n * n; i++) {
      var t = Math.pow(Math.min(1, Math.max(0, datos[inicio + i] / max)), 0.75);
      var a = Math.min(1, t * 1.25), b = Math.max(0, (t - 0.8) / 0.2) * 0.55;
      for (var c = 0; c < 3; c++) {
        var v = 11 + (color[c] - 11) * a;
        p[4 * i + c] = v + (255 - v) * b;
      }
      p[4 * i + 3] = 255;
    }
    xm.putImageData(id, 0, 0);
    var ctx = lienzo.getContext("2d");
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(cm, 0, 0, lienzo.width, lienzo.height);
  }
  // De los 64 o 128 mapas de cada capa se muestran los 16 más activos
  function masActivos(datos, canales, area, k) {
    var medias = [];
    for (var c = 0; c < canales; c++) {
      var s = 0;
      for (var i = 0; i < area; i++) s += datos[c * area + i];
      medias.push([s / area, c]);
    }
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

  // Repite una función mientras el elemento está en pantalla
  function mientrasVisible(el, ms, fn) {
    var t = null, hecho = false;
    new IntersectionObserver(function (es) {
      if (es[0].isIntersecting) {
        if (!hecho || !reduce) fn();
        hecho = true;
        if (!reduce && !t) t = setInterval(function () { if (document.visibilityState === "visible") fn(); }, ms);
      } else if (t) { clearInterval(t); t = null; }
    }, { rootMargin: "150px 0px" }).observe(el);
  }

  // ------------------------------------------------- portada: muro de gatos de la red Generadora
  (function () {
    var muro = $("#muro"), hero = $(".hero"), cols = 12;
    var filas = Math.ceil(innerHeight / (innerWidth / cols)) + 1;
    var celdas = lienzos(muro, cols * filas, 112), pendientes = 0;
    muro.style.setProperty("--cols", cols);
    // Se llenan del centro hacia afuera: primero lo que se ve en la portada
    var cx = (cols - 1) / 2, cy = (filas - 1) / 2;
    celdas.map(function (_, i) {
      return [Math.hypot(i % cols - cx, (Math.floor(i / cols) - cy) * 1.2) + Math.random() * 0.9, i];
    }).sort(function (a, b) { return a[0] - b[0]; }).forEach(function (o) {
      pendientes++;
      Motor.generar(nuevoZ()).then(function (r) { pintar(celdas[o[1]], r.img); celdas[o[1]].classList.add("on"); pendientes--; });
    });
    if (reduce) return;
    setInterval(function () {
      if (pendientes > 0 || document.visibilityState !== "visible" || hero.getBoundingClientRect().bottom < 0) return;
      var c = celdas[Math.floor(Math.random() * celdas.length)];
      pendientes++;
      c.classList.remove("on");
      Motor.generar(nuevoZ()).then(function (r) {
        setTimeout(function () { pintar(c, r.img); c.classList.add("on"); pendientes--; }, 450);
      });
    }, 900);
  })();

  // ------------------------------------------------- 1. dentro de la red Generadora
  (function () {
    var zsvg = $("#vg-z"), c1 = lienzos($("#vg-c1"), 16, 28), c2 = lienzos($("#vg-c2"), 16, 56), out = $("#vg-out");
    var sel = null;
    mientrasVisible($("#generadora"), 3500, function () {
      var z = nuevoZ();
      Motor.generar(z, { intermedios: true, alta: true }).then(function (r) {
        if (!sel) sel = { c1: masActivos(r.capa1, 128, 49, 16), c2: masActivos(r.capa2, 64, 196, 16) };
        dibujarZ(zsvg, z);
        pintarCapa(c1, r.capa1, sel.c1, 7, COLOR.g);
        pintarCapa(c2, r.capa2, sel.c2, 14, COLOR.g);
        pintar(out, r.img);
      });
    });
  })();

  // ------------------------------------------------- 2. dentro de la red Discriminadora
  (function () {
    var cin = $("#vd-in"), tipo = $("#vd-tipo"), c1 = lienzos($("#vd-c1"), 16, 56), c2 = lienzos($("#vd-c2"), 16, 28);
    var marca = $("#vd-medidor .marca"), prob = $("#vd-prob"), sel = null, turnoReal = true;
    mientrasVisible($("#discriminadora"), 3500, function () {
      var esReal = turnoReal; turnoReal = !turnoReal;
      var k = Math.floor(Math.random() * N_REALES);
      var entrada = esReal ? Promise.resolve(realTanh(k))
        : Motor.generar(nuevoZ(), { alta: true }).then(function (r) { return r.img; });
      entrada.then(function (img) {
        return Motor.juzgar(img, { intermedios: true, alta: true }).then(function (r) {
          if (!sel) sel = { c1: masActivos(r.capa1, 64, 196, 16), c2: masActivos(r.capa2, 128, 49, 16) };
          pintar(cin, esReal ? real(k) : img);
          tipo.innerHTML = esReal ? 'Entrada<small><span class="etq d">real</span> · Quick Draw</small>'
            : 'Entrada<small><span class="etq g">generada</span> por G</small>';
          pintarCapa(c1, r.capa1, sel.c1, 14, COLOR.d);
          pintarCapa(c2, r.capa2, sel.c2, 7, COLOR.d);
          marca.style.left = (r.prob * 100) + "%";
          prob.innerHTML = f(r.prob) + '<small class="etq ' + (r.prob >= 0.5 ? "d" : "g") + '">' +
            (r.prob >= 0.5 ? "D dice: real" : "D dice: falso") + "</small>";
        });
      });
    });
  })();

  // ------------------------------------------------------------------ gráficas
  var tooltip = $("#tooltip"), idGraf = 0;
  function grafica(cont, cfg) {
    var clip = "rec" + (++idGraf);
    function dibujar() {
      var W = cont.clientWidth, H = cont.clientHeight, m = { l: 42, r: cfg.derecha || 58, t: 12, b: 30 };
      if (!W) return;
      var X = function (v) { return m.l + (v - cfg.x[0]) / (cfg.x[1] - cfg.x[0]) * (W - m.l - m.r); };
      var Y = function (v) { return m.t + (1 - (v - cfg.y[0]) / (cfg.y[1] - cfg.y[0])) * (H - m.t - m.b); };
      var s = '<defs><clipPath id="' + clip + '"><rect x="' + m.l + '" y="' + m.t + '" width="' + (W - m.l - m.r) + '" height="' + (H - m.t - m.b) + '"/></clipPath></defs>';
      cfg.yTicks.forEach(function (v) {
        s += '<line class="rej" x1="' + m.l + '" x2="' + (W - m.r) + '" y1="' + Y(v) + '" y2="' + Y(v) + '"/>' +
          '<text class="eje" x="' + (m.l - 10) + '" y="' + (Y(v) + 4) + '" text-anchor="end">' + cfg.fmtY(v) + "</text>";
      });
      cfg.xTicks.forEach(function (v) {
        s += '<text class="eje" x="' + X(v) + '" y="' + (H - 6) + '" text-anchor="middle">' + cfg.fmtX(v) + "</text>";
      });
      if (cfg.ref != null) {
        s += '<line class="ref" x1="' + m.l + '" x2="' + (W - m.r) + '" y1="' + Y(cfg.ref) + '" y2="' + Y(cfg.ref) + '"/>';
        if (cfg.refTexto) s += '<text class="ref-t" x="' + (W - m.r) + '" y="' + (Y(cfg.ref) - 8) + '" text-anchor="end">' + cfg.refTexto + "</text>";
      }
      s += '<g clip-path="url(#' + clip + ')">';
      cfg.series.forEach(function (se) {
        s += '<path class="s ' + se.color + (se.punteada ? " pt" : "") + '" d="M' +
          se.puntos.map(function (p) { return X(p[0]).toFixed(1) + "," + Y(Math.max(cfg.y[0] - 1, Math.min(cfg.y[1] + 1, p[1]))).toFixed(1); }).join("L") + '"/>';
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
        var mejor = cfg.series[0].puntos.map(function (p) { return p[0]; })
          .reduce(function (a, b) { return Math.abs(X(b) - x) < Math.abs(X(a) - x) ? b : a; });
        g.style.display = "";
        linea.setAttribute("x1", X(mejor)); linea.setAttribute("x2", X(mejor));
        tooltip.innerHTML = "<b>Época " + mejor + "</b>" + cfg.series.map(function (se, k) {
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

  // 3. Funciones de pérdida según la salida del discriminador
  (function () {
    var ps = [];
    for (var i = 0; i <= 200; i++) ps.push(0.004 + i * (0.992 / 200));
    var curva = function (fn) { return ps.map(function (p) { return [p, fn(p)]; }); };
    var ticksP = [0, 0.25, 0.5, 0.75, 1], fmtP = function (v) { return f(v); };
    grafica($("#graf-ld"), {
      titulo: "Pérdida del discriminador", x: [0, 1], y: [0, 5.5], xTicks: ticksP, yTicks: [0, 1, 2, 3, 4, 5],
      fmtX: fmtP, fmtY: String, derecha: 16,
      series: [
        { color: "d", puntos: curva(function (p) { return -Math.log(p); }) },
        { color: "d", punteada: true, puntos: curva(function (p) { return -Math.log(1 - p); }) },
      ],
    });
    grafica($("#graf-lg"), {
      titulo: "Pérdida del generador", x: [0, 1], y: [-5.5, 5.5], xTicks: ticksP, yTicks: [-4, -2, 0, 2, 4],
      fmtX: fmtP, fmtY: String, derecha: 16,
      series: [
        { color: "g", puntos: curva(function (p) { return -Math.log(p); }) },
        { color: "g", punteada: true, puntos: curva(function (p) { return Math.log(1 - p); }) },
      ],
    });
  })();

  // 5. Valores de G y D por época
  function serieEpocas(nombre, clave, color) {
    return {
      nombre: nombre, color: color,
      puntos: HIST.map(function (h) { return [h.epoca, h[clave]]; }),
      valor: function (e) { return HIST[e - 1][clave]; },
    };
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
  $("#tabla-epocas tbody").innerHTML = HIST.map(function (h) {
    return "<tr><td>" + h.epoca + '</td><td class="n">' + f(h.perdida_D, 4) + '</td><td class="n">' + f(h.perdida_G, 4) +
      '</td><td class="n">' + f(h.D_x, 3) + '</td><td class="n">' + f(h.D_G_z, 3) + "</td></tr>";
  }).join("");

  // Escena de valores: el scroll recorre las 25 épocas
  (function () {
    var celdas = lienzos($("#tira-celdas"), POR_EPOCA, 112, "dib"), actual = 0;
    var num = $("#tira-ep"), barra = $("#tira-barra"), epNum = $("#ep-num"), epDx = $("#ep-dx"), epDgz = $("#ep-dgz");
    window.mostrarEpoca = function (e) {
      if (e === actual) return;
      actual = e;
      celdas.forEach(function (c, k) { pintar(c, deEpoca(e, k)); });
      var h = HIST[e - 1];
      num.textContent = e; epNum.textContent = e;
      barra.style.width = (e / 25 * 100) + "%";
      epDx.textContent = f(h.D_x); epDgz.textContent = f(h.D_G_z);
    };
    if (window.redibujarEscenas) window.redibujarEscenas();
  })();

  // 6. Resultado: reales de Quick Draw y generados en la época 25 (los mismos del notebook)
  lienzos($("#rej-reales"), 24, 112, "dib").forEach(function (c, k) { pintar(c, real(k)); });
  lienzos($("#rej-generados"), 24, 112, "dib").forEach(function (c, k) { pintar(c, deEpoca(25, k)); });

  // ------------------------------------------------- 4. código: resaltado y enlaces al repositorio
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
  // En GitHub Pages (usuario.github.io/repo) se ofrece además abrir el archivo en GitHub
  var REPO = (function () {
    var h = location.hostname;
    if (!/\.github\.io$/.test(h)) return null;
    var r = location.pathname.split("/").filter(Boolean)[0];
    return r && !/\.html?$/.test(r) ? "https://github.com/" + h.split(".")[0] + "/" + r : null;
  })();

  // Visor: gan_quickdraw.py completo con números de línea (no depende de ningún enlace externo)
  (function () {
    var ver = $("#cod-ver"), pre = $("#cod-pre"), visor = $("#visor"), rango = $("#cod-rango"), gh = $("#cod-github");
    var tabs = $$(".visor .segmento button");
    var lineas = (window.CODIGO_GAN || "").replace(/\n$/, "").split("\n");
    ver.innerHTML = lineas.map(function (l, i) {
      return '<span class="ln" id="L' + (i + 1) + '"><i>' + (i + 1) + "</i>" + resaltar(l) + "</span>";
    }).join("");
    function seleccionar(desde, hasta, bajarPagina) {
      $$(".ln.sel", ver).forEach(function (e) { e.classList.remove("sel"); });
      for (var i = desde; i <= hasta; i++) { var l = $("#L" + i); if (l) l.classList.add("sel"); }
      rango.textContent = desde === hasta ? "línea " + desde : "líneas " + desde + "–" + hasta;
      tabs.forEach(function (t) { t.setAttribute("aria-selected", String(+t.dataset.desde === desde)); });
      if (REPO) { gh.href = REPO + "/blob/main/gan_quickdraw.py#L" + desde + (hasta > desde ? "-L" + hasta : ""); gh.hidden = false; }
      var primera = $("#L" + Math.max(1, desde - 2));
      if (primera) pre.scrollTo({ top: primera.offsetTop - 18, behavior: bajarPagina && !reduce ? "smooth" : "auto" });
      if (bajarPagina) visor.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
    }
    $$('a[href="#visor"][data-desde], .visor .segmento button').forEach(function (el) {
      el.addEventListener("click", function (ev) {
        ev.preventDefault();
        seleccionar(+el.dataset.desde, +el.dataset.hasta, el.tagName === "A");
      });
    });
    seleccionar(242, 274, false);
  })();
})();
