/*
 * animaciones.js: las animaciones de la primera página (plantilla scroll-cinematic-landing).
 * - Portada: la pared de gatos crece de tarjeta a pantalla completa; el texto se va hacia arriba
 *   y aparece la frase final.
 * - Escenas (01, 02 y 05): la imagen se acerca, el panel sube desde abajo y la cubre, y su texto
 *   se enciende palabra por palabra según el scroll. En la 05 el scroll recorre las 25 épocas.
 * - Títulos que suben palabra por palabra desde una máscara, bloques que aparecen al entrar y
 *   contadores; todo se repite al volver.
 * Las lecciones de la plantilla se mantienen: curva casi lineal en los extremos, seguimiento del
 * scroll con factor 0.32, nada de smooth-scroll propio y observadores que nunca se desconectan.
 */
(function () {
  "use strict";
  var reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  var vivo = function () { return !reduce && innerWidth >= 1024; };
  var vh = function () { return innerHeight; };
  var clamp = function (v, a, b) { return Math.min(b, Math.max(a, v)); };
  // curva casi lineal en los extremos: al invertir el scroll el movimiento responde de inmediato
  var ease = function (t) { var s = t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; return t + (s - t) * 0.4; };
  var easeOut = function (t) { return 1 - Math.pow(1 - t, 4); };
  var miles = function (n) { return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, " "); };

  // divide el texto en palabras (conserva etiquetas internas como <b> o <code>)
  function split(el) {
    var out = [];
    (function walk(n) {
      [].slice.call(n.childNodes).forEach(function (c) {
        if (c.nodeType === 3) {
          var frag = document.createDocumentFragment();
          c.textContent.split(/(\s+)/).forEach(function (tok) {
            if (!tok) return;
            if (/^\s+$/.test(tok)) { frag.appendChild(document.createTextNode(" ")); return; }
            var w = document.createElement("span"), i = document.createElement("span");
            w.className = "w"; i.textContent = tok; w.appendChild(i); frag.appendChild(w); out.push(i);
          });
          n.replaceChild(frag, c);
        } else if (c.nodeType === 1) { walk(c); }
      });
    })(el);
    return out;
  }

  // progreso 0..1 de una sección alta con contenido sticky, sobre la posición suavizada
  var cur = scrollY;
  function prog(el) {
    var r = el.getBoundingClientRect();
    var top = r.top + (scrollY - cur);
    return clamp(-top / (r.height - vh()), 0, 1);
  }

  var hero = document.querySelector(".hero");
  var hMedia = document.getElementById("hero-media"), hCampo = document.getElementById("campo");
  var hCopy = document.getElementById("hero-copy"), hDim = document.getElementById("hero-dim"), hText = hero.querySelector(".reveal2");
  var scenes = [].slice.call(document.querySelectorAll("[data-scene]"));
  scenes.forEach(function (s) {
    s._w = [];
    [].forEach.call(s.querySelectorAll(".panel h2, .panel h3, .panel p, .panel li"), function (e) { s._w = s._w.concat(split(e)); });
  });

  function frame() {
    var v = vivo();

    // Portada: la tarjeta de gatos se expande y el texto se desvanece hacia arriba
    var c = window.HERO_CARD || { t: 0, r: 0, b: 0, l: 0 };
    var p = v ? prog(hero) : 0;
    var e = ease(clamp(p / 0.55, 0, 1));
    hMedia.style.clipPath = "inset(" + c.t * (1 - e) + "px " + c.r * (1 - e) + "px " + c.b * (1 - e) + "px " + c.l * (1 - e) + "px round " + 28 * (1 - e) + "px)";
    hCampo.style.transform = "scale(" + (1.1 - 0.1 * e) + ")";
    var fo = clamp(p / 0.3, 0, 1);
    hCopy.style.opacity = 1 - fo;
    hCopy.style.transform = fo ? "translateY(" + (-60 * fo) + "px) scale(" + (1 - 0.06 * fo) + ")" : "";
    hCopy.style.visibility = fo >= 1 ? "hidden" : "";
    var ti = clamp((p - 0.62) / 0.25, 0, 1);
    hText.style.opacity = ti;
    hText.style.transform = "translateY(" + (40 * (1 - ti)) + "px)";
    hDim.style.opacity = ti;

    // Escenas: el panel sube desde abajo y cubre la imagen
    scenes.forEach(function (s) {
      var m = s.querySelector(".media"), t = s.querySelector(".title"), pn = s.querySelector(".panel");
      var epocas = s.hasAttribute("data-epocas") && window.mostrarEpoca;
      if (!v) {   // tableta, celular o movimiento reducido: la composición final, sin movimiento
        m.style.transform = pn.style.transform = t.style.transform = t.style.opacity = "";
        s._w.forEach(function (w) { w.style.opacity = ""; w.style.transform = ""; });
        if (epocas) window.mostrarEpoca(25);
        return;
      }
      var q = prog(s);
      m.style.transform = "scale(" + (1.05 - 0.05 * ease(clamp(q / 0.4, 0, 1))) + ")";
      var pe = ease(clamp((q - 0.08) / 0.5, 0, 1));
      pn.style.transform = "translateY(" + (100 * (1 - pe)) + "%)";
      // el texto del panel se enciende palabra por palabra según el scroll
      var N = s._w.length, rv = clamp((q - 0.3) / 0.55, 0, 1) * (N + 6);
      s._w.forEach(function (w, k) { var o = clamp(rv - k, 0, 6) / 6; w.style.opacity = o; w.style.transform = "translateY(" + ((1 - o) * 14) + "px)"; });
      // la escena de valores recorre las 25 épocas
      if (epocas) window.mostrarEpoca(1 + Math.round(clamp((q - 0.04) / 0.82, 0, 1) * 24));
      var tf = clamp((q - 0.25) / 0.35, 0, 1);
      t.style.opacity = 1 - 0.85 * tf;
      t.style.transform = "translateY(" + (-50 * tf) + "px)";
    });
  }
  window.redibujarAnimaciones = frame;

  // seguimiento rápido del scroll (factor 0.32): quita el efecto escalón de la rueda sin reemplazar el scroll nativo
  var ticking = false;
  function tick() {
    var d = scrollY - cur;
    if (!vivo() || Math.abs(d) < 0.3) { cur = scrollY; frame(); ticking = false; return; }
    cur += d * 0.32;
    frame();
    requestAnimationFrame(tick);
  }
  function onScroll() { if (!ticking) { ticking = true; requestAnimationFrame(tick); } }
  addEventListener("scroll", onScroll, { passive: true });
  addEventListener("resize", onScroll);
  frame();

  // aparición de bloques: se activa al entrar y se reinicia al salir, para que se repita al volver
  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (en) {
      if (!en.isIntersecting) { en.target.classList.remove("in"); return; }
      if (en.intersectionRatio >= .2) en.target.classList.add("in");
    });
  }, { threshold: [0, .2] });
  [].forEach.call(document.querySelectorAll(".hero h1, .hero__lead, .hero__note"), function (e) { e.classList.add("rv"); });
  // títulos y entradas: cada palabra sube desde una máscara, con un escalonado máximo de ~700 ms por bloque
  [].forEach.call(document.querySelectorAll(".hero h1, .hero__lead, main > section:not(.scene) .head h2, main > section:not(.scene) .head .lead, .cierre__nombre"), function (e) {
    var ws = split(e), st = Math.min(45, 700 / ws.length);
    ws.forEach(function (w, k) { w.style.transitionDelay = (k * st) + "ms"; });
    e.classList.add("sp");
  });
  [].forEach.call(document.querySelectorAll(".rv"), function (n) { io.observe(n); });

  // contadores: cuentan al entrar y vuelven a 0 al salir
  var contados = new IntersectionObserver(function (entries) {
    entries.forEach(function (en) {
      var el = en.target, fin = +el.dataset.count, t0 = null;
      var run = el._run = (el._run || 0) + 1;
      if (!en.isIntersecting) { el.textContent = reduce ? miles(fin) : 0; return; }
      if (reduce) { el.textContent = miles(fin); return; }
      (function paso(ts) {
        if (run !== el._run) return;
        if (t0 === null) t0 = ts;
        var k = clamp((ts - t0) / 1400, 0, 1);
        el.textContent = miles(Math.round(fin * easeOut(k)));
        if (k < 1) requestAnimationFrame(paso);
      })(performance.now());
    });
  }, { threshold: .6 });
  [].forEach.call(document.querySelectorAll("[data-count]"), function (n) { contados.observe(n); });
})();
