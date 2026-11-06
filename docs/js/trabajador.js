/*
 * trabajador.js: Web Worker que ejecuta las redes en segundo plano,
 * para que generar gatos no trabe el scroll de la página.
 * Los pedidos "alta" (lo que el usuario acaba de tocar) pasan al frente de la cola.
 */
importScripts("../modelo/pesos.js", "red.js");

let cola = [];
let corriendo = false;

onmessage = (ev) => {
  const m = ev.data;
  if (m.tipo === "cancelar") {
    cola = cola.filter((t) => {
      if (t.grupo !== m.grupo) return true;
      postMessage({ id: t.id, cancelado: true });
      return false;
    });
    return;
  }
  if (m.alta) cola.unshift(m);
  else cola.push(m);
  if (!corriendo) {
    corriendo = true;
    setTimeout(paso, 0);
  }
};

function paso() {
  const m = cola.shift();
  if (m) {
    const r = GAN.ejecutar(m);
    const transferir = [r.img, r.capa1, r.capa2].filter(Boolean).map((a) => a.buffer);
    postMessage(r, transferir);
  }
  if (cola.length) setTimeout(paso, 0);
  else corriendo = false;
}
