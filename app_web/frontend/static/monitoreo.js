"use strict";

/* ============================================================
   Monitoreo IoT: muestra en tiempo real los segmentos ECG que envía
   un dispositivo (simulado) y el resultado del modelo para cada uno.
   Usa utilidades de comun.js ($, css, fmt, api, colorNivel) y
   prepararCanvas / estado de app.js.
   ============================================================ */

const monitoreo = {
  dispositivos: [],
  seleccionado: null,
  datos: null,
  temporizador: null,
};

const INTERVALO_MONITOREO_MS = 3000;

function vistaMonitoreoActiva() {
  return $("vista-monitoreo").classList.contains("activa");
}

document.querySelectorAll(".pestana").forEach((boton) =>
  boton.addEventListener("click", () => {
    if (boton.dataset.vista === "monitoreo") iniciarMonitoreo();
    else detenerMonitoreo();
  })
);

$("mon-dispositivo").addEventListener("change", (e) => {
  monitoreo.seleccionado = Number(e.target.value);
  monitoreo.datos = null;
  actualizarMonitoreo();
});

window.addEventListener("resize", () => {
  if (vistaMonitoreoActiva() && monitoreo.datos) dibujarMonitoreo();
});

async function iniciarMonitoreo() {
  detenerMonitoreo();
  try {
    monitoreo.dispositivos = await api("/api/monitoreo/dispositivos");
  } catch {
    monitoreo.dispositivos = [];
  }

  const selector = $("mon-dispositivo");
  selector.innerHTML = monitoreo.dispositivos
    .map((d) => `<option value="${d.id}">${escapar(d.codigo)} · ${escapar(d.nombre)}</option>`)
    .join("");
  selector.hidden = monitoreo.dispositivos.length === 0;

  if (!monitoreo.dispositivos.length) {
    mostrarVacioMonitoreo(
      "Sin dispositivos asignados",
      "El administrador debe registrar un dispositivo y asignarlo a tu cuenta. Cuando envíe segmentos ECG, aparecerán aquí."
    );
    pintarEstado(false, "Sin dispositivo");
    return;
  }

  if (!monitoreo.dispositivos.some((d) => d.id === monitoreo.seleccionado)) {
    monitoreo.seleccionado = monitoreo.dispositivos[0].id;
  }
  selector.value = String(monitoreo.seleccionado);

  await actualizarMonitoreo();
  monitoreo.temporizador = setInterval(actualizarMonitoreo, INTERVALO_MONITOREO_MS);
}

function detenerMonitoreo() {
  if (monitoreo.temporizador) clearInterval(monitoreo.temporizador);
  monitoreo.temporizador = null;
}

async function actualizarMonitoreo() {
  if (!monitoreo.seleccionado) return;
  let datos;
  try {
    datos = await api(`/api/monitoreo/dispositivos/${monitoreo.seleccionado}`);
  } catch {
    return;
  }
  monitoreo.datos = datos;

  const s = datos.sesion;
  const enCurso = s && !s.finalizada;
  pintarEstado(
    datos.dispositivo.en_linea && enCurso,
    datos.dispositivo.en_linea && enCurso ? "Recibiendo datos" : s && s.finalizada ? "Sesión finalizada" : "Desconectado"
  );

  if (!s || !s.lecturas.length) {
    mostrarVacioMonitoreo(
      "Esperando datos del dispositivo",
      `El dispositivo ${datos.dispositivo.codigo} aún no ha enviado segmentos ECG.`
    );
    return;
  }

  $("mon-vacio").hidden = true;
  $("mon-panel").hidden = false;

  const ultima = s.lecturas[s.lecturas.length - 1];
  $("mon-ultimo").innerHTML = `${fmt(ultima.probabilidad, 3)} <small style="color:${colorNivel(ultima.nivel)}">${ultima.nivel}</small>`;
  $("mon-promedio").innerHTML = `${fmt(s.promedio, 3)} <small style="color:${colorNivel(s.nivel)}">${s.nivel}</small>`;
  $("mon-kpi-sesion").className = `kpi kpi--${s.nivel}`;
  $("mon-n").textContent = s.lecturas.length;
  const segundos = s.lecturas.length * 10;
  $("mon-duracion").textContent = segundos >= 60 ? `${Math.floor(segundos / 60)} min ${segundos % 60} s` : `${segundos} s`;
  $("mon-sesion").textContent = `Sesión ${s.sesion}`;
  $("mon-pie-sesion").textContent = s.finalizada
    ? `Sesión finalizada el ${fechaCorta(s.finalizada)} y guardada en "Mis análisis" (análisis #${s.analisis_id}).`
    : "Sesión en curso · el nivel de la sesión es el promedio de todos los segmentos recibidos.";

  dibujarMonitoreo();
}

function mostrarVacioMonitoreo(titulo, texto) {
  $("mon-panel").hidden = true;
  $("mon-vacio").hidden = false;
  $("mon-vacio-titulo").textContent = titulo;
  $("mon-vacio-texto").textContent = texto;
}

function pintarEstado(activo, texto) {
  const chip = $("mon-estado");
  chip.textContent = texto;
  chip.classList.toggle("estado-disp--activo", activo);
}

function dibujarMonitoreo() {
  const s = monitoreo.datos && monitoreo.datos.sesion;
  if (!s || !s.ultima || $("mon-panel").hidden) return;
  dibujarSegmentoVivo(s.ultima.muestras, s.ultima.indice, s.lecturas);
  dibujarBarrasMonitoreo(s.lecturas);
}

function dibujarSegmentoVivo(datos, indice, lecturas) {
  const lectura = lecturas.find((l) => l.indice === indice);
  $("mon-seg-etiqueta").textContent = `Segmento ${indice + 1} · ${indice * 10}–${(indice + 1) * 10} s`;
  $("mon-prob").textContent = lectura ? `prob. ${fmt(lectura.probabilidad, 4)} · ${lectura.nivel}` : "";

  const { ctx, ancho, alto } = prepararCanvas($("mon-ecg"));
  const izq = 12, der = 12, inf = 20, h = alto - inf, w = ancho - izq - der, L = datos.length;
  ctx.clearRect(0, 0, ancho, alto);

  for (let t = 0; t <= 10.0001; t += 0.2) {
    const x = izq + (t / 10) * w;
    const fuerte = Math.abs(t - Math.round(t)) < 1e-6;
    ctx.strokeStyle = fuerte ? css("--monitor-rejilla-fuerte") : css("--monitor-rejilla");
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    if (fuerte) {
      ctx.fillStyle = "rgba(127, 231, 174, .6)";
      ctx.font = "10px " + css("--mono");
      ctx.textAlign = t === 0 ? "left" : t >= 10 ? "right" : "center";
      ctx.fillText(`${(indice * 10 + t).toFixed(0)}s`, x, alto - 5);
    }
  }
  for (let k = 0; k <= 10; k++) {
    const y = (k / 10) * h;
    ctx.strokeStyle = k % 5 === 0 ? css("--monitor-rejilla-fuerte") : css("--monitor-rejilla");
    ctx.beginPath(); ctx.moveTo(izq, y); ctx.lineTo(ancho - der, y); ctx.stroke();
  }

  const ordenados = [...datos].sort((a, b) => a - b);
  let min = ordenados[Math.floor(L * 0.002)], max = ordenados[Math.ceil(L * 0.998) - 1];
  if (max - min < 1e-9) { min -= 1; max += 1; }
  const pad = (max - min) * 0.12; min -= pad; max += pad;

  const traza = () => {
    ctx.beginPath();
    datos.forEach((v, k) => {
      const x = izq + (k / (L - 1)) * w;
      const y = h - ((Math.min(Math.max(v, min), max) - min) / (max - min)) * h;
      k === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
  };
  ctx.lineJoin = "round";
  ctx.strokeStyle = "rgba(60, 240, 143, .25)"; ctx.lineWidth = 5; traza(); ctx.stroke();
  ctx.strokeStyle = css("--monitor-linea"); ctx.lineWidth = 1.6;
  ctx.shadowColor = "rgba(60, 240, 143, .8)"; ctx.shadowBlur = 6; traza(); ctx.stroke(); ctx.shadowBlur = 0;
}

function dibujarBarrasMonitoreo(lecturas) {
  const niveles = (typeof estado !== "undefined" && estado.modelo) ? estado.modelo.niveles : [];
  const { ctx, ancho, alto } = prepararCanvas($("mon-barras"));
  const izq = 38, der = 10, sup = 12, inf = 22, w = ancho - izq - der, h = alto - sup - inf;
  const y = (p) => sup + (1 - p) * h;
  ctx.clearRect(0, 0, ancho, alto);
  ctx.font = "11px " + css("--fuente");

  niveles.forEach((n) => {
    ctx.fillStyle = css(`--${n.nivel.toLowerCase()}-suave`);
    ctx.fillRect(izq, y(n.hasta), w, y(n.desde) - y(n.hasta));
  });
  [0, 0.25, 0.5, 0.75, 1].forEach((p) => {
    ctx.fillStyle = css("--texto-tenue"); ctx.textAlign = "right";
    ctx.fillText(p.toFixed(2), izq - 8, y(p) + 4);
  });

  // Al menos 30 posiciones para que las primeras barras no ocupen todo el ancho.
  const posiciones = Math.max(lecturas.length, 30);
  const paso = w / posiciones;
  const anchoBarra = Math.max(Math.min(paso * 0.62, 26), 2);
  lecturas.forEach((l, k) => {
    const x = izq + k * paso + (paso - anchoBarra) / 2;
    const top = y(l.probabilidad);
    ctx.fillStyle = colorNivel(l.nivel);
    ctx.globalAlpha = k === lecturas.length - 1 ? 1 : 0.6;
    ctx.beginPath();
    const r = Math.min(5, anchoBarra / 2);
    if (ctx.roundRect) ctx.roundRect(x, top, anchoBarra, y(0) - top, [r, r, 0, 0]);
    else ctx.rect(x, top, anchoBarra, y(0) - top);
    ctx.fill();
  });
  ctx.globalAlpha = 1;

  const promedio = lecturas.reduce((a, l) => a + l.probabilidad, 0) / lecturas.length;
  ctx.strokeStyle = css("--texto"); ctx.setLineDash([5, 4]); ctx.lineWidth = 1.4;
  ctx.beginPath(); ctx.moveTo(izq, y(promedio)); ctx.lineTo(izq + w, y(promedio)); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = css("--texto"); ctx.textAlign = "right";
  ctx.fillText(`promedio ${promedio.toFixed(2)}`, izq + w - 4, y(promedio) - 6);
  ctx.fillStyle = css("--texto-tenue"); ctx.textAlign = "left";
  ctx.fillText("segmentos recibidos →", izq, alto - 5);
}
