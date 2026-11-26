"use strict";

/* ============================================================
   Estado de la página
   ============================================================ */

const estado = {
  modelo: null,
  ejemplos: [],
  archivo: null,
  ejemploId: null,
  ultimo: null,          // última respuesta de /api/analizar
  segmento: 0,           // segmento mostrado en el monitor ECG
};

// Utilidades ($, css, escapar, fmt, colorNivel, api, avisar) en comun.js

/* ============================================================
   Navegación
   ============================================================ */

document.querySelectorAll(".pestana").forEach((boton) => {
  boton.addEventListener("click", () => {
    document.querySelectorAll(".pestana").forEach((b) => b.classList.remove("activa"));
    document.querySelectorAll(".vista").forEach((v) => v.classList.remove("activa"));
    boton.classList.add("activa");
    $(`vista-${boton.dataset.vista}`).classList.add("activa");
    window.scrollTo({ top: 0 });

    if (boton.dataset.vista === "historial") cargarHistorial();
    if (boton.dataset.vista === "analisis" && estado.ultimo) dibujarGraficas();
  });
});

/* ============================================================
   Indicador de pasos
   ============================================================ */

function actualizarPasos() {
  const hayEntrada =
    (origenActual() === "archivo" && estado.archivo) ||
    (origenActual() === "ejemplo" && estado.ejemploId);
  const hayResultado = Boolean(estado.ultimo);

  const p1 = $("paso-1"), p2 = $("paso-2"), p3 = $("paso-3");
  [p1, p2, p3].forEach((p) => p.classList.remove("activo", "hecho"));

  if (hayResultado) {
    p1.classList.add("hecho"); p2.classList.add("hecho"); p3.classList.add("activo");
  } else if (hayEntrada) {
    p1.classList.add("hecho"); p2.classList.add("activo");
  } else {
    p1.classList.add("activo");
  }
}

/* ============================================================
   Entrada: archivo o registro de ejemplo
   ============================================================ */

const origenActual = () => document.querySelector('input[name="origen"]:checked').value;

document.querySelectorAll('input[name="origen"]').forEach((radio) => {
  radio.addEventListener("change", () => {
    const esArchivo = origenActual() === "archivo";
    $("panel-archivo").hidden = !esArchivo;
    $("panel-ejemplo").hidden = esArchivo;
    ocultarError();
    actualizarPasos();
  });
});

const zona = $("zona-carga");

$("archivo").addEventListener("change", (e) => seleccionarArchivo(e.target.files[0]));

["dragenter", "dragover"].forEach((ev) =>
  zona.addEventListener(ev, (e) => {
    e.preventDefault();
    zona.classList.add("arrastrando");
  })
);

["dragleave", "drop"].forEach((ev) =>
  zona.addEventListener(ev, (e) => {
    e.preventDefault();
    zona.classList.remove("arrastrando");
  })
);

zona.addEventListener("drop", (e) => seleccionarArchivo(e.dataTransfer.files[0]));

const TEXTO_ZONA = $("zona-carga-texto").textContent;
const DETALLE_ZONA = $("zona-carga-detalle").innerHTML;

// Quita el resultado mostrado (pertenece al registro anterior).
function limpiarResultado() {
  estado.ultimo = null;
  estado.segmento = 0;
  $("resultado").hidden = true;
  $("analizando").hidden = true;
  $("sin-resultado").hidden = false;
}

function limpiarDatosRegistro() {
  ["codigo", "edad", "sexo", "peso", "talla"].forEach((campo) => ($(campo).value = ""));
  actualizarIMC();
}

// Botón "Nuevo análisis": deja la página como recién abierta, sin recargar.
function nuevoAnalisis() {
  estado.archivo = null;
  estado.ejemploId = null;
  $("archivo").value = "";
  zona.classList.remove("cargado");
  $("zona-carga-texto").textContent = TEXTO_ZONA;
  $("zona-carga-detalle").innerHTML = DETALLE_ZONA;
  document.querySelectorAll(".ejemplo").forEach((b) => b.classList.remove("seleccionado"));
  limpiarDatosRegistro();
  limpiarResultado();
  ocultarError();
  actualizarPasos();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

$("boton-nuevo").addEventListener("click", nuevoAnalisis);

function seleccionarArchivo(archivo) {
  if (!archivo) return;
  // Otro registro = otra persona: no se conservan los datos ni el resultado
  // del anterior (sí se conservan si es el primer archivo elegido).
  if (estado.archivo || estado.ejemploId || estado.ultimo) limpiarDatosRegistro();
  estado.ejemploId = null;
  document.querySelectorAll(".ejemplo").forEach((b) => b.classList.remove("seleccionado"));
  limpiarResultado();
  estado.archivo = archivo;
  $("archivo").value = "";   // permite volver a elegir el mismo archivo
  zona.classList.add("cargado");
  $("zona-carga-texto").textContent = archivo.name;
  $("zona-carga-detalle").innerHTML =
    `${(archivo.size / 1024).toFixed(0)} KB · <u>cambiar archivo</u>`;
  ocultarError();
  actualizarPasos();
  completarDesdeMat(archivo);
}

// Los .mat de PulseDB traen edad, sexo, peso y talla: se completan los
// campos vacíos (el usuario puede corregirlos antes de analizar).
async function completarDesdeMat(archivo) {
  if (!archivo.name.toLowerCase().endsWith(".mat")) return;

  const datos = new FormData();
  datos.append("archivo", archivo);

  let d;
  try {
    d = await api("/api/leer-demograficos", { method: "POST", body: datos });
  } catch {
    return;
  }
  if (estado.archivo !== archivo) return;

  const completados = ["edad", "sexo", "peso", "talla"].filter(
    (campo) => d[campo] != null && $(campo).value === ""
  );
  completados.forEach((campo) => ($(campo).value = d[campo]));
  if (!completados.length) return;

  actualizarIMC();
  avisar("Datos demográficos completados desde el archivo .mat (puede corregirlos).");
}

async function cargarEjemplos() {
  try {
    estado.ejemplos = await api("/api/ejemplos");
  } catch {
    estado.ejemplos = [];
  }

  const lista = $("lista-ejemplos");

  if (!estado.ejemplos.length) {
    lista.innerHTML =
      '<p class="nota">No hay registros de ejemplo. Ejecuta herramientas/exportar_ejemplos.py.</p>';
    return;
  }

  lista.innerHTML = estado.ejemplos
    .map((e, i) => {
      const d = e.demograficos || {};
      const sexo = d.sexo === "F" ? "Femenino" : d.sexo === "M" ? "Masculino" : "—";
      return `
        <button type="button" class="ejemplo" data-id="${escapar(e.id)}">
          <span class="ejemplo__avatar">${String(i + 1).padStart(2, "0")}</span>
          <span>
            <span class="ejemplo__nombre">${escapar(e.nombre.replace("Registro de prueba", "Registro"))}</span>
            <span class="ejemplo__detalle">${e.duracion_s} s · ${fmt(d.edad, 0)} años · ${sexo}</span>
          </span>
        </button>`;
    })
    .join("");

  lista.querySelectorAll(".ejemplo").forEach((boton) => {
    boton.addEventListener("click", () => {
      lista.querySelectorAll(".ejemplo").forEach((b) => b.classList.remove("seleccionado"));
      boton.classList.add("seleccionado");
      estado.ejemploId = boton.dataset.id;
      limpiarResultado();
      $("codigo").value = "";

      // Completar el formulario con los datos del registro.
      const ej = estado.ejemplos.find((x) => x.id === estado.ejemploId);
      const d = (ej && ej.demograficos) || {};
      $("edad").value = d.edad ?? "";
      $("sexo").value = d.sexo ?? "";
      $("peso").value = d.peso ?? "";
      $("talla").value = d.talla ?? "";
      actualizarIMC();
      ocultarError();
      actualizarPasos();
    });
  });
}

/* ============================================================
   Datos demográficos
   ============================================================ */

function actualizarIMC() {
  const peso = parseFloat($("peso").value);
  const talla = parseFloat($("talla").value);
  $("imc").textContent =
    peso > 0 && talla > 0 ? (peso / (talla / 100) ** 2).toFixed(1) + " kg/m²" : "—";
}

["peso", "talla"].forEach((id) => $(id).addEventListener("input", actualizarIMC));

/* ============================================================
   Envío del análisis
   ============================================================ */

function mostrarError(texto) {
  $("error").textContent = texto;
  $("error").hidden = false;
}

function ocultarError() {
  $("error").hidden = true;
}

$("formulario").addEventListener("submit", async (e) => {
  e.preventDefault();
  ocultarError();

  const datos = new FormData();

  if (origenActual() === "archivo") {
    if (!estado.archivo) return mostrarError("Selecciona un archivo CSV, TXT o MAT con la señal ECG.");
    datos.append("archivo", estado.archivo);
  } else {
    if (!estado.ejemploId) return mostrarError("Selecciona un registro de ejemplo.");
    datos.append("ejemplo_id", estado.ejemploId);
  }

  ["edad", "sexo", "peso", "talla"].forEach((campo) => datos.append(campo, $(campo).value));
  datos.append("codigo_registro", $("codigo").value.trim());

  const boton = $("boton-analizar");
  boton.disabled = true;
  boton.classList.add("ocupado");
  $("sin-resultado").hidden = true;
  $("resultado").hidden = true;
  $("analizando").hidden = false;

  const inicio = performance.now();

  try {
    const cuerpo = await api("/api/analizar", { method: "POST", body: datos });

    // Pequeña pausa para que la animación de análisis sea perceptible.
    const espera = 700 - (performance.now() - inicio);
    if (espera > 0) await new Promise((ok) => setTimeout(ok, espera));

    estado.ultimo = cuerpo;
    estado.segmento = 0;
    mostrarResultado(cuerpo);
  } catch (err) {
    $("analizando").hidden = true;
    if (estado.ultimo) $("resultado").hidden = false;
    else $("sin-resultado").hidden = false;
    mostrarError(err.message);
  } finally {
    boton.disabled = false;
    boton.classList.remove("ocupado");
    actualizarPasos();
  }
});

/* ============================================================
   Medidor semicircular
   ============================================================ */

const CENTRO = 110, RADIO = 86;

function punto(p, r = RADIO) {
  const ang = Math.PI * (1 - p);
  return [CENTRO + r * Math.cos(ang), CENTRO - r * Math.sin(ang)];
}

function arco(p1, p2) {
  const [x1, y1] = punto(p1);
  const [x2, y2] = punto(p2);
  return `M ${x1.toFixed(2)} ${y1.toFixed(2)} A ${RADIO} ${RADIO} 0 0 1 ${x2.toFixed(2)} ${y2.toFixed(2)}`;
}

function dibujarMedidor(prob) {
  const niveles = estado.modelo ? estado.modelo.niveles : [];
  const brecha = 0.006;

  let svg = `<path class="medidor__fondo" d="${arco(0, 1)}"/>`;

  niveles.forEach((n, i) => {
    const a = n.desde + (i ? brecha : 0);
    const b = n.hasta - (i < niveles.length - 1 ? brecha : 0);
    svg += `<path class="medidor__arco" stroke="${colorNivel(n.nivel)}" d="${arco(a, b)}"/>`;
  });

  // Marcas en los cortes.
  [0, ...niveles.map((n) => n.hasta)].forEach((c) => {
    const [x, y] = punto(c, RADIO + 20);
    svg += `<text class="medidor__marca" x="${x.toFixed(1)}" y="${(y + 3).toFixed(1)}" text-anchor="middle">${fmt(c, 2)}</text>`;
  });

  $("medidor-arcos").innerHTML = svg;

  const aguja = $("medidor-aguja");
  aguja.style.transform = "rotate(-90deg)";
  requestAnimationFrame(() =>
    requestAnimationFrame(() => {
      aguja.style.transform = `rotate(${(Math.min(Math.max(prob, 0), 1) - 0.5) * 180}deg)`;
    })
  );
}

function animarNumero(elemento, destino, decimales = 2, duracion = 900) {
  const t0 = performance.now();
  const paso = (t) => {
    const k = Math.min((t - t0) / duracion, 1);
    const suave = 1 - Math.pow(1 - k, 3);
    elemento.textContent = (destino * suave).toFixed(decimales);
    if (k < 1) requestAnimationFrame(paso);
  };
  requestAnimationFrame(paso);
}

/* ============================================================
   Resultado
   ============================================================ */

function mostrarResultado(r) {
  const res = r.resultado;
  const niveles = estado.modelo ? estado.modelo.niveles : [];

  $("analizando").hidden = true;
  $("sin-resultado").hidden = true;
  $("resultado").hidden = false;

  $("tarjeta-alerta").className = `tarjeta alerta nivel-${res.nivel}`;
  $("nivel").textContent = res.nivel;
  $("mensaje").textContent = res.mensaje;
  $("aviso").textContent = res.aviso;

  animarNumero($("probabilidad"), res.probabilidad);
  dibujarMedidor(res.probabilidad);

  $("leyenda-niveles").innerHTML = niveles
    .map(
      (n) =>
        `<span><i style="background:${colorNivel(n.nivel)}"></i>${escapar(n.nivel)} ${fmt(n.desde, 2)}–${fmt(n.hasta, 2)}</span>`
    )
    .join("");

  // Indicadores.
  $("r-segmentos").textContent = res.n_segmentos;
  $("r-duracion").textContent = `${res.duracion_analizada_s} s`;
  $("r-umbral").textContent = fmt(res.umbral, 4);
  $("r-modelo").textContent = r.modelo.version;

  // Referencia del dataset y avisos.
  const notas = [
    `<strong>Código de registro:</strong> <span class="mono">${escapar(r.codigo_registro)}</span> · análisis #${r.id} guardado en tu historial.`,
  ];

  if (r.referencia) {
    notas.push(`
      <strong>Valor de referencia del dataset</strong> (no usado por el modelo):
      SBP promedio ${fmt(r.referencia.sbp_promedio, 1)} mmHg ·
      DBP promedio ${fmt(r.referencia.dbp_promedio, 1)} mmHg ·
      ${r.referencia.segmentos_140_90} de ${r.referencia.segmentos_total} segmentos cumplen el criterio 140/90.`);
  }

  // 1250 muestras = 10 s a 125 Hz. Si sobra más de un segmento, el registro
  // superó el máximo analizado (no son muestras incompletas).
  const MUESTRAS_SEGMENTO = 1250;
  if (res.muestras_descartadas >= MUESTRAS_SEGMENTO) {
    const minutos = (res.muestras_descartadas / MUESTRAS_SEGMENTO / 6).toFixed(0);
    notas.push(
      `<small>El registro supera el máximo analizado (${res.n_segmentos} segmentos = ` +
      `${(res.duracion_analizada_s / 60).toFixed(0)} min): se analizó el inicio y no se ` +
      `analizaron los ~${minutos} min restantes.</small>`
    );
  } else if (res.muestras_descartadas > 0) {
    notas.push(
      `<small>Se descartaron ${res.muestras_descartadas} muestras finales que no completaban un segmento de 10 s.</small>`
    );
  }

  $("referencia").hidden = notas.length === 0;
  $("referencia").innerHTML = notas.join("<br>");

  dibujarGraficas();
  $("resultado").scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ============================================================
   Gráficas (canvas, sin librerías externas)
   ============================================================ */

function prepararCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const ancho = canvas.clientWidth;
  // El alto lógico se guarda la primera vez: asignar canvas.height cambia el
  // atributo, y con zoom distinto de 100 % la gráfica se encogería al redibujar.
  if (!canvas.dataset.alto) canvas.dataset.alto = canvas.getAttribute("height");
  const alto = Number(canvas.dataset.alto);
  canvas.width = ancho * ratio;
  canvas.height = alto * ratio;
  canvas.style.height = `${alto}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, ancho, alto };
}

function dibujarGraficas() {
  if (!estado.ultimo || $("resultado").hidden) return;
  dibujarECG();
  dibujarSegmentos();
}

function dibujarECG() {
  const r = estado.ultimo;
  const fs = r.frecuencia_muestreo_hz;
  const L = 10 * fs;
  const total = r.segmentos_en_vista;
  const i = estado.segmento;
  const seg = r.resultado.segmentos[i];

  $("seg-etiqueta").textContent = `Segmento ${i + 1}/${r.resultado.n_segmentos}`;
  $("monitor-tiempo").textContent = `${i * 10}–${(i + 1) * 10} s`;
  $("monitor-prob").textContent = seg ? `prob. segmento ${fmt(seg.probabilidad, 4)} · ${seg.nivel}` : "";
  $("seg-anterior").disabled = i <= 0;
  $("seg-siguiente").disabled = i >= total - 1;

  const datos = r.senal_vista.slice(i * L, (i + 1) * L);
  const { ctx, ancho, alto } = prepararCanvas($("grafica-ecg"));
  const izq = 12, der = 12, inf = 22;
  const w = ancho - izq - der, h = alto - inf;

  ctx.clearRect(0, 0, ancho, alto);

  // Rejilla del monitor: líneas finas cada 0.2 s, fuertes cada 1 s.
  for (let t = 0; t <= 10.0001; t += 0.2) {
    const x = izq + (t / 10) * w;
    const fuerte = Math.abs(t - Math.round(t)) < 1e-6;
    ctx.strokeStyle = fuerte ? css("--monitor-rejilla-fuerte") : css("--monitor-rejilla");
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();

    if (fuerte) {
      ctx.fillStyle = "rgba(127, 231, 174, .6)";
      ctx.font = "10px " + css("--mono");
      ctx.textAlign = t === 0 ? "left" : t >= 10 ? "right" : "center";
      ctx.fillText(`${(i * 10 + t).toFixed(0)}s`, x, alto - 6);
    }
  }

  for (let k = 0; k <= 10; k++) {
    const y = (k / 10) * h;
    ctx.strokeStyle = k % 5 === 0 ? css("--monitor-rejilla-fuerte") : css("--monitor-rejilla");
    ctx.beginPath();
    ctx.moveTo(izq, y);
    ctx.lineTo(ancho - der, y);
    ctx.stroke();
  }

  if (!datos.length) return;

  // Escala robusta (percentiles) para que un pico aislado no aplane la señal.
  const ordenados = [...datos].sort((a, b) => a - b);
  let min = ordenados[Math.floor(ordenados.length * 0.002)];
  let max = ordenados[Math.ceil(ordenados.length * 0.998) - 1];
  if (max - min < 1e-9) { min -= 1; max += 1; }
  const pad = (max - min) * 0.12;
  min -= pad; max += pad;

  const traza = () => {
    ctx.beginPath();
    datos.forEach((v, k) => {
      const x = izq + (k / (L - 1)) * w;
      const y = h - ((Math.min(Math.max(v, min), max) - min) / (max - min)) * h;
      k === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
  };

  // Brillo + línea principal.
  ctx.lineJoin = "round";
  ctx.strokeStyle = "rgba(60, 240, 143, .25)";
  ctx.lineWidth = 5;
  traza();
  ctx.stroke();

  ctx.strokeStyle = css("--monitor-linea");
  ctx.lineWidth = 1.6;
  ctx.shadowColor = "rgba(60, 240, 143, .8)";
  ctx.shadowBlur = 6;
  traza();
  ctx.stroke();
  ctx.shadowBlur = 0;
}

let barras = [];   // posiciones de las barras para detectar clics

function dibujarSegmentos() {
  const res = estado.ultimo.resultado;
  const niveles = estado.modelo ? estado.modelo.niveles : [];
  const { ctx, ancho, alto } = prepararCanvas($("grafica-segmentos"));
  const izq = 38, der = 10, sup = 14, inf = 24;
  const w = ancho - izq - der, h = alto - sup - inf;
  const y = (p) => sup + (1 - p) * h;

  ctx.clearRect(0, 0, ancho, alto);
  ctx.font = "11px " + css("--fuente");

  // Bandas de nivel de fondo.
  niveles.forEach((n) => {
    ctx.fillStyle = css(`--${n.nivel.toLowerCase()}-suave`);
    ctx.fillRect(izq, y(n.hasta), w, y(n.desde) - y(n.hasta));
  });

  // Eje Y.
  [0, 0.25, 0.5, 0.75, 1].forEach((p) => {
    ctx.fillStyle = css("--texto-tenue");
    ctx.textAlign = "right";
    ctx.fillText(p.toFixed(2), izq - 8, y(p) + 4);
  });

  // Barras con esquinas redondeadas.
  const n = res.segmentos.length;
  const paso = w / n;
  const anchoBarra = Math.max(Math.min(paso * 0.62, 44), 2);
  barras = [];

  res.segmentos.forEach((s, k) => {
    const x = izq + k * paso + (paso - anchoBarra) / 2;
    const top = y(s.probabilidad);
    const radio = Math.min(6, anchoBarra / 2);

    ctx.fillStyle = colorNivel(s.nivel);
    ctx.globalAlpha = k === estado.segmento ? 1 : 0.55;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, top, anchoBarra, y(0) - top, [radio, radio, 0, 0]);
    else ctx.rect(x, top, anchoBarra, y(0) - top);
    ctx.fill();
    ctx.globalAlpha = 1;

    if (k === estado.segmento) {
      ctx.strokeStyle = css("--texto");
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    barras.push({ x0: izq + k * paso, x1: izq + (k + 1) * paso, k });

    if (n <= 30) {
      ctx.fillStyle = css("--texto-tenue");
      ctx.textAlign = "center";
      ctx.fillText(String(k + 1), x + anchoBarra / 2, alto - 6);
    }
  });

  // Línea del promedio final.
  ctx.strokeStyle = css("--texto");
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  ctx.moveTo(izq, y(res.probabilidad));
  ctx.lineTo(ancho - der, y(res.probabilidad));
  ctx.stroke();
  ctx.setLineDash([]);

  const etiqueta = `promedio ${res.probabilidad.toFixed(2)}`;
  ctx.font = "600 11px " + css("--fuente");
  const tw = ctx.measureText(etiqueta).width + 12;
  const ey = Math.max(y(res.probabilidad) - 22, 0);
  ctx.fillStyle = css("--texto");
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(ancho - der - tw, ey, tw, 18, 9);
  else ctx.rect(ancho - der - tw, ey, tw, 18);
  ctx.fill();
  ctx.fillStyle = "#fff";
  ctx.textAlign = "center";
  ctx.fillText(etiqueta, ancho - der - tw / 2, ey + 13);
}

$("grafica-segmentos").addEventListener("click", (e) => {
  if (!estado.ultimo) return;
  const rect = e.currentTarget.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const barra = barras.find((b) => x >= b.x0 && x < b.x1);
  if (!barra) return;

  if (barra.k >= estado.ultimo.segmentos_en_vista) {
    mostrarError(`El monitor muestra la señal de los primeros ${estado.ultimo.segmentos_en_vista} segmentos.`);
    return;
  }
  estado.segmento = barra.k;
  dibujarGraficas();
});

$("seg-anterior").addEventListener("click", () => {
  estado.segmento = Math.max(estado.segmento - 1, 0);
  dibujarGraficas();
});

$("seg-siguiente").addEventListener("click", () => {
  estado.segmento = Math.min(estado.segmento + 1, estado.ultimo.segmentos_en_vista - 1);
  dibujarGraficas();
});

let temporizador;
window.addEventListener("resize", () => {
  clearTimeout(temporizador);
  temporizador = setTimeout(dibujarGraficas, 120);
});

/* ============================================================
   Mis análisis
   ============================================================ */

async function cargarHistorial() {
  let filas = [];
  try {
    filas = await api("/api/mis-analisis");
  } catch {
    filas = [];
  }

  const contar = (nivel) => filas.filter((f) => f.nivel === nivel).length;
  $("kpi-total").textContent = filas.length;
  $("kpi-bajo").textContent = contar("Bajo");
  $("kpi-moderado").textContent = contar("Moderado");
  $("kpi-alto").textContent = contar("Alto");

  $("historial-vacio").hidden = filas.length > 0;
  document.querySelector(".tabla").hidden = filas.length === 0;

  $("tabla-historial").innerHTML = filas
    .map(
      (f) => `
      <tr>
        <td class="num">${f.id}</td>
        <td class="num">${escapar(fechaCorta(f.fecha))}</td>
        <td><span class="codigo">${escapar(f.codigo_registro || "—")}</span></td>
        <td class="origen" title="${escapar(f.origen)}">${escapar(f.origen)}</td>
        <td class="num">${f.edad ?? "—"}</td>
        <td>${f.sexo ?? "—"}</td>
        <td class="num">${f.imc ?? "—"}</td>
        <td class="num">${f.n_segmentos}</td>
        <td>
          <div class="barra-prob">
            <div class="barra-prob__pista"><i style="width:${f.probabilidad * 100}%;background:${colorNivel(f.nivel)}"></i></div>
            <span class="num">${fmt(f.probabilidad, 4)}</span>
          </div>
        </td>
        <td><span class="pildora ${escapar(f.nivel)}">${escapar(f.nivel)}</span></td>
        <td><span class="version">${escapar(f.version_modelo)}</span></td>
      </tr>`
    )
    .join("");
}

$("refrescar-historial").addEventListener("click", cargarHistorial);

/* ============================================================
   Información del modelo
   ============================================================ */

async function cargarModelo() {
  const m = await api("/api/modelo");
  estado.modelo = m;

  $("punto-servidor").className = "punto en-linea";
  $("texto-servidor").textContent = "Sistema disponible";
  $("lateral-modelo").textContent = "Análisis de señales ECG";
  // Versión legible ("ensemble-140-90-v1" → "Versión 1"); el detalle técnico
  // completo queda en la pestaña "Modelo".
  const numero = (m.version.match(/v(\d+)$/) || [])[1];
  $("lateral-version").textContent = numero ? `Versión ${numero}` : m.version;

  $("nota-demograficos").textContent = m.usa_variables_demograficas
    ? "El modelo utiliza estos datos junto con la señal ECG."
    : "El modelo actual analiza solo la señal ECG. Estos datos se registran para la siguiente versión del modelo.";

  $("info-modelo").innerHTML = htmlInfoModelo(m);
}

/* ============================================================
   Inicio
   ============================================================ */

iniciarSesionUI()
  .then(() => {
    cargarModelo().catch(() => {
      $("punto-servidor").className = "punto fuera";
      $("texto-servidor").textContent = "Sistema no disponible";
      mostrarError("No se pudo conectar con el sistema. Recarga la página.");
    });
    cargarEjemplos();
    actualizarPasos();
  })
  .catch(() => {});
