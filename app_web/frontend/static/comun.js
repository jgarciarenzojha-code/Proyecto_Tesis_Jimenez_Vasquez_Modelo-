"use strict";

/* ============================================================
   Utilidades compartidas por /app y /admin
   ============================================================ */

const $ = (id) => document.getElementById(id);

const css = (nombre) =>
  getComputedStyle(document.documentElement).getPropertyValue(nombre).trim();

function escapar(texto) {
  const div = document.createElement("div");
  div.textContent = texto ?? "";
  return div.innerHTML;
}

const fmt = (n, d = 2) =>
  n === null || n === undefined || Number.isNaN(Number(n)) ? "—" : Number(n).toFixed(d);

const colorNivel = (nivel) => css(`--${String(nivel).toLowerCase()}`);

const fechaCorta = (iso) => (iso ? iso.replace("T", " ").slice(0, 16) : "—");

/* ============================================================
   Llamadas a la API (con sesión y protección CSRF)
   ============================================================ */

async function api(url, opciones = {}) {
  const cabeceras = { "X-Solicitud": "app-ecg", ...(opciones.headers || {}) };

  let cuerpo = opciones.body;
  if (cuerpo && !(cuerpo instanceof FormData) && typeof cuerpo !== "string") {
    cuerpo = JSON.stringify(cuerpo);
    cabeceras["Content-Type"] = "application/json";
  }

  const r = await fetch(url, { ...opciones, headers: cabeceras, body: cuerpo, credentials: "same-origin" });

  if (r.status === 401 && !url.startsWith("/api/auth/login")) {
    window.location.href = "/login";
    throw new Error("Sesión expirada.");
  }

  const tipo = r.headers.get("content-type") || "";
  const datos = tipo.includes("application/json") ? await r.json() : await r.text();

  if (!r.ok) {
    const detalle = datos && datos.detail;
    const mensaje = Array.isArray(detalle)
      ? "Datos no válidos."
      : detalle || "Ocurrió un error inesperado.";
    throw new Error(mensaje);
  }

  return datos;
}

/* ============================================================
   Avisos flotantes (toast)
   ============================================================ */

function avisar(texto, tipo = "ok") {
  let contenedor = $("avisos");
  if (!contenedor) {
    contenedor = document.createElement("div");
    contenedor.id = "avisos";
    contenedor.className = "avisos";
    document.body.appendChild(contenedor);
  }
  const aviso = document.createElement("div");
  aviso.className = `toast toast--${tipo}`;
  aviso.textContent = texto;
  contenedor.appendChild(aviso);
  setTimeout(() => aviso.classList.add("saliendo"), 3600);
  setTimeout(() => aviso.remove(), 4000);
}

/* ============================================================
   Sesión: tarjeta de usuario, cierre de sesión, cambio de contraseña
   ============================================================ */

const NOMBRE_ROL = { administrador: "Administrador", especialista: "Especialista" };

async function iniciarSesionUI() {
  const usuario = await api("/api/auth/yo");

  const iniciales = usuario.nombre
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((p) => p[0].toUpperCase())
    .join("");

  $("usuario-iniciales").textContent = iniciales || "U";
  $("usuario-nombre").textContent = usuario.nombre;
  $("usuario-rol").textContent = NOMBRE_ROL[usuario.rol] || usuario.rol;

  $("boton-salir").addEventListener("click", cerrarSesion);
  $("boton-contrasena").addEventListener("click", abrirDialogoContrasena);

  document.querySelectorAll("[data-solo-admin]").forEach((el) => {
    el.hidden = usuario.rol !== "administrador";
  });

  return usuario;
}

async function cerrarSesion() {
  try {
    await api("/api/auth/logout", { method: "POST" });
  } finally {
    window.location.href = "/login";
  }
}

function abrirDialogoContrasena() {
  let dialogo = $("dialogo-contrasena");

  if (!dialogo) {
    dialogo = document.createElement("dialog");
    dialogo.id = "dialogo-contrasena";
    dialogo.className = "dialogo";
    dialogo.innerHTML = `
      <form method="dialog" class="dialogo__cuerpo" id="form-contrasena">
        <h2>Cambiar contraseña</h2>
        <p class="nota">Mínimo 8 caracteres, combinando letras y números.</p>
        <label class="campo"><span>Contraseña actual</span>
          <div class="campo__control"><input type="password" id="clave-actual" autocomplete="current-password" required></div>
        </label>
        <label class="campo"><span>Nueva contraseña</span>
          <div class="campo__control"><input type="password" id="clave-nueva" autocomplete="new-password" required></div>
        </label>
        <label class="campo"><span>Repetir nueva contraseña</span>
          <div class="campo__control"><input type="password" id="clave-repetir" autocomplete="new-password" required></div>
        </label>
        <p class="error" id="error-contrasena" hidden></p>
        <div class="dialogo__acciones">
          <button type="button" class="boton-secundario" id="cancelar-contrasena">Cancelar</button>
          <button type="submit" class="boton-compacto">Guardar</button>
        </div>
      </form>`;
    document.body.appendChild(dialogo);

    $("cancelar-contrasena").addEventListener("click", () => dialogo.close());

    $("form-contrasena").addEventListener("submit", async (e) => {
      e.preventDefault();
      const error = $("error-contrasena");
      error.hidden = true;

      if ($("clave-nueva").value !== $("clave-repetir").value) {
        error.textContent = "Las contraseñas nuevas no coinciden.";
        error.hidden = false;
        return;
      }

      try {
        await api("/api/auth/cambiar-contrasena", {
          method: "POST",
          body: { actual: $("clave-actual").value, nueva: $("clave-nueva").value },
        });
        dialogo.close();
        avisar("Contraseña actualizada correctamente.");
      } catch (err) {
        error.textContent = err.message;
        error.hidden = false;
      }
    });
  }

  dialogo.querySelectorAll("input").forEach((i) => (i.value = ""));
  $("error-contrasena").hidden = true;
  dialogo.showModal();
}

/* ============================================================
   Información del modelo (usada en /app y /admin)
   ============================================================ */

function nombreMetrica(clave) {
  return (
    {
      roc_auc: "ROC-AUC",
      pr_auc: "PR-AUC",
      exactitud: "Exactitud",
      precision: "Precisión",
      sensibilidad: "Sensibilidad",
      especificidad: "Especificidad",
      f1: "F1-score",
      prevalencia: "Prevalencia",
    }[clave] || clave
  );
}

function htmlInfoModelo(m) {
  const metricas = (bloque) =>
    Object.entries(bloque)
      .filter(([k]) => k !== "descripcion")
      .map(
        ([k, v]) => `
        <div class="metrica">
          <span>${escapar(nombreMetrica(k))}</span>
          <strong>${fmt(v, 4)}</strong>
          <div class="metrica__pista"><i style="width:${Math.min(Math.max(v, 0), 1) * 100}%"></i></div>
        </div>`
      )
      .join("");

  const bloquesMetricas = Object.values(m.metricas || {})
    .map(
      (b) => `
      <div class="tarjeta">
        <h3>${escapar(b.descripcion || "Métricas")}</h3>
        <div class="metricas">${metricas(b)}</div>
      </div>`
    )
    .join("");

  const escala = m.niveles
    .map((n) => `<div style="width:${(n.hasta - n.desde) * 100}%;background:${colorNivel(n.nivel)}"></div>`)
    .join("");

  const marcas = [0, ...m.niveles.map((n) => n.hasta)]
    .map((c) => `<span style="left:${c * 100}%">${fmt(c, 4)}</span>`)
    .join("");

  const pildoras = m.niveles
    .map((n) => `<span class="pildora ${escapar(n.nivel)}">${escapar(n.nivel)}: ${fmt(n.desde, 4)} – ${fmt(n.hasta, 4)}</span>`)
    .join(" ");

  return `
    <div class="tarjeta hero-modelo ancho-completo">
      <svg class="hero-modelo__ecg" viewBox="0 0 300 80" aria-hidden="true"><polyline points="0,45 40,45 58,30 66,45 88,52 96,8 104,70 112,45 140,36 155,45 190,45 198,30 206,45 228,52 236,8 244,70 252,45 280,36 300,45"/></svg>
      <h2>${escapar(m.nombre)}</h2>
      <p>${escapar(m.descripcion)}</p>
      <div class="hero-modelo__chips">
        <span>${escapar(m.version)}</span>
        <span>${escapar(m.entradas.join(", "))}</span>
        <span>${escapar(m.criterio_etiqueta)}</span>
        ${m.estado ? `<span class="chip-aviso">${escapar(m.estado)}</span>` : ""}
      </div>
    </div>

    <div class="tarjeta">
      <h3>Configuración</h3>
      <dl class="lista-datos">
        <dt>Arquitectura</dt><dd>${escapar(m.arquitectura)}</dd>
        <dt>Frecuencia de muestreo</dt><dd>${m.frecuencia_muestreo_hz} Hz</dd>
        <dt>Segmento</dt><dd>${m.muestras_por_segmento} muestras (10 s)</dd>
        <dt>Preprocesamiento</dt><dd>Z-score por segmento</dd>
        <dt>Agregación</dt><dd>${escapar(m.agregacion)}</dd>
        <dt>Archivo del modelo</dt><dd>${escapar(m.archivo_checkpoint)} (época ${m.epoca_checkpoint ?? "—"})</dd>
        <dt>Cómputo</dt><dd>${escapar(m.computo)}</dd>
      </dl>
    </div>

    <div class="tarjeta">
      <h3>Umbral y niveles de alerta</h3>
      <dl class="lista-datos">
        <dt>Umbral de decisión</dt><dd>${fmt(m.umbral_decision, 4)}</dd>
      </dl>
      <div class="escala-modelo" style="margin-top:18px">
        <div class="escala-modelo__barra">${escala}</div>
        <div class="escala-modelo__marcas">${marcas}</div>
      </div>
      <p style="display:flex; gap:6px; flex-wrap:wrap">${pildoras}</p>
      <p class="nota">${escapar(m.origen_umbrales)}</p>
    </div>

    ${bloquesMetricas}
  `;
}
