"use strict";

// Utilidades ($, api, escapar, fmt, colorNivel, avisar, htmlInfoModelo) en comun.js

const admin = { yo: null, usuarios: [], resumen: null };

const NOMBRE_EVENTO = {
  inicio_sesion: "Inicio de sesión",
  inicio_sesion_fallido: "Intento fallido",
  cierre_sesion: "Cierre de sesión",
  analisis: "Análisis",
  analisis_eliminado: "Análisis eliminado",
  usuario_creado: "Usuario creado",
  usuario_editado: "Usuario editado",
  contrasena_restablecida: "Contraseña restablecida",
  cambio_contrasena: "Cambio de contraseña",
  exportacion_csv: "Exportación CSV",
};

/* ============================================================
   Navegación
   ============================================================ */

const CARGAS = {
  resumen: () => cargarResumen(),
  analisis: () => cargarAnalisis(),
  usuarios: () => cargarUsuarios(),
  modelo: () => cargarModelo(),
  auditoria: () => cargarAuditoria(),
};

document.querySelectorAll("button.pestana").forEach((boton) => {
  boton.addEventListener("click", () => {
    document.querySelectorAll(".pestana").forEach((b) => b.classList.remove("activa"));
    document.querySelectorAll(".vista").forEach((v) => v.classList.remove("activa"));
    boton.classList.add("activa");
    $(`vista-${boton.dataset.vista}`).classList.add("activa");
    window.scrollTo({ top: 0 });
    CARGAS[boton.dataset.vista]().catch((e) => avisar(e.message, "error"));
  });
});

/* ============================================================
   Resumen
   ============================================================ */

async function cargarResumen() {
  const r = await api("/api/admin/resumen");
  admin.resumen = r;

  const pct = (n) => (r.total ? ` (${Math.round((n / r.total) * 100)} %)` : "");
  $("k-total").textContent = r.total;
  $("k-especialistas").textContent = r.especialistas_activos;
  $("k-alto").textContent = `${r.por_nivel.Alto}${pct(r.por_nivel.Alto)}`;
  $("k-prob").textContent = fmt(r.probabilidad_media, 2);

  dibujarDias(r.por_dia);
  dibujarDona(r.por_nivel, r.total);

  $("tabla-especialistas").innerHTML = r.por_especialista.length
    ? r.por_especialista
        .map(
          (e) => `
        <tr>
          <td><b>${escapar(e.especialista)}</b></td>
          <td class="num">${e.total}</td>
          <td class="num"><span class="pildora Bajo">${e.bajo}</span></td>
          <td class="num"><span class="pildora Moderado">${e.moderado}</span></td>
          <td class="num"><span class="pildora Alto">${e.alto}</span></td>
          <td class="num">${escapar(fechaCorta(e.ultimo))}</td>
        </tr>`
        )
        .join("")
    : `<tr><td colspan="6" class="celda-vacia">Aún no hay análisis registrados.</td></tr>`;
}

$("refrescar-resumen").addEventListener("click", () => cargarResumen().catch((e) => avisar(e.message, "error")));

function prepararCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const ancho = canvas.clientWidth;
  const alto = Number(canvas.getAttribute("height"));
  canvas.width = ancho * ratio;
  canvas.height = alto * ratio;
  canvas.style.height = `${alto}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, ancho, alto };
}

function dibujarDias(serie) {
  const { ctx, ancho, alto } = prepararCanvas($("grafica-dias"));
  const izq = 30, der = 8, sup = 12, inf = 26;
  const w = ancho - izq - der, h = alto - sup - inf;
  const max = Math.max(4, ...serie.map((d) => d.total));
  const techo = Math.ceil(max / 4) * 4;
  const y = (v) => sup + (1 - v / techo) * h;

  ctx.clearRect(0, 0, ancho, alto);
  ctx.font = "11px " + css("--fuente");

  for (let k = 0; k <= 4; k++) {
    const v = (techo / 4) * k;
    ctx.strokeStyle = css("--borde");
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(izq, y(v));
    ctx.lineTo(ancho - der, y(v));
    ctx.stroke();
    ctx.fillStyle = css("--texto-tenue");
    ctx.textAlign = "right";
    ctx.fillText(String(v), izq - 6, y(v) + 4);
  }

  const paso = w / serie.length;
  const anchoBarra = Math.max(paso * 0.62, 2);
  const grad = ctx.createLinearGradient(0, sup, 0, sup + h);
  grad.addColorStop(0, "#2563eb");
  grad.addColorStop(1, "#06b6d4");

  serie.forEach((d, i) => {
    const x = izq + i * paso + (paso - anchoBarra) / 2;
    if (d.total > 0) {
      ctx.fillStyle = grad;
      ctx.beginPath();
      const r = Math.min(4, anchoBarra / 2);
      if (ctx.roundRect) ctx.roundRect(x, y(d.total), anchoBarra, y(0) - y(d.total), [r, r, 0, 0]);
      else ctx.rect(x, y(d.total), anchoBarra, y(0) - y(d.total));
      ctx.fill();
    }
    if (i % 5 === 4 || i === serie.length - 1) {
      ctx.fillStyle = css("--texto-tenue");
      ctx.textAlign = "center";
      ctx.fillText(d.dia.slice(8, 10) + "/" + d.dia.slice(5, 7), x + anchoBarra / 2, alto - 8);
    }
  });
}

function dibujarDona(porNivel, total) {
  const niveles = ["Bajo", "Moderado", "Alto"];
  const R = 46, C = 2 * Math.PI * R;
  let acumulado = 0;
  let svg = `<circle cx="60" cy="60" r="${R}" fill="none" stroke="${css("--superficie-2")}" stroke-width="16"/>`;

  if (total > 0) {
    niveles.forEach((n) => {
      const parte = porNivel[n] / total;
      if (parte > 0) {
        svg += `<circle cx="60" cy="60" r="${R}" fill="none" stroke="${colorNivel(n)}" stroke-width="16"
                 stroke-dasharray="${parte * C} ${C}" stroke-dashoffset="${-acumulado * C}"
                 transform="rotate(-90 60 60)"/>`;
        acumulado += parte;
      }
    });
  }

  svg += `<text x="60" y="58" text-anchor="middle" class="dona__total">${total}</text>
          <text x="60" y="74" text-anchor="middle" class="dona__rotulo">análisis</text>`;
  $("dona").innerHTML = svg;

  $("dona-leyenda").innerHTML = niveles
    .map((n) => {
      const pct = total ? Math.round((porNivel[n] / total) * 100) : 0;
      return `<div><i style="background:${colorNivel(n)}"></i><span>${n}</span><b>${porNivel[n]}</b><em>${pct} %</em></div>`;
    })
    .join("");
}

/* ============================================================
   Análisis (todos)
   ============================================================ */

function parametrosFiltro() {
  const p = new URLSearchParams();
  if ($("f-usuario").value) p.set("usuario_id", $("f-usuario").value);
  if ($("f-nivel").value) p.set("nivel", $("f-nivel").value);
  if ($("f-desde").value) p.set("desde", $("f-desde").value);
  if ($("f-hasta").value) p.set("hasta", $("f-hasta").value);
  return p.toString();
}

async function cargarAnalisis() {
  if (!admin.usuarios.length) {
    admin.usuarios = await api("/api/admin/usuarios");
  }

  const seleccionado = $("f-usuario").value;
  $("f-usuario").innerHTML =
    `<option value="">Todos</option>` +
    admin.usuarios.map((u) => `<option value="${u.id}">${escapar(u.nombre)}</option>`).join("");
  $("f-usuario").value = seleccionado;

  const filas = await api(`/api/admin/analisis?${parametrosFiltro()}`);

  $("analisis-vacio").hidden = filas.length > 0;

  $("tabla-analisis").innerHTML = filas
    .map(
      (f) => `
      <tr>
        <td class="num">${f.id}</td>
        <td class="num">${escapar(fechaCorta(f.fecha))}</td>
        <td><span class="codigo">${escapar(f.codigo_registro || "—")}</span></td>
        <td><b>${escapar(f.especialista)}</b></td>
        <td class="origen" title="${escapar(f.origen)}">${escapar(f.origen)}</td>
        <td class="num">${f.edad ?? "—"}</td>
        <td>${f.sexo ?? "—"}</td>
        <td>
          <div class="barra-prob">
            <div class="barra-prob__pista"><i style="width:${f.probabilidad * 100}%;background:${colorNivel(f.nivel)}"></i></div>
            <span class="num">${fmt(f.probabilidad, 4)}</span>
          </div>
        </td>
        <td><span class="pildora ${escapar(f.nivel)}">${escapar(f.nivel)}</span></td>
        <td><span class="version">${escapar(f.version_modelo)}</span></td>
        <td><button class="eliminar" data-id="${f.id}">Eliminar</button></td>
      </tr>`
    )
    .join("");

  $("tabla-analisis").querySelectorAll(".eliminar").forEach((b) =>
    b.addEventListener("click", async () => {
      if (!confirm(`¿Eliminar definitivamente el análisis #${b.dataset.id}? Quedará registrado en la auditoría.`)) return;
      try {
        await api(`/api/admin/analisis/${b.dataset.id}`, { method: "DELETE" });
        avisar(`Análisis #${b.dataset.id} eliminado.`);
        cargarAnalisis();
      } catch (e) {
        avisar(e.message, "error");
      }
    })
  );
}

$("filtros").addEventListener("submit", (e) => {
  e.preventDefault();
  cargarAnalisis().catch((err) => avisar(err.message, "error"));
});

$("exportar-csv").addEventListener("click", () => {
  window.location.href = `/api/admin/analisis.csv?${parametrosFiltro()}`;
});

/* ============================================================
   Usuarios
   ============================================================ */

async function cargarUsuarios() {
  admin.usuarios = await api("/api/admin/usuarios");

  $("tabla-usuarios").innerHTML = admin.usuarios
    .map((u) => {
      const esYo = u.id === admin.yo.id;
      const rol = u.rol === "administrador" ? "Administrador" : "Especialista";
      return `
      <tr class="${u.activo ? "" : "fila-inactiva"}">
        <td>
          <div class="persona">
            <span class="persona__avatar ${u.rol === "administrador" ? "persona__avatar--admin" : ""}">${escapar(
              u.nombre.split(/\s+/).slice(0, 2).map((p) => p[0]?.toUpperCase() || "").join("")
            )}</span>
            <b>${escapar(u.nombre)}</b>${esYo ? ' <span class="etiqueta-yo">tú</span>' : ""}
          </div>
        </td>
        <td><span class="mono">${escapar(u.usuario)}</span></td>
        <td><span class="pildora-rol ${u.rol}">${rol}</span></td>
        <td>${u.activo ? '<span class="estado estado--activo">Activo</span>' : '<span class="estado estado--inactivo">Inactivo</span>'}</td>
        <td class="num">${u.total_analisis}</td>
        <td class="num">${escapar(fechaCorta(u.ultimo_acceso))}</td>
        <td>
          <div class="acciones-fila">
            <button class="accion" data-accion="restablecer" data-id="${u.id}">Restablecer contraseña</button>
            ${
              esYo
                ? ""
                : `<button class="accion ${u.activo ? "accion--peligro" : ""}" data-accion="${u.activo ? "desactivar" : "activar"}" data-id="${u.id}">${u.activo ? "Desactivar" : "Activar"}</button>`
            }
          </div>
        </td>
      </tr>`;
    })
    .join("");

  $("tabla-usuarios").querySelectorAll(".accion").forEach((b) =>
    b.addEventListener("click", () => accionUsuario(b.dataset.accion, Number(b.dataset.id)))
  );
}

async function accionUsuario(accion, id) {
  const u = admin.usuarios.find((x) => x.id === id);
  if (!u) return;

  try {
    if (accion === "restablecer") {
      if (!confirm(`¿Generar una nueva contraseña para ${u.nombre}? Su sesión actual se cerrará.`)) return;
      const r = await api(`/api/admin/usuarios/${id}/restablecer`, { method: "POST" });
      mostrarCredenciales("Contraseña restablecida", r.usuario, r.contrasena_temporal);
    } else {
      const activo = accion === "activar";
      if (!activo && !confirm(`¿Desactivar la cuenta de ${u.nombre}? No podrá iniciar sesión.`)) return;
      await api(`/api/admin/usuarios/${id}`, { method: "PATCH", body: { activo } });
      avisar(activo ? "Cuenta activada." : "Cuenta desactivada.");
    }
    cargarUsuarios();
  } catch (e) {
    avisar(e.message, "error");
  }
}

$("nuevo-usuario").addEventListener("click", () => {
  ["u-nombre", "u-usuario", "u-contrasena"].forEach((id) => ($(id).value = ""));
  $("u-rol").value = "especialista";
  $("error-usuario").hidden = true;
  $("dialogo-usuario").showModal();
});

$("cancelar-usuario").addEventListener("click", () => $("dialogo-usuario").close());

$("form-usuario").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("error-usuario").hidden = true;

  try {
    const r = await api("/api/admin/usuarios", {
      method: "POST",
      body: {
        nombre: $("u-nombre").value,
        usuario: $("u-usuario").value,
        rol: $("u-rol").value,
        contrasena: $("u-contrasena").value || null,
      },
    });
    $("dialogo-usuario").close();
    mostrarCredenciales("Usuario creado", r.usuario, r.contrasena_inicial);
    cargarUsuarios();
  } catch (err) {
    $("error-usuario").textContent = err.message;
    $("error-usuario").hidden = false;
  }
});

function mostrarCredenciales(titulo, usuario, contrasena) {
  $("cred-titulo").textContent = titulo;
  $("cred-usuario").textContent = usuario;
  $("cred-contrasena").textContent = contrasena;
  $("cred-enlace").textContent = `${window.location.origin}/login`;
  $("dialogo-credenciales").showModal();
}

$("copiar-credenciales").addEventListener("click", async () => {
  const texto =
    `Acceso a Alerta Temprana ECG\n` +
    `Enlace: ${$("cred-enlace").textContent}\n` +
    `Usuario: ${$("cred-usuario").textContent}\n` +
    `Contraseña: ${$("cred-contrasena").textContent}`;
  try {
    await navigator.clipboard.writeText(texto);
    avisar("Credenciales copiadas.");
  } catch {
    avisar("No se pudo copiar; selecciónalas manualmente.", "error");
  }
});

$("cerrar-credenciales").addEventListener("click", () => $("dialogo-credenciales").close());

/* ============================================================
   Modelo y auditoría
   ============================================================ */

async function cargarModelo() {
  $("info-modelo").innerHTML = htmlInfoModelo(await api("/api/modelo"));
}

async function cargarAuditoria() {
  const filas = await api("/api/admin/auditoria");

  $("tabla-auditoria").innerHTML = filas.length
    ? filas
        .map(
          (f) => `
        <tr>
          <td class="num">${escapar(f.fecha.replace("T", " "))}</td>
          <td><span class="mono">${escapar(f.usuario || "—")}</span></td>
          <td><span class="evento evento--${escapar(f.evento)}">${escapar(NOMBRE_EVENTO[f.evento] || f.evento)}</span></td>
          <td class="detalle">${escapar(f.detalle || "")}</td>
        </tr>`
        )
        .join("")
    : `<tr><td colspan="4" class="celda-vacia">Sin eventos.</td></tr>`;
}

$("refrescar-auditoria").addEventListener("click", () => cargarAuditoria().catch((e) => avisar(e.message, "error")));

/* ============================================================
   Inicio
   ============================================================ */

let temporizador;
window.addEventListener("resize", () => {
  clearTimeout(temporizador);
  temporizador = setTimeout(() => {
    if (admin.resumen && $("vista-resumen").classList.contains("activa")) {
      dibujarDias(admin.resumen.por_dia);
    }
  }, 120);
});

iniciarSesionUI()
  .then((yo) => {
    admin.yo = yo;
    return cargarResumen();
  })
  .catch((e) => avisar(e.message, "error"));
