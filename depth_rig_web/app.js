"use strict";
// 深度测试相机回放前端。画布逐像素绘制后端返回的「最终显示缓冲」，
// 断言与界面读取的是同一份 DisplayBuffer，前端不自行重算遮挡。

const state = {
  preset: "golden",
  frame: 12,
  speed: 18.0,
  plateZ: 400.0,
  history: 2,
  showFailure: true,
  showRegion: false,
  info: null,
  lastPayload: null,
  failure: null,
};

const $ = (id) => document.getElementById(id);
const canvas = $("mainCanvas");
const ctx = canvas.getContext("2d");

const REGION_COLORS = {
  free: "#3a4150",
  occluded: "#eb5757",
  hole_visible: "#6fcf97",
  edge_uncertain: "#f2c94c",
};
const STATUS_COLORS = {
  ok_valid: "#8aa0bd",
  missing_invalid: "#bb6bd9",
  edge_uncertain: "#f2c94c",
};

async function fetchJson(url) {
  const resp = await fetch(url);
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error || "请求失败");
  return data;
}

function showError(msg) {
  const b = $("errorBanner");
  b.textContent = `服务错误：${msg}`;
  b.classList.remove("hidden");
}
function clearError() { $("errorBanner").classList.add("hidden"); }

function drawBuffer(payload, failureMask) {
  const buf = payload.buffer;
  const h = buf.height_px, w = buf.width_px;
  const scale = Math.floor(canvas.width / w);
  canvas.width = w * scale;
  canvas.height = h * scale;

  const img = ctx.createImageData(w, h);
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const i = (r * w + c) * 4;
      const [rr, gg, bb] = buf.color[r][c];
      img.data[i] = rr; img.data[i + 1] = gg; img.data[i + 2] = bb;
      img.data[i + 3] = 255;
    }
  }
  const off = document.createElement("canvas");
  off.width = w; off.height = h;
  off.getContext("2d").putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, w * scale, h * scale);

  // 缺测（零深度无效）像素：紫色描边，明确区别于实体遮挡
  ctx.strokeStyle = STATUS_COLORS.missing_invalid;
  ctx.lineWidth = 1;
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      if (buf.status[r][c] === "missing_invalid") {
        ctx.strokeRect(c * scale + 0.5, r * scale + 0.5, scale - 1, scale - 1);
      }
    }
  }

  if (state.showRegion) {
    ctx.globalAlpha = 0.35;
    for (let r = 0; r < h; r++) {
      for (let c = 0; c < w; c++) {
        const label = payload.regions[r][c];
        if (label === "free") continue;
        ctx.fillStyle = REGION_COLORS[label];
        ctx.fillRect(c * scale, r * scale, scale, scale);
      }
    }
    ctx.globalAlpha = 1.0;
  }

  if (state.showFailure && failureMask) {
    ctx.fillStyle = "rgba(235,87,87,0.85)";
    for (let r = 0; r < h; r++) {
      for (let c = 0; c < w; c++) {
        if (failureMask[r][c]) ctx.fillRect(c * scale, r * scale, scale, scale);
      }
    }
  }
}

function renderInfo(payload) {
  $("frameInfo").innerHTML = [
    ["帧 / 时间戳", `${payload.frame_index} / ${payload.buffer.timestamp_s.toFixed(4)}s`],
    ["板中心 X (mm)", payload.plate_center_x_mm.toFixed(2)],
    ["板平面 z (mm)", payload.plate_z_mm.toFixed(1)],
    ["速度 (mm/帧)", payload.speed_mm_per_frame],
    ["前序时间戳", payload.history_timestamps_s.map((t) => t.toFixed(3)).join(", ")],
  ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");

  const s = payload.buffer.stats;
  $("statsList").innerHTML = [
    ["缺测（有效位=False）像素", s.missing_invalid_px],
    ["边缘不确定带像素", s.edge_uncertain_px],
    ["孔内补深度像素", s.hole_fill_px],
    ["时间错配覆写像素", s.temporal_overwritten_px],
    ["缺测被当最近遮挡", s.missing_as_occluder ? "是（故障）" : "否"],
  ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
}

function renderLegend() {
  $("legend").innerHTML = [
    ...Object.entries(REGION_COLORS).map(([k, v]) =>
      `<span><i class="swatch" style="background:${v}"></i>${k}</span>`),
    `<span><i class="swatch" style="background:${STATUS_COLORS.missing_invalid}"></i>missing_invalid（零深度无效）</span>`,
    `<span><i class="swatch" style="background:#eb5757"></i>失败像素</span>`,
  ].join("");
}

async function refresh() {
  const params = new URLSearchParams({
    preset: state.preset,
    frame: state.frame,
    speed: state.speed,
    plate_z: state.plateZ,
    history: state.history,
  });
  try {
    const payload = await fetchJson(`/api/frame?${params}`);
    state.lastPayload = payload;
    clearError();
    state.failure = null;
    if (state.preset !== "golden" && state.showFailure) {
      const d = await fetchJson(`/api/diff?${new URLSearchParams({
        preset: state.preset, frame: state.frame, speed: state.speed,
      })}`);
      state.failure = d.failure_mask;
    }
    drawBuffer(payload, state.failure);
    renderInfo(payload);
  } catch (err) {
    showError(err.message);
  }
}

async function runRegression() {
  const box = $("regressionBox");
  box.textContent = "运行中……";
  try {
    const r = await fetchJson("/api/regression");
    box.innerHTML = r.checks.map((c) =>
      `<div class="${c.passed ? "ok" : "fail"}">[${c.passed ? "PASS" : "FAIL"}] ` +
      `${c.name} <span style="color:var(--muted)">${c.detail}</span></div>`).join("")
      + `<div class="summary">${r.passed}/${r.total} 通过</div>`;
  } catch (err) {
    box.innerHTML = `<div class="fail">${err.message}</div>`;
  }
}

function bind() {
  $("preset").addEventListener("change", (e) => { state.preset = e.target.value; refresh(); });
  $("frame").addEventListener("input", (e) => {
    state.frame = Number(e.target.value); $("frameVal").textContent = state.frame; refresh();
  });
  document.querySelectorAll(".speed-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".speed-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.speed = btn.dataset.speed === "fast" ? state.info.speeds_mm_per_frame.fast
        : btn.dataset.speed === "slow" ? state.info.speeds_mm_per_frame.slow : 0.0;
      refresh();
    });
  });
  $("plateZ").addEventListener("input", (e) => {
    state.plateZ = Number(e.target.value); $("plateZVal").textContent = state.plateZ; refresh();
  });
  $("history").addEventListener("input", (e) => {
    state.history = Number(e.target.value); $("historyVal").textContent = state.history; refresh();
  });
  $("showFailure").addEventListener("change", (e) => { state.showFailure = e.target.checked; refresh(); });
  $("showRegion").addEventListener("change", (e) => { state.showRegion = e.target.checked; refresh(); });
  $("regressionBtn").addEventListener("click", runRegression);
}

async function init() {
  try {
    state.info = await fetchJson("/api/info");
    $("preset").innerHTML = state.info.presets
      .map((p) => `<option value="${p}">${p}</option>`).join("");
    $("frame").max = state.info.frame_count - 1;
    $("frameVal").textContent = state.frame;
    $("plateZVal").textContent = state.plateZ;
    $("historyVal").textContent = state.history;
    renderLegend();
    bind();
    refresh();
  } catch (err) {
    showError(err.message);
  }
}
init();
