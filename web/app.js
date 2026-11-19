// 实景遮挡内核的前端控件。所有位姿/容差都提交给后端内核校验，
// 内核拒绝时把中文错误回显到视窗，控件不自行假设有效范围。
"use strict";

const state = {
  frame: 0,
  frameCount: 1,
  view: "composite",
  ox: 0, oy: 0, oz: 0,
  bias: 10,
  missing: true,
  playing: false,
  playTimer: null,
  lastPayload: null,
};

const $ = (id) => document.getElementById(id);
const img = $("mainImage");
const errorBanner = $("errorBanner");
const missingBadge = $("missingBadge");

const RELATION_META = {
  0: ["背景/无虚拟覆盖", "#5a6270"],
  1: ["虚拟在前（覆盖实景）", "#6fcf97"],
  2: ["现实在前（展品被遮挡）", "#eb5757"],
  3: ["深度接近/共面混合", "#f2c94c"],
  4: ["实景深度缺测", "#f2994a"],
  5: ["近裁剪剔除", "#bb6bd9"],
  6: ["超出远裁剪面", "#56ccf2"],
};

function queryString() {
  return new URLSearchParams({
    frame: state.frame,
    ox: state.ox, oy: state.oy, oz: state.oz,
    bias: state.bias,
    missing: state.missing ? 1 : 0,
  }).toString();
}

function refreshImage() {
  const qs = queryString();
  img.src = `/api/image?view=${state.view}&${qs}&t=${Date.now()}`;
}

function showError(message) {
  errorBanner.textContent = `内核拒绝：${message}`;
  errorBanner.classList.remove("hidden");
}
function clearError() {
  errorBanner.classList.add("hidden");
}

async function refreshEvaluation() {
  try {
    const resp = await fetch(`/api/evaluate?${queryString()}`);
    const payload = await resp.json();
    if (!resp.ok) {
      showError(payload.error || "未知内核错误");
      return;
    }
    clearError();
    state.lastPayload = payload;
    renderStats(payload);
    renderEdge(payload.edge);
    renderWarnings(payload.warnings);
    missingBadge.classList.toggle("hidden", payload.stats.missing_px === 0);
    updateSpatialAudio(payload);
    refreshImage();
  } catch (err) {
    showError(`服务不可达：${err.message}`);
  }
}

function renderStats(p) {
  const s = p.stats;
  const tbody = document.querySelector("#relationTable tbody");
  const rows = [
    [1, s.virtual_front_px], [2, s.real_front_px], [3, s.planar_px],
    [4, s.missing_px], [5, s.near_clipped_px], [6, s.far_virtual_px],
  ];
  tbody.innerHTML = rows.map(([code, n]) => {
    const [label, color] = RELATION_META[code];
    return `<tr><td><span class="dot" style="background:${color}"></span>${label}</td>
      <td class="num">${n}</td><td>${code === 1 ? "z_virtual &lt; z_real" :
        code === 2 ? "z_real &lt; z_virtual" :
        code === 3 ? "|Δz| ≤ 容差" : code === 4 ? "实景无有效深度" :
        code === 5 ? `z &lt; ${s.near_mm}mm` : `z &gt; ${s.far_mm}mm`}</td></tr>`;
  }).join("");

  const fmt = (v) => v === null || v === undefined ? "—" :
    (typeof v === "number" ? v.toFixed(1) : String(v));
  $("statsList").innerHTML = [
    ["虚拟覆盖像素", s.virtual_coverage_px],
    ["虚实重叠像素", s.overlap_px],
    ["实景无效像素（整帧）", s.real_invalid_total_px],
    ["平均深度差 实景−虚拟 (mm)", fmt(s.mean_margin_mm)],
    ["最小深度差 (mm)", fmt(s.min_margin_mm)],
    ["最大深度差 (mm)", fmt(s.max_margin_mm)],
    ["当前展品世界坐标 (mm)", `[${p.exhibit_xyz_mm.map(v => v.toFixed(0)).join(", ")}]`],
    ["参考位姿 (mm)", `[${p.reference_exhibit_xyz_mm.map(v => v.toFixed(0)).join(", ")}]`],
  ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
}

function renderEdge(e) {
  $("edgeList").innerHTML = [
    ["颜色边缘像素", e.color_edge_px],
    ["深度边缘像素", e.depth_edge_px],
    ["深度边缘命中颜色边缘", `${e.matched_depth_edge_px}（${(e.match_ratio * 100).toFixed(1)}%）`],
    ["平均边缘距离 (px)", e.mean_dist_px.toFixed(2)],
    ["95 分位距离 (px)", e.p95_dist_px.toFixed(2)],
    ["最大距离 (px)", e.max_dist_px.toFixed(2)],
    ["配准容差 (px)", e.tolerance_px.toFixed(1)],
  ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
}

function renderWarnings(warnings) {
  const ul = $("warningList");
  if (!warnings.length) {
    ul.className = "warnings ok";
    ul.innerHTML = "<li>内核未报告异常：缺测、近/远裁剪均在可控范围。</li>";
    return;
  }
  ul.className = "warnings";
  ul.innerHTML = warnings.map((w) => `<li>${w}</li>`).join("");
}

// ---- 空间音频：声源跟随展品屏幕位置，被现实遮挡时变暗（低通+衰减）----
const audio = {
  ctx: null, osc: null, gain: null, panner: null, filter: null, enabled: false,
};
async function enableAudio() {
  const Ctx = window.AudioContext || window.webkitAudioContext;
  audio.ctx = new Ctx();
  audio.osc = audio.ctx.createOscillator();
  audio.osc.type = "sine";
  audio.osc.frequency.value = 440;
  audio.filter = audio.ctx.createBiquadFilter();
  audio.filter.type = "lowpass";
  audio.filter.frequency.value = 2000;
  audio.panner = audio.ctx.createStereoPanner();
  audio.gain = audio.ctx.createGain();
  audio.gain.gain.value = 0.0;
  audio.osc.connect(audio.filter).connect(a.panner).connect(audio.gain)
    .connect(audio.ctx.destination);
  audio.osc.start();
  audio.enabled = true;
  $("audioToggle").textContent = "关闭空间音频";
  $("audioState").textContent = "音频开启：声像随展品，遮挡时变暗";
  $("audioState").className = "on";
  updateSpatialAudio(state.lastPayload);
}
function disableAudio() {
  if (audio.ctx) audio.ctx.close();
  audio.ctx = null; audio.enabled = false;
  $("audioToggle").textContent = "开启空间音频";
  $("audioState").textContent = "音频关闭（默认静音）";
  $("audioState").className = "muted";
}
function updateSpatialAudio(payload) {
  if (!audio.enabled || !audio.ctx || !payload) return;
  // 用展品世界 x 与画面宽度近似声像；现实在前占比越高越闷、越轻
  const x = payload.exhibit_xyz_mm[0];
  const pan = Math.max(-1, Math.min(1, x / 700));
  const s = payload.stats;
  const occludedRatio = s.overlap_px > 0 ? s.real_front_px / s.overlap_px : 0;
  const t = audio.ctx.currentTime;
  audio.panner.pan.setTargetAtTime(pan, t, 0.1);
  audio.filter.frequency.setTargetAtTime(2000 * (1 - 0.8 * occludedRatio), t, 0.15);
  audio.gain.gain.setTargetAtTime(0.12 * (1 - 0.6 * occludedRatio), t, 0.15);
  audio.osc.frequency.setTargetAtTime(360 + 160 * (1 - occludedRatio), t, 0.2);
}

// ---- 控件绑定 ----
function bindSlider(id, key, valId, reevaluate = true) {
  const el = $(id);
  el.addEventListener("input", () => {
    state[key] = Number(el.value);
    $(valId).textContent = el.value;
    if (reevaluate) refreshEvaluation();
  });
}
bindSlider("ox", "ox", "oxVal");
bindSlider("oy", "oy", "oyVal");
bindSlider("oz", "oz", "ozVal");
bindSlider("bias", "bias", "biasVal");
$("missingWarn").addEventListener("change", (e) => {
  state.missing = e.target.checked;
  refreshEvaluation();
});
$("resetBtn").addEventListener("click", () => {
  state.ox = state.oy = state.oz = 0; state.bias = 10;
  for (const [id, v] of [["ox", 0], ["oy", 0], ["oz", 0], ["bias", 10]]) {
    $(id).value = v;
  }
  $("oxVal").textContent = $("oyVal").textContent = $("ozVal").textContent = "0";
  $("biasVal").textContent = "10";
  refreshEvaluation();
});

$("viewSwitch").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-view]");
  if (!btn) return;
  state.view = btn.dataset.view;
  document.querySelectorAll("#viewSwitch button").forEach((b) =>
    b.classList.toggle("active", b === btn));
  refreshImage();
});

$("frameSlider").addEventListener("input", (e) => {
  state.frame = Number(e.target.value);
  $("frameLabel").textContent = `帧 ${state.frame}`;
  refreshEvaluation();
});
$("playBtn").addEventListener("click", () => {
  state.playing = !state.playing;
  $("playBtn").textContent = state.playing ? "暂停轨迹" : "播放轨迹";
  if (state.playing) {
    state.playTimer = setInterval(() => {
      state.frame = (state.frame + 1) % state.frameCount;
      $("frameSlider").value = state.frame;
      $("frameLabel").textContent = `帧 ${state.frame}`;
      refreshEvaluation();
    }, 900);
  } else {
    clearInterval(state.playTimer);
  }
});
$("audioToggle").addEventListener("click", () => {
  if (audio.enabled) disableAudio(); else enableAudio();
});

$("gtCheckBtn").addEventListener("click", async () => {
  const box = $("gtCheck");
  box.classList.remove("hidden", "good", "bad");
  box.textContent = "核对中……";
  try {
    const resp = await fetch(`/api/gt_check?frame=${state.frame}`);
    const p = await resp.json();
    if (!resp.ok) {
      box.classList.add("bad");
      box.textContent = `核对失败：${p.error}`;
      return;
    }
    const good = p.mae_rgb < 25 && p.mismatch_ratio < 0.05;
    box.classList.add(good ? "good" : "bad");
    box.innerHTML =
      `帧 ${p.frame_index} 参考位姿 vs 录制真值（排除缺测区，比较 ${p.compared_px} 像素）<br>` +
      `平均色差 <b>${p.mae_rgb.toFixed(2)}</b>，严重不一致占比 ` +
      `<b>${(p.mismatch_ratio * 100).toFixed(2)}%</b><br>` +
      (good ? "结论：空间位置与遮挡边界与真值一致。" : "结论：与真值偏差过大，请检查标定。") +
      `<br><span style="color:#9aa3b2">${p.note}</span>`;
  } catch (err) {
    box.classList.add("bad");
    box.textContent = `核对请求失败：${err.message}`;
  }
});

async function init() {
  const info = await (await fetch("/api/info")).json();
  state.frameCount = info.frame_count;
  $("frameSlider").max = info.frame_count - 1;
  $("nearFoot").textContent = info.near_mm;
  $("farFoot").textContent = info.far_mm;
  $("invalidFoot").textContent = info.depth_invalid_value;
  // 用内核下发的移动范围覆盖滑块范围，避免前端写死
  const lim = info.move_limits;
  $("ox").min = lim.x[0]; $("ox").max = lim.x[1];
  $("oy").min = lim.y[0]; $("oy").max = lim.y[1];
  $("oz").min = lim.z[0]; $("oz").max = lim.z[1];
  refreshEvaluation();
}
init();
