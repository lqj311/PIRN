const STORAGE_KEY = "pirn_runs_v1";

function byId(id) {
  return document.getElementById(id);
}

function nowText() {
  const d = new Date();
  const p = (x) => String(x).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

function readConfig() {
  return {
    dataRoot: byId("dataRoot").value.trim(),
    outputDir: byId("outputDir").value.trim(),
    device: byId("device").value,
    seed: Number(byId("seed").value),
    epochs: Number(byId("epochs").value),
    batchSize: Number(byId("batchSize").value),
    lr: Number(byId("lr").value),
    wd: Number(byId("wd").value),
    workers: Number(byId("workers").value),
    gradClip: Number(byId("gradClip").value),
    dim: Number(byId("dim").value),
    tokens: Number(byId("tokens").value),
    protoRgb: Number(byId("protoRgb").value),
    protoSn: Number(byId("protoSn").value),
    tau: Number(byId("tau").value),
    sinkhornIters: Number(byId("sinkhornIters").value),
    mncHeads: Number(byId("mncHeads").value),
    mncDrop: Number(byId("mncDrop").value),
    recW: Number(byId("recW").value),
    semW: Number(byId("semW").value),
    divW: Number(byId("divW").value),
  };
}

function buildTrainCommand(cfg) {
  return [
    "python -m pirn_paper.train",
    `--data-root "${cfg.dataRoot}"`,
    `--output-dir "${cfg.outputDir}"`,
    `--epochs ${cfg.epochs}`,
    `--batch-size ${cfg.batchSize}`,
    `--lr ${cfg.lr}`,
    `--weight-decay ${cfg.wd}`,
    `--num-workers ${cfg.workers}`,
    `--grad-clip ${cfg.gradClip}`,
    `--seed ${cfg.seed}`,
    `--device ${cfg.device}`,
    `--dim ${cfg.dim}`,
    `--num-tokens ${cfg.tokens}`,
    `--num-proto-rgb ${cfg.protoRgb}`,
    `--num-proto-sn ${cfg.protoSn}`,
    `--sinkhorn-tau ${cfg.tau}`,
    `--sinkhorn-iters ${cfg.sinkhornIters}`,
    `--mnc-heads ${cfg.mncHeads}`,
    `--mnc-dropout ${cfg.mncDrop}`,
    `--rec-weight ${cfg.recW}`,
    `--sem-weight ${cfg.semW}`,
    `--div-weight ${cfg.divW}`,
  ].join(" ");
}

function getRuns() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
  } catch {
    return [];
  }
}

function setRuns(runs) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(runs));
}

function renderRuns() {
  const rows = getRuns();
  const body = byId("runsTableBody");
  body.innerHTML = "";
  rows.forEach((r) => {
    const tr = document.createElement("tr");
    const cmdShort = r.command.length > 80 ? `${r.command.slice(0, 80)}...` : r.command;
    tr.innerHTML = `
      <td>${r.time}</td>
      <td>${r.dataRoot}</td>
      <td>${r.epochs}</td>
      <td>${r.auroc ?? "-"}</td>
      <td title="${r.command.replace(/"/g, "&quot;")}">${cmdShort}</td>
    `;
    body.appendChild(tr);
  });
}

function exportConfig() {
  const cfg = readConfig();
  const blob = new Blob([JSON.stringify(cfg, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `pirn_config_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

function importResultFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const obj = JSON.parse(String(reader.result));
      const cfg = readConfig();
      const runs = getRuns();
      runs.unshift({
        time: nowText(),
        dataRoot: cfg.dataRoot,
        epochs: cfg.epochs,
        command: buildTrainCommand(cfg),
        auroc: obj.auroc ?? obj.eval_auroc ?? obj.best_auroc ?? "-",
      });
      setRuns(runs);
      renderRuns();
      alert("结果已导入并写入实验记录。");
    } catch (e) {
      alert("JSON 解析失败，请检查文件格式。");
    }
  };
  reader.readAsText(file);
}

function bindEvents() {
  byId("genCmdBtn").addEventListener("click", () => {
    byId("commandBox").textContent = buildTrainCommand(readConfig());
  });

  byId("saveRunBtn").addEventListener("click", () => {
    const cfg = readConfig();
    const runs = getRuns();
    runs.unshift({
      time: nowText(),
      dataRoot: cfg.dataRoot,
      epochs: cfg.epochs,
      command: buildTrainCommand(cfg),
      auroc: "-",
    });
    setRuns(runs);
    renderRuns();
  });

  byId("clearRunsBtn").addEventListener("click", () => {
    if (!confirm("确认清空所有实验记录吗？")) return;
    setRuns([]);
    renderRuns();
  });

  byId("exportConfigBtn").addEventListener("click", exportConfig);

  byId("importResultFile").addEventListener("change", (e) => {
    const input = e.target;
    if (!input.files || !input.files[0]) return;
    importResultFile(input.files[0]);
    input.value = "";
  });
}

function init() {
  byId("commandBox").textContent = buildTrainCommand(readConfig());
  renderRuns();
  bindEvents();
}

init();

