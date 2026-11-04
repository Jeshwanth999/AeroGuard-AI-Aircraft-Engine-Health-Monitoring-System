let chart;
const $ = (id) => document.getElementById(id);
async function api(url, options={}) {
  const response = await fetch(url, {headers:{"Content-Type":"application/json"}, ...options});
  if (!response.ok) {
    const data = await response.json().catch(()=>({detail:"Request failed"}));
    throw new Error(data.detail || "Request failed");
  }
  return response.json();
}
async function loadMetrics() {
  const m = await api("/api/metrics");
  $("accuracy").textContent = (m.risk_model_accuracy*100).toFixed(1)+"%";
  $("mae").textContent = m.rul_model_mae_cycles+" cycles";
  $("features").textContent = m.features;
}
async function loadHistory() {
  const rows = await api("/api/history");
  $("total").textContent = rows.length;
  const body = $("historyBody");
  if (!rows.length) { body.innerHTML='<tr><td colspan="6" class="empty-table">No predictions yet.</td></tr>'; return; }
  body.innerHTML = rows.map(r => {
    const cls = r.risk_level.toLowerCase();
    const date = new Date(r.created_at).toLocaleString();
    return `<tr><td>#AG-${String(r.id).padStart(4,"0")}</td><td>${date}</td><td>${escapeHtml(r.aircraft_id)}</td><td><span class="risk-pill pill-${cls}">${r.risk_level}</span></td><td>${r.rul_cycles} cycles</td><td>${r.priority}</td></tr>`;
  }).join("");
}
function escapeHtml(s){return String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
function renderResult(data) {
  $("emptyState").classList.add("hidden"); $("result").classList.remove("hidden");
  const risk = data.risk_level;
  $("risk").textContent = risk; $("risk").className = "risk-value risk-"+risk.toLowerCase();
  const pct = Math.round(data.risk_probabilities.high*100);
  $("gauge").textContent = pct+"%";
  $("rul").textContent = data.estimated_rul_cycles;
  $("priority").textContent = data.maintenance_priority;
  $("modelNote").textContent = data.model_note;
  const p = data.risk_probabilities;
  if(chart) chart.destroy();
  chart = new Chart($("riskChart"), {type:"bar",data:{labels:["Low","Moderate","High"],datasets:[{data:[p.low,p.moderate,p.high],backgroundColor:["#58c9a4","#e7b75d","#e46c78"],borderRadius:5,barThickness:24}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>(c.raw*100).toFixed(1)+"%"}}},scales:{y:{min:0,max:1,ticks:{font:{size:9},callback:v=>(v*100)+"%"},grid:{color:"#eff2f7"}},x:{ticks:{font:{size:9}},grid:{display:false}}}}});
  $("explain").innerHTML = data.explanation.map(x=>`<div class="explain-row"><span>${x.feature}</span><div class="bar"><i style="width:${x.importance*100}%"></i></div><b>${(x.importance*100).toFixed(1)}%</b></div>`).join("");
}
$("sensorForm").addEventListener("submit", async (e)=>{
  e.preventDefault(); $("error").textContent="";
  const btn=$("runBtn"); btn.disabled=true; btn.textContent="Analyzing…";
  const fd=new FormData(e.target); const payload=Object.fromEntries(fd.entries());
  for(const k of ["fan_inlet_temp","core_speed","vibration","oil_pressure","exhaust_temp","fuel_flow","operating_cycles"]) payload[k]=Number(payload[k]);
  try { const data=await api("/api/predict",{method:"POST",body:JSON.stringify(payload)}); renderResult(data); await loadHistory(); }
  catch(err){$("error").textContent=err.message;}
  finally{btn.disabled=false;btn.innerHTML='Analyze engine <span>→</span>';}
});
$("sampleBtn").addEventListener("click",()=>{
  const values={fan_inlet_temp:47,core_speed:8450,vibration:4.4,oil_pressure:34,exhaust_temp:680,fuel_flow:2.65,operating_cycles:420};
  Object.entries(values).forEach(([k,v])=>document.querySelector(`[name="${k}"]`).value=v);
});
$("refreshBtn").addEventListener("click",loadHistory);

// =====================================
// NASA FD001 SENSOR INPUT GENERATION
// =====================================

const nasaSensorContainer = $("nasaSensors");

for (let i = 1; i <= 21; i++) {
  const label = document.createElement("label");

  label.innerHTML = `
    Sensor ${i}
    <input
      type="number"
      step="any"
      name="sensor_${i}"
      value="0"
      required
    >
  `;

  nasaSensorContainer.appendChild(label);
}

// =====================================
// NASA SAMPLE DATA
// =====================================

const nasaSample = {
  cycle: 1,
  setting_1: -0.0007,
  setting_2: -0.0004,
  setting_3: 100.0,

  sensor_1: 518.67,
  sensor_2: 641.82,
  sensor_3: 1589.70,
  sensor_4: 1400.60,
  sensor_5: 14.62,
  sensor_6: 21.61,
  sensor_7: 554.36,
  sensor_8: 2388.06,
  sensor_9: 9046.19,
  sensor_10: 1.30,
  sensor_11: 47.47,
  sensor_12: 521.66,
  sensor_13: 2388.02,
  sensor_14: 8138.62,
  sensor_15: 8.4195,
  sensor_16: 0.03,
  sensor_17: 392.0,
  sensor_18: 2388.0,
  sensor_19: 100.0,
  sensor_20: 39.06,
  sensor_21: 23.4190
};

// Load NASA sample values
$("nasaSampleBtn").addEventListener("click", () => {
  Object.entries(nasaSample).forEach(([name, value]) => {
    const input = document.querySelector(
      `#nasaForm [name="${name}"]`
    );

    if (input) input.value = value;
  });
});

// =====================================
// NASA RUL PREDICTION API
// =====================================

$("nasaForm").addEventListener("submit", async (e) => {
  e.preventDefault();

  $("nasaError").textContent = "";
  $("nasaResult").classList.add("hidden");

  const btn = $("nasaRunBtn");
  btn.disabled = true;
  btn.textContent = "Predicting...";

  const fd = new FormData(e.target);

  const engineId = fd.get("engine_id");
  const cycle = Number(fd.get("cycle"));

  const settings = [
    Number(fd.get("setting_1")),
    Number(fd.get("setting_2")),
    Number(fd.get("setting_3"))
  ];

  const sensors = [];

  for (let i = 1; i <= 21; i++) {
    sensors.push(Number(fd.get(`sensor_${i}`)));
  }

  const payload = {
    engine_id: engineId,
    cycle: cycle,
    settings: settings,
    sensors: sensors
  };

  try {
    const data = await api("/api/nasa/predict", {
      method: "POST",
      body: JSON.stringify(payload)
    });

    $("nasaRul").textContent = data.predicted_rul_cycles;
    $("nasaMessage").textContent = data.message;

    $("nasaResult").classList.remove("hidden");

  } catch (err) {
    $("nasaError").textContent = err.message;
  } finally {
    btn.disabled = false;
    btn.textContent = "Predict NASA RUL →";
  }
});

// Load existing dashboard metrics and history
Promise.all([loadMetrics(), loadHistory()])
  .catch(e => console.error(e));

Promise.all([loadMetrics(),loadHistory()]).catch(e=>console.error(e));
