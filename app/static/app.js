const $ = (s) => document.querySelector(s);
const LABEL = { likely_ok: "🟢 Ehtimol halol", doubtful: "🟡 So'rang", haram: "🔴 Harom" };
let mode = "menu";
let file = null;

document.querySelectorAll(".tab").forEach((t) =>
  t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    mode = t.dataset.mode;
    $("#results").innerHTML = "";
    $("#status").textContent = "";
  })
);

$("#file").addEventListener("change", (e) => {
  file = e.target.files[0];
  if (!file) return;
  $("#preview").src = URL.createObjectURL(file);
  $("#preview").hidden = false;
  $("#hint").hidden = true;
  $("#scan").disabled = false;
});

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function renderMenu(data) {
  const s = data.summary;
  let html = `<div class="summary">
    <div class="pill"><b>${s.likely_ok}</b>🟢 halol</div>
    <div class="pill"><b>${s.doubtful}</b>🟡 so'rang</div>
    <div class="pill"><b>${s.haram}</b>🔴 harom</div></div>`;
  if (!data.ai_used) html += `<p id="note" style="font-size:13px;color:#6B6F76;margin-bottom:10px">AI ishlamadi — faqat kalit so'zlar bo'yicha tekshirildi.</p>`;
  for (const it of data.items) {
    html += `<div class="card ${it.verdict}">
      <div class="top"><div><div class="name">${esc(it.name)}</div>
        ${it.name_uz ? `<div class="tr">${esc(it.name_uz)} · ${esc(it.name_en)}</div>` : ""}</div>
        ${it.price ? `<div class="price">₩${it.price.toLocaleString()}</div>` : ""}</div>
      <span class="badge ${it.verdict}">${LABEL[it.verdict]}</span>
      <ul>${it.reasons.map((r) => `<li>${esc(r)}</li>`).join("")}</ul>
      ${it.ingredients.length ? `<div class="ing">Tarkibi: ${it.ingredients.map(esc).join(", ")}</div>` : ""}
    </div>`;
  }
  $("#results").innerHTML = data.items.length ? html : "<p>Menyuda taom nomlari topilmadi.</p>";
}

function renderDish(d) {
  if (!d.recognized) {
    $("#results").innerHTML = `<div class="card"><div class="name">${esc(d.message)}</div>
      <div class="ing">Ehtimollar: ${d.top.map((t) => `${esc(t.dish)} ${(t.confidence * 100).toFixed(0)}%`).join(", ")}</div></div>`;
    return;
  }
  $("#results").innerHTML = `<div class="card ${d.verdict}">
    <div class="top"><div><div class="name">${esc(d.uz)}</div><div class="tr">${esc(d.ko)} · ${esc(d.en)}</div></div>
    <div class="price">${(d.confidence * 100).toFixed(0)}%</div></div>
    <span class="badge ${d.verdict}">${LABEL[d.verdict]}</span>
    <ul><li>${esc(d.reason_uz)}</li></ul>
    <div class="ing">Odatiy tarkibi: ${d.ingredients.map(esc).join(", ")}</div></div>`;
}

$("#scan").addEventListener("click", async () => {
  if (!file) return;
  $("#scan").disabled = true;
  $("#status").textContent = mode === "menu" ? "Menyu o'qilmoqda…" : "Taom aniqlanmoqda…";
  $("#results").innerHTML = "";
  const body = new FormData();
  body.append("file", file);
  try {
    const res = await fetch(`/api/scan-${mode}`, { method: "POST", body });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    $("#status").textContent = "";
    mode === "menu" ? renderMenu(data) : renderDish(data);
  } catch (err) {
    $("#status").textContent = "Xato: " + err.message;
  } finally {
    $("#scan").disabled = false;
  }
});
