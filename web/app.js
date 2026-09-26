const DATA_URL = "./public/data/predictions.json";
const RECENT_LIMIT = 10;
const PAGE_SIZE = 20;

let allHistory = [];
let historyPage = 0;

function parseISODate(iso) {
  return new Date(`${iso}T12:00:00`);
}

function formatDate(iso) {
  const d = parseISODate(iso);
  return d.toLocaleDateString(undefined, {
    weekday: "short",
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function pickClass(pickLabel) {
  if (!pickLabel) return "";
  if (pickLabel === "Draw") return "draw";
  return "win";
}

function topProb(probs, team, opponent) {
  const entries = [
    [team, probs.teamWin],
    ["Draw", probs.draw],
    [opponent, probs.oppWin],
  ];
  return entries.sort((a, b) => b[1] - a[1])[0];
}

function renderProbBars(probs, team, opponent) {
  const rows = [
    { label: team, key: "teamWin", cls: "win" },
    { label: "Draw", key: "draw", cls: "draw" },
    { label: opponent, key: "oppWin", cls: "loss" },
  ];
  return rows
    .map((r) => {
      const pct = probs[r.key];
      const short =
        r.label.length > 18 ? r.label.slice(0, 16) + "…" : r.label;
      return `
    <div class="prob-row">
      <span title="${r.label}">${short}</span>
      <div class="bar-track"><div class="bar-fill ${r.cls}" style="width:${pct}%"></div></div>
      <span>${pct.toFixed(1)}%</span>
    </div>`;
    })
    .join("");
}

function formatPosition(pos) {
  return pos != null && pos > 0 ? `#${pos}` : "—";
}

function renderFixture(item) {
  const pick = item.pickLabel ?? item.pred;
  return `
    <article class="card">
      <div class="card-date">${formatDate(item.date)}</div>
      <div class="matchup">
        <div class="team-block">
          <div class="team-name">${item.team}</div>
          <div class="team-pos">${formatPosition(item.teamPosition)}</div>
        </div>
        <span class="vs">vs</span>
        <div class="team-block" style="text-align:right">
          <div class="team-name">${item.opponent}</div>
          <div class="team-pos">${formatPosition(item.oppPosition)}</div>
        </div>
      </div>
      <span class="pred-badge ${pickClass(pick)}">${pick}</span>
      <div class="prob-bars">${renderProbBars(item.probs, item.team, item.opponent)}</div>
    </article>`;
}

function historyRowHtml(row) {
  const pick = row.pickLabel ?? row.pred;
  const actual =
    row.actualLabel ?? row.actual ?? "—";
  const [label, pct] = topProb(row.probs, row.team, row.opponent);
  const resultCls =
    row.correct === true
      ? "correct"
      : row.correct === false
        ? "incorrect"
        : "";
  return `<tr>
    <td>${formatDate(row.date)}</td>
    <td>${row.team} vs ${row.opponent}</td>
    <td>${pick}</td>
    <td class="result-cell ${resultCls}">${actual}</td>
    <td>${label} ${pct.toFixed(1)}%</td>
  </tr>`;
}

function renderHistoryTable(tbody, rows) {
  tbody.innerHTML = rows.length
    ? rows.map(historyRowHtml).join("")
    : `<tr><td colspan="5" class="empty-cell">No scored predictions yet.</td></tr>`;
}

function renderMetricsButton(metrics) {
  const el = document.getElementById("metrics");
  if (!metrics) {
    el.hidden = true;
    return;
  }
  el.hidden = false;
  el.innerHTML = `
    <button type="button" class="metrics-btn" id="open-archive" aria-expanded="false">
      <span class="metric-label">Track record</span>
      <span class="metric-values">
        <strong>${metrics.n}</strong> scored ·
        <strong>${(metrics.accuracy * 100).toFixed(1)}%</strong> accuracy
      </span>
      <span class="metric-cta">View all →</span>
    </button>`;
  document.getElementById("open-archive").addEventListener("click", openArchive);
}

function openArchive() {
  const archive = document.getElementById("history-archive");
  archive.hidden = false;
  historyPage = 0;
  renderArchivePage();
  archive.scrollIntoView({ behavior: "smooth", block: "start" });
  const btn = document.getElementById("open-archive");
  if (btn) btn.setAttribute("aria-expanded", "true");
}

function closeArchive() {
  document.getElementById("history-archive").hidden = true;
  const btn = document.getElementById("open-archive");
  if (btn) btn.setAttribute("aria-expanded", "false");
}

function renderArchivePage() {
  const total = allHistory.length;
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  historyPage = Math.min(historyPage, totalPages - 1);
  historyPage = Math.max(historyPage, 0);

  const start = historyPage * PAGE_SIZE;
  const pageRows = allHistory.slice(start, start + PAGE_SIZE);

  renderHistoryTable(
    document.querySelector("#archive-table tbody"),
    pageRows
  );

  const summary = document.getElementById("archive-summary");
  if (summary) {
    summary.textContent = `Page ${historyPage + 1} of ${totalPages} · ${total} fixtures total (newest first). Green = pick matched result.`;
  }

  const pageInfo = document.getElementById("page-info");
  if (pageInfo) {
    pageInfo.textContent = `Page ${historyPage + 1} of ${totalPages}`;
  }

  document.getElementById("page-prev").disabled = historyPage <= 0;
  document.getElementById("page-next").disabled = historyPage >= totalPages - 1;
}

function renderModelTagline(meta) {
  const el = document.getElementById("model-tagline");
  if (!el) return;
  const label = meta?.modelLabel ?? "Auto-selected model";
  el.textContent = `${label} · Premier League 1X2`;
}

async function main() {
  const fixturesEl = document.getElementById("fixtures");
  const updatedEl = document.getElementById("last-updated");
  const noteEl = document.getElementById("positions-note");

  document.getElementById("close-archive").addEventListener("click", closeArchive);
  document.getElementById("page-prev").addEventListener("click", () => {
    historyPage -= 1;
    renderArchivePage();
  });
  document.getElementById("page-next").addEventListener("click", () => {
    historyPage += 1;
    renderArchivePage();
  });

  try {
    const res = await fetch(DATA_URL, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    updatedEl.textContent = data.generatedAt
      ? `Updated ${new Date(data.generatedAt).toLocaleString()}`
      : "Updated recently";

    renderModelTagline(data.meta);

    if (noteEl && data.meta?.positionsNote) {
      noteEl.textContent = data.meta.positionsNote;
    }

    allHistory = [...(data.history ?? [])].sort(
      (a, b) => parseISODate(b.date) - parseISODate(a.date)
    );

    renderMetricsButton(data.metrics);

    const upcoming = [...(data.upcoming ?? [])].sort(
      (a, b) => parseISODate(a.date) - parseISODate(b.date)
    );

    if (!upcoming.length) {
      fixturesEl.innerHTML =
        '<p class="empty">No upcoming fixtures in the prediction file. Run <code>./scripts/weekly_update.sh</code> after scraping the latest gameweek.</p>';
    } else {
      fixturesEl.innerHTML = upcoming.map(renderFixture).join("");
    }

    const recent = allHistory.slice(0, RECENT_LIMIT);
    renderHistoryTable(
      document.querySelector("#history-table tbody"),
      recent
    );
  } catch (err) {
    updatedEl.textContent = "Could not load data";
    fixturesEl.innerHTML = `<p class="empty">Failed to load predictions.json (${err.message}). Run <code>./scripts/weekly_update.sh</code> first.</p>`;
  }
}

main();
