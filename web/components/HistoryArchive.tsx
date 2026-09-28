"use client";

import { useState } from "react";

import { HistoryTable } from "@/components/HistoryTable";
import type { Fixture, LeagueMetrics } from "@/lib/types";

const PAGE_SIZE = 20;

type Props = {
  history: Fixture[];
  metrics: LeagueMetrics | null | undefined;
};

export function HistoryArchive({ history, metrics }: Props) {
  const [open, setOpen] = useState(false);
  const [page, setPage] = useState(0);

  const total = history.length;
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const safePage = Math.min(Math.max(page, 0), totalPages - 1);
  const pageRows = history.slice(safePage * PAGE_SIZE, (safePage + 1) * PAGE_SIZE);

  function openArchive() {
    setPage(0);
    setOpen(true);
    requestAnimationFrame(() => {
      document.getElementById("history-archive")?.scrollIntoView({ behavior: "smooth" });
    });
  }

  return (
    <>
      <section className="metrics-bar">
        {!metrics ? (
          <div className="metrics-btn metrics-btn--static" role="status">
            <span className="metric-label">Track record</span>
            <span className="metric-values">No scored predictions yet for this league.</span>
          </div>
        ) : (
          <button
            type="button"
            className="metrics-btn"
            onClick={openArchive}
            aria-expanded={open}
          >
            <span className="metric-label">Track record</span>
            <span className="metric-values">
              <strong>{metrics.n}</strong> scored ·{" "}
              <strong>{(metrics.accuracy * 100).toFixed(1)}%</strong> accuracy
            </span>
            <span className="metric-cta">View all →</span>
          </button>
        )}
      </section>

      {open && (
        <section id="history-archive" className="history-section">
          <div className="archive-header">
            <h2>All scored predictions</h2>
            <button type="button" className="btn-text" onClick={() => setOpen(false)}>
              Close
            </button>
          </div>
          <p className="history-hint">
            Page {safePage + 1} of {totalPages} · {total} fixtures total (newest first).
          </p>
          <HistoryTable rows={pageRows} />
          <div className="pagination">
            <button
              type="button"
              className="btn-secondary"
              disabled={safePage <= 0}
              onClick={() => setPage((p) => p - 1)}
            >
              Newer
            </button>
            <span className="page-info">Page {safePage + 1} of {totalPages}</span>
            <button
              type="button"
              className="btn-secondary"
              disabled={safePage >= totalPages - 1}
              onClick={() => setPage((p) => p + 1)}
            >
              Older
            </button>
          </div>
        </section>
      )}
    </>
  );
}
