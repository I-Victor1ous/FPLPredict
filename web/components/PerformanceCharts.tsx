"use client";

import { useMemo, useState } from "react";

import type { ModelTimelineSegment, WeeklyPerformanceWeek } from "@/lib/types";

const TIMELINE_PAGE_SIZE = 5;

function formatWeek(iso: string, seasonEnd?: number | null): string {
  const d = new Date(`${iso}T12:00:00`);
  const date = d.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
  if (seasonEnd == null || Number.isNaN(seasonEnd)) {
    return date;
  }
  const start = seasonEnd - 1;
  const endShort = String(seasonEnd).slice(-2);
  return `${date} · ${start}–${endShort}`;
}

function formatWhen(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

type Props = {
  weekly?: WeeklyPerformanceWeek[];
  timeline?: ModelTimelineSegment[];
};

export function PerformanceCharts({ weekly, timeline }: Props) {
  const weeks = weekly ?? [];
  const segments = timeline ?? [];
  const [timelinePage, setTimelinePage] = useState(0);

  const timelinePages = Math.max(1, Math.ceil(segments.length / TIMELINE_PAGE_SIZE));
  const safeTimelinePage = Math.min(Math.max(timelinePage, 0), timelinePages - 1);

  const visibleSegments = useMemo(
    () =>
      segments.slice(
        safeTimelinePage * TIMELINE_PAGE_SIZE,
        (safeTimelinePage + 1) * TIMELINE_PAGE_SIZE
      ),
    [segments, safeTimelinePage]
  );

  return (
    <div className="performance-charts">
      <section className="history-section">
        <h2>Gameweek accuracy</h2>
        <p className="hero-sub chart-hint">
          Each bar is one competition gameweek (from match data). Bar height is accuracy (0–100%).
          One row per fixture.
        </p>
        {weeks.length === 0 ? (
          <p className="empty">No scored fixtures yet to chart.</p>
        ) : (
          <div className="weekly-chart" role="img" aria-label="Gameweek prediction accuracy">
            {weeks.map((w) => {
              const pct = Math.min(100, Math.max(0, (w.accuracy ?? 0) * 100));
              return (
                <div className="weekly-bar-col" key={`${w.seasonEnd ?? ""}-${w.gameweek ?? w.weekStart}`}>
                  <div className="weekly-bar-value">{pct.toFixed(0)}%</div>
                  <div className="weekly-bar-track">
                    <div
                      className="weekly-bar-fill"
                      style={{ height: `${pct}%` }}
                      title={`${w.correct}/${w.n} correct`}
                    />
                  </div>
                  <div className="weekly-bar-label">{w.gameweek ?? formatWeek(w.weekStart, w.seasonEnd)}</div>
                  <div className="weekly-bar-dates">{formatWeek(w.weekStart, w.seasonEnd)}</div>
                  <div className="weekly-bar-meta">
                    n={w.n}
                    {w.modelLabel ? ` · ${w.modelLabel}` : ""}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </section>

      <section className="history-section">
        <h2>Model strategy timeline</h2>
        <p className="hero-sub chart-hint">
          When the deployed ensemble strategy changed (from prediction log and model selection).
        </p>
        {segments.length === 0 ? (
          <p className="empty">No strategy history yet.</p>
        ) : (
          <>
            <div className="strategy-timeline-nav">
              <button
                type="button"
                className="strategy-timeline-btn"
                disabled={safeTimelinePage <= 0}
                onClick={() => setTimelinePage((p) => Math.max(0, p - 1))}
                aria-label="Previous strategy periods"
              >
                ←
              </button>
              <span className="strategy-timeline-page">
                {safeTimelinePage + 1} / {timelinePages}
              </span>
              <button
                type="button"
                className="strategy-timeline-btn"
                disabled={safeTimelinePage >= timelinePages - 1}
                onClick={() => setTimelinePage((p) => Math.min(timelinePages - 1, p + 1))}
                aria-label="Next strategy periods"
              >
                →
              </button>
            </div>
            <ul className="strategy-timeline">
              {visibleSegments.map((seg, i) => (
                <li key={`${seg.from}-${seg.strategy}-${safeTimelinePage}-${i}`}>
                  <div className="strategy-timeline-main">
                    <span className="strategy-timeline-label">{seg.label ?? seg.strategy}</span>
                    <span className="strategy-timeline-range">
                      {formatWhen(seg.from)}
                      {seg.to ? ` → ${formatWhen(seg.to)}` : " → present"}
                    </span>
                  </div>
                  {seg.note ? <p className="strategy-timeline-note">{seg.note}</p> : null}
                </li>
              ))}
            </ul>
          </>
        )}
      </section>
    </div>
  );
}
