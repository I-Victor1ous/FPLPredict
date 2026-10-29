import type {
  ConfidenceTier,
  LeagueEvaluation,
  LeagueMetrics,
  RollingBacktestSeason,
} from "@/lib/types";

type Props = {
  evaluation?: LeagueEvaluation;
  metrics?: LeagueMetrics | null;
  tiers?: ConfidenceTier[];
};

function pct(v: number | null | undefined, digits = 1): string {
  if (v == null || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

export function PerformancePanel({ evaluation, metrics, tiers }: Props) {
  const test = evaluation?.testMetrics;

  return (
    <div className="performance-panel">
      <section className="history-section">
        <h2>Held-out test season (selected model)</h2>
        {test ? (
          <ul className="perf-list">
            <li>
              Test accuracy: {pct(test.accuracy)} ({test.n ?? "—"} fixtures in test
              season)
            </li>
            <li>Test RPS: {test.rps != null ? test.rps.toFixed(4) : "—"} (lower is better)</li>
            <li>Test log loss: {test.log_loss?.toFixed(4) ?? "—"}</li>
            <li>Always-pick-win baseline: {pct(test.home_win_baseline_accuracy)}</li>
            {test.market_baseline ? (
              <li>
                Bookmaker baseline (Avg odds, de-vigged): {pct(test.market_baseline.accuracy)}{" "}
                ({test.market_baseline.n} of{" "}
                {test.market_baseline.test_n ?? test.n ?? "—"} test fixtures with odds
                {test.market_baseline.coverage != null
                  ? `, ${(test.market_baseline.coverage * 100).toFixed(0)}% coverage`
                  : ""}
                ; log loss {test.market_baseline.log_loss.toFixed(4)})
              </li>
            ) : null}
            {test.brier != null ? (
              <li>Brier score: {test.brier.toFixed(4)} (lower is better)</li>
            ) : null}
          </ul>
        ) : (
          <p className="empty">Run model reselection to populate test-season metrics.</p>
        )}
        {test?.calibration && test.calibration.length > 0 ? (
          <div className="table-wrap calibration-table">
            <table>
              <caption className="table-caption">
                Calibration by top-pick confidence (held-out test season)
              </caption>
              <thead>
                <tr>
                  <th>Confidence bin</th>
                  <th>n</th>
                  <th>Mean predicted</th>
                  <th>Observed hit rate</th>
                </tr>
              </thead>
              <tbody>
                {test.calibration.map((bin) => (
                  <tr key={`${bin.low}-${bin.high}`}>
                    <td>
                      {(bin.low * 100).toFixed(0)}%–{(bin.high * 100).toFixed(0)}%
                    </td>
                    <td>{bin.n}</td>
                    <td>{(bin.meanPredicted * 100).toFixed(1)}%</td>
                    <td>{(bin.observed * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </section>

      {evaluation?.rollingBacktest && evaluation.rollingBacktest.length > 0 ? (
        <section className="history-section">
          <h2>Rolling-origin backtest (by season)</h2>
          <p className="hero-sub">
            Each row scores one season using only data from earlier seasons and the
            deployed strategy.
          </p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Season</th>
                  <th>n</th>
                  <th>Accuracy</th>
                  <th>Log loss</th>
                  <th>Market acc.</th>
                </tr>
              </thead>
              <tbody>
                {evaluation.rollingBacktest.map((row: RollingBacktestSeason) => (
                  <tr key={row.seasonEnd}>
                    <td>{row.seasonEnd - 1}–{String(row.seasonEnd).slice(-2)}</td>
                    <td>{row.n}</td>
                    <td>{pct(row.accuracy)}</td>
                    <td>{row.log_loss.toFixed(4)}</td>
                    <td>
                      {row.market_accuracy != null ? pct(row.market_accuracy) : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      <section className="history-section">
        <h2>Live track record</h2>
        {metrics && metrics.n > 0 && metrics.accuracy != null ? (
          <p className="hero-sub">
            {metrics.n} scored fixtures · {(metrics.accuracy * 100).toFixed(1)}% accuracy
            {metrics.ci95Low != null && metrics.ci95High != null ? (
              <>
                {" "}
                (approx. 95% interval {(metrics.ci95Low * 100).toFixed(0)}%–
                {(metrics.ci95High * 100).toFixed(0)}%)
              </>
            ) : null}
          </p>
        ) : (
          <p className="hero-sub">No live scored predictions yet for this league.</p>
        )}
      </section>

      {tiers && tiers.length > 0 ? (
        <section className="history-section">
          <h2>Confidence tiers (live)</h2>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Min top probability</th>
                  <th>n</th>
                  <th>Accuracy</th>
                </tr>
              </thead>
              <tbody>
                {tiers.map((t) => (
                  <tr key={t.minTopProb}>
                    <td>≥ {t.minTopProb}%</td>
                    <td>{t.n}</td>
                    <td>{(t.accuracy * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}
    </div>
  );
}
