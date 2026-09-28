import { formatDate, topProb } from "@/lib/format";
import type { Fixture } from "@/lib/types";

type Props = {
  rows: Fixture[];
  emptyMessage?: string;
};

export function HistoryTable({ rows, emptyMessage = "No scored predictions yet." }: Props) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Date</th>
            <th>Match</th>
            <th>Our pick</th>
            <th>Result</th>
            <th>Top prob</th>
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <tr>
              <td colSpan={5} className="empty-cell">{emptyMessage}</td>
            </tr>
          ) : (
            rows.map((row) => {
              const pick = row.pickLabel ?? row.pred ?? "";
              const actual = row.actualLabel ?? row.actual ?? "—";
              const [label, pct] = topProb(row.probs, row.team, row.opponent);
              const resultCls =
                row.correct === true ? "correct" : row.correct === false ? "incorrect" : "";
              return (
                <tr key={`${row.date}-${row.team}-${row.opponent}`}>
                  <td>{formatDate(row.date)}</td>
                  <td>{row.team} vs {row.opponent}</td>
                  <td>{pick}</td>
                  <td className={`result-cell ${resultCls}`}>{actual}</td>
                  <td>{label} {pct.toFixed(1)}%</td>
                </tr>
              );
            })
          )}
        </tbody>
      </table>
    </div>
  );
}
