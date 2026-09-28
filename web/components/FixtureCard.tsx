import { formatDate, formatPosition, pickClass } from "@/lib/format";
import type { Fixture } from "@/lib/types";

function ProbBars({
  probs,
  team,
  opponent,
}: {
  probs: Fixture["probs"];
  team: string;
  opponent: string;
}) {
  const rows = [
    { label: team, key: "teamWin" as const, cls: "win" },
    { label: "Draw", key: "draw" as const, cls: "draw" },
    { label: opponent, key: "oppWin" as const, cls: "loss" },
  ];

  return (
    <div className="prob-bars">
      {rows.map((r) => {
        const pct = Number(probs[r.key]);
        if (!Number.isFinite(pct)) return null;
        return (
          <div className="prob-row" key={r.key}>
            <span title={r.label}>{r.label}</span>
            <div className="bar-track">
              <div className={`bar-fill ${r.cls}`} style={{ width: `${pct}%` }} />
            </div>
            <span>{pct.toFixed(1)}%</span>
          </div>
        );
      })}
    </div>
  );
}

type Props = {
  item: Fixture;
};

export function FixtureCard({ item }: Props) {
  const pickRaw = item.pickLabel ?? item.pred ?? "";

  return (
    <article className="card">
      <div className="card-date">{formatDate(item.date)}</div>
      <div className="matchup">
        <div className="team-block">
          <div className="team-name">{item.team}</div>
          <div className="team-pos">{formatPosition(item.teamPosition)}</div>
        </div>
        <span className="vs">vs</span>
        <div className="team-block" style={{ textAlign: "right" }}>
          <div className="team-name">{item.opponent}</div>
          <div className="team-pos">{formatPosition(item.oppPosition)}</div>
        </div>
      </div>
      <span className={`pred-badge ${pickClass(pickRaw)}`}>{pickRaw}</span>
      <ProbBars probs={item.probs} team={item.team} opponent={item.opponent} />
    </article>
  );
}
