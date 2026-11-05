import type { FixtureProbs } from "@/lib/types";

type Props = {
  model: FixtureProbs;
  market?: FixtureProbs;
  team: string;
  opponent: string;
};

export function MarketCompare({ model, market, team, opponent }: Props) {
  if (!market) return null;
  const modelTop = Math.max(model.teamWin, model.draw, model.oppWin);
  const marketTop = Math.max(market.teamWin, market.draw, market.oppWin);
  const delta = modelTop - marketTop;
  const deltaLabel =
    Math.abs(delta) < 1
      ? "Similar to bookmakers"
      : delta > 0
        ? "Model more confident than market"
        : "Market more confident than model";

  return (
    <div className="market-compare">
      <p className="market-compare-title">Model vs bookmakers (de-vigged avg odds)</p>
      <p className="market-compare-hint">{deltaLabel}</p>
      <div className="market-compare-grid">
        <span />
        <span>Model</span>
        <span>Market</span>
        {[
          [team, model.teamWin, market.teamWin],
          ["Draw", model.draw, market.draw],
          [opponent, model.oppWin, market.oppWin],
        ].map(([label, m, b]) => (
          <div className="market-compare-row" key={String(label)}>
            <span>{label}</span>
            <span>{Number(m).toFixed(1)}%</span>
            <span>{Number(b).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}
