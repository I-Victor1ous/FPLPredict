"use client";

import { useMemo, useState } from "react";

import type { Fixture } from "@/lib/types";

type Props = {
  fixtures: Fixture[];
};

function outcomeProb(item: Fixture, choice: "team" | "draw" | "opp"): number {
  const p = item.probs;
  if (choice === "team") return p.teamWin / 100;
  if (choice === "draw") return p.draw / 100;
  return p.oppWin / 100;
}

export function AccaBuilder({ fixtures }: Props) {
  const [selected, setSelected] = useState<Record<string, "team" | "draw" | "opp">>({});

  const combined = useMemo(() => {
    let product = 1;
    for (const item of fixtures) {
      const key = `${item.date}-${item.team}-${item.opponent}`;
      const choice = selected[key];
      if (!choice) continue;
      product *= outcomeProb(item, choice);
    }
    return product;
  }, [fixtures, selected]);

  const count = Object.keys(selected).length;

  return (
    <div className="acca-builder">
      <p className="hero-sub">
        Select one outcome per fixture. Combined probability assumes independence (not
        betting advice).
      </p>
      <ul className="acca-list">
        {fixtures.map((item) => {
          const key = `${item.date}-${item.team}-${item.opponent}`;
          const cur = selected[key];
          return (
            <li key={key} className="acca-row">
              <div className="acca-match">
                <strong>{item.team}</strong> vs {item.opponent}
                <span className="acca-date">{item.date}</span>
              </div>
              <div className="acca-choices">
                {(
                  [
                    ["team", item.team, item.probs.teamWin],
                    ["draw", "Draw", item.probs.draw],
                    ["opp", item.opponent, item.probs.oppWin],
                  ] as const
                ).map(([id, label, pct]) => (
                  <button
                    type="button"
                    key={id}
                    className={cur === id ? "acca-btn active" : "acca-btn"}
                    aria-pressed={cur === id}
                    onClick={() =>
                      setSelected((s) => ({
                        ...s,
                        [key]: id,
                      }))
                    }
                  >
                    <span className="acca-btn-label">{label}</span>
                    <span className="acca-btn-pct">{pct.toFixed(1)}%</span>
                  </button>
                ))}
              </div>
            </li>
          );
        })}
      </ul>
      <p className="acca-total">
        {count === 0
          ? "Select at least one leg."
          : `${count} leg(s) · combined model probability ${(combined * 100).toFixed(2)}%`}
      </p>
    </div>
  );
}
