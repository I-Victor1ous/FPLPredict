import { LeagueSwitcher } from "@/components/LeagueSwitcher";
import { ThemeToggle } from "@/components/ThemeToggle";
import type { LeagueEntry, LeagueMeta } from "@/lib/types";

type Props = {
  leagues: LeagueEntry[];
  leagueName: string;
  meta?: LeagueMeta;
  generatedAt?: string;
};

export function SiteHeader({ leagues, leagueName, meta, generatedAt }: Props) {
  const updated = generatedAt
    ? `Updated ${new Date(generatedAt).toLocaleString()}`
    : "Updated recently";
  const tagline = `${meta?.modelLabel ?? "Auto-selected model"} · ${leagueName} 1X2`;

  return (
    <header className="site-header">
      <div className="container header-inner">
        <div className="brand">
          <span className="brand-mark">FP</span>
          <div>
            <h1>Football Predict</h1>
            <p className="tagline">{tagline}</p>
          </div>
        </div>
        <div className="header-actions">
          <ThemeToggle />
          <LeagueSwitcher leagues={leagues} />
          <p className="updated">{updated}</p>
        </div>
      </div>
    </header>
  );
}
