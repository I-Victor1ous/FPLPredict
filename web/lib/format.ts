export function parseISODate(iso: string): Date {
  return new Date(`${iso}T12:00:00`);
}

export function startOfToday(): Date {
  const d = new Date();
  d.setHours(0, 0, 0, 0);
  return d;
}

export function formatDate(iso: string): string {
  return parseISODate(iso).toLocaleDateString(undefined, {
    weekday: "short",
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

export function pickClass(pickLabel: string): string {
  if (!pickLabel) return "";
  return pickLabel === "Draw" ? "draw" : "win";
}

export function formatPosition(pos: number | null | undefined): string {
  return pos != null && pos > 0 ? `#${pos}` : "—";
}

export function topProb(
  probs: { teamWin: number; draw: number; oppWin: number },
  team: string,
  opponent: string
): [string, number] {
  const entries: [string, number][] = [
    [team, probs.teamWin],
    ["Draw", probs.draw],
    [opponent, probs.oppWin],
  ];
  entries.sort((a, b) => b[1] - a[1]);
  return entries[0];
}

/** Display date like "20 Sep 2026" from an ISO timestamp or YYYY-MM-DD. */
export function formatDataAsOf(iso: string | null | undefined): string | null {
  if (!iso) return null;
  const d = new Date(iso.length <= 10 ? `${iso}T12:00:00Z` : iso);
  if (Number.isNaN(d.getTime())) return null;
  return d.toLocaleDateString("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  });
}
