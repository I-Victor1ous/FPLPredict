import { parseISODate, startOfToday } from "@/lib/format";
import type { Fixture } from "@/lib/types";

export function upcomingItems(items: Fixture[] | undefined): Fixture[] {
  const today = startOfToday();
  return (items ?? []).filter((item) => {
    if (!item?.team || !item?.opponent || item.team === item.opponent) {
      return false;
    }
    if (!item.probs) return false;
    if (
      !["teamWin", "draw", "oppWin"].every((k) =>
        Number.isFinite(Number(item.probs[k as keyof typeof item.probs]))
      )
    ) {
      return false;
    }
    if (item.actual || item.actualLabel) return false;
    return parseISODate(item.date) >= today;
  });
}

export function sortHistory(history: Fixture[]): Fixture[] {
  return [...history].sort(
    (a, b) => parseISODate(b.date).getTime() - parseISODate(a.date).getTime()
  );
}

export function sortUpcoming(upcoming: Fixture[]): Fixture[] {
  return [...upcoming].sort(
    (a, b) => parseISODate(a.date).getTime() - parseISODate(b.date).getTime()
  );
}
