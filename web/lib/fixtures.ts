import { parseISODate, startOfToday } from "@/lib/format";
import type { Fixture } from "@/lib/types";

/** Same fixture from home or away perspective (date + club pair). */
export function fixturePairKey(item: Pick<Fixture, "date" | "team" | "opponent">): string {
  const teams = [item.team, item.opponent].sort().join("\0");
  return `${item.date.slice(0, 10)}\0${teams}`;
}

export function scoredFixtureKeys(history: Fixture[]): Set<string> {
  const keys = new Set<string>();
  for (const row of history) {
    if (row.actual || row.actualLabel) {
      keys.add(fixturePairKey(row));
    }
  }
  return keys;
}

export function excludeScoredFixtures(
  items: Fixture[],
  history: Fixture[]
): Fixture[] {
  const scored = scoredFixtureKeys(history);
  if (scored.size === 0) {
    return items;
  }
  return items.filter((item) => !scored.has(fixturePairKey(item)));
}

function isUnscoredFixture(item: Fixture | undefined): item is Fixture {
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
  return !item.actual && !item.actualLabel;
}

export function upcomingItems(items: Fixture[] | undefined): Fixture[] {
  const today = startOfToday();
  return (items ?? []).filter(
    (item) => isUnscoredFixture(item) && parseISODate(item.date) >= today
  );
}

/** Kicked-off fixtures still waiting on a result (includes upcoming ones that passed since export). */
export function awaitingItems(
  data: {
    upcoming?: Fixture[];
    awaitingResult?: Fixture[];
  },
  history: Fixture[] = []
): Fixture[] {
  const today = startOfToday();
  const scored = scoredFixtureKeys(history);
  const stillOpen = (item: Fixture) =>
    isUnscoredFixture(item) && !scored.has(fixturePairKey(item));
  const passed = (data.upcoming ?? []).filter(
    (item) => stillOpen(item) && parseISODate(item.date) < today
  );
  const awaiting = (data.awaitingResult ?? []).filter(stillOpen);
  return [...awaiting, ...passed];
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
