export type LeagueEntry = {
  name: string;
  slug: string;
};

export type Manifest = {
  generatedAt?: string;
  defaultLeague?: string;
  leagues: LeagueEntry[];
};

export type FixtureProbs = {
  teamWin: number;
  draw: number;
  oppWin: number;
};

export type Fixture = {
  date: string;
  team: string;
  opponent: string;
  pred?: string;
  pickLabel?: string;
  actual?: string | null;
  actualLabel?: string | null;
  teamPosition?: number | null;
  oppPosition?: number | null;
  probs: FixtureProbs;
  correct?: boolean | null;
};

export type LeagueMetrics = {
  n: number;
  accuracy: number;
};

export type LeagueMeta = {
  modelLabel?: string;
  positionsNote?: string;
};

export type LeagueData = {
  generatedAt?: string;
  league?: string;
  upcoming: Fixture[];
  history: Fixture[];
  metrics?: LeagueMetrics | null;
  meta?: LeagueMeta;
};
