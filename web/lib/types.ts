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
  marketProbs?: FixtureProbs;
  correct?: boolean | null;
  topProb?: number;
};

export type RollingBacktestSeason = {
  seasonEnd: number;
  n: number;
  accuracy: number;
  log_loss: number;
  rps?: number;
  market_accuracy?: number;
  market_log_loss?: number;
};

export type LeagueMetrics = {
  n: number;
  accuracy: number | null;
  ci95Low?: number | null;
  ci95High?: number | null;
};

export type ConfidenceTier = {
  minTopProb: number;
  n: number;
  accuracy: number;
};

export type ScoreRow = {
  model: string;
  accuracy: number;
  log_loss: number;
  rps?: number;
  n?: number;
};

export type MarketBaseline = {
  /** Fixtures in the held-out test set. */
  test_n?: number;
  /** Test fixtures with bookmaker odds available. */
  n: number;
  coverage?: number;
  accuracy: number;
  log_loss: number;
  rps?: number;
  model_accuracy?: number;
  model_log_loss?: number;
  model_rps?: number;
};

export type CalibrationBin = {
  low: number;
  high: number;
  n: number;
  meanPredicted: number;
  observed: number;
};

export type LeagueEvaluation = {
  testMetrics?: ScoreRow & {
    home_win_baseline_accuracy?: number;
    brier?: number;
    calibration?: CalibrationBin[];
    market_baseline?: MarketBaseline;
  };
  rollingBacktest?: RollingBacktestSeason[] | null;
};

export type PipelineLeagueStatus = {
  name: string;
  slug: string;
  status: "ok" | "failed";
  error?: string | null;
  newestCompletedDate?: string | null;
  xgMissingRate?: number | null;
  upcomingFixtures?: number | null;
};

export type PipelineStatus = {
  runAt: string;
  ok: boolean;
  leagues: PipelineLeagueStatus[];
};

export type WeeklyPerformanceWeek = {
  gameweek?: string;
  weekStart: string;
  /** FBref season end-year (e.g. 2027 = 2026–27). */
  seasonEnd?: number | null;
  n: number;
  correct: number;
  accuracy: number | null;
  modelStrategy?: string | null;
  modelLabel?: string | null;
};

export type ModelTimelineSegment = {
  from: string;
  to?: string | null;
  strategy: string;
  label?: string;
  note?: string;
};

export type LeagueMeta = {
  modelLabel?: string;
  positionsNote?: string;
  modelTestAccuracy?: number;
  modelTestRps?: number;
};

export type LeagueData = {
  generatedAt?: string;
  league?: string;
  upcoming: Fixture[];
  /** Played fixtures whose result has not been scraped yet. */
  awaitingResult?: Fixture[];
  history: Fixture[];
  metrics?: LeagueMetrics | null;
  confidenceTiers?: ConfidenceTier[];
  weeklyPerformance?: WeeklyPerformanceWeek[];
  modelTimeline?: ModelTimelineSegment[];
  evaluation?: LeagueEvaluation;
  confidentPicks?: Fixture[];
  meta?: LeagueMeta;
};
