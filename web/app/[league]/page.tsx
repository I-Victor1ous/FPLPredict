import type { Metadata } from "next";
import { notFound } from "next/navigation";

import { FixtureCard } from "@/components/FixtureCard";
import { HistoryArchive } from "@/components/HistoryArchive";
import { HistoryTable } from "@/components/HistoryTable";
import { SiteHeader } from "@/components/SiteHeader";
import { getLeagueData, getManifest, isKnownSlug } from "@/lib/data";
import { sortHistory, sortUpcoming, upcomingItems } from "@/lib/fixtures";

const RECENT_LIMIT = 10;

type PageProps = {
  params: Promise<{ league: string }>;
};

export async function generateStaticParams() {
  const manifest = await getManifest();
  return manifest.leagues.map((l) => ({ league: l.slug }));
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { league: slug } = await params;
  const data = await getLeagueData(slug);
  const title = data?.league ?? slug;
  return { title };
}

export default async function LeaguePage({ params }: PageProps) {
  const { league: slug } = await params;
  const manifest = await getManifest();

  if (!isKnownSlug(manifest, slug)) {
    notFound();
  }

  const data = await getLeagueData(slug);
  const leagueEntry = manifest.leagues.find((l) => l.slug === slug)!;
  const displayLeague = data?.league ?? leagueEntry.name;

  if (!data) {
    return (
      <>
        <SiteHeader leagues={manifest.leagues} leagueName={leagueEntry.name} />
        <main className="container">
          <p className="empty">
            Failed to load {leagueEntry.name}. Run the weekly pipeline to refresh{" "}
            <code>web/public/data/{slug}.json</code>.
          </p>
        </main>
      </>
    );
  }

  const history = sortHistory(data.history ?? []);
  const upcoming = sortUpcoming(upcomingItems(data.upcoming));

  return (
    <>
      <SiteHeader
        leagues={manifest.leagues}
        leagueName={displayLeague}
        meta={data.meta}
        generatedAt={data.generatedAt}
      />

      <main className="container">
        <section className="hero">
          <h2>{displayLeague} — this week’s picks</h2>
          <p className="hero-sub">
            Win / draw / loss probabilities from rolling form, head-to-head, and rating
            features.
          </p>
          {data.meta?.positionsNote ? (
            <p className="hero-sub positions-note">{data.meta.positionsNote}</p>
          ) : null}
        </section>

        <HistoryArchive history={history} metrics={data.metrics} />

        <section className="fixture-grid">
          {upcoming.length === 0 ? (
            <p className="empty">No upcoming fixtures for {displayLeague}.</p>
          ) : (
            upcoming.map((item) => (
              <FixtureCard key={`${item.date}-${item.team}-${item.opponent}`} item={item} />
            ))
          )}
        </section>

        <section className="history-section">
          <h2>Recent results</h2>
          <p className="history-hint">
            Green actual = our pick matched the result. Showing the 10 most recent scored
            fixtures.
          </p>
          <HistoryTable rows={history.slice(0, RECENT_LIMIT)} />
        </section>
      </main>

      <footer className="site-footer">
        <div className="container">
          <p>Not betting advice. Model updated weekly via automated pipeline.</p>
        </div>
      </footer>

    </>
  );
}
