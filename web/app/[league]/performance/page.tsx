import type { Metadata } from "next";
import { notFound } from "next/navigation";

import { LeagueSubnav } from "@/components/LeagueSubnav";
import { PerformanceCharts } from "@/components/PerformanceCharts";
import { PerformancePanel } from "@/components/PerformancePanel";
import { SiteHeader } from "@/components/SiteHeader";
import { getLeagueData, getManifest, isKnownSlug } from "@/lib/data";

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
  const name = data?.league ?? slug;
  return { title: `${name} — model evaluation` };
}

export default async function PerformancePage({ params }: PageProps) {
  const { league: slug } = await params;
  const manifest = await getManifest();
  if (!isKnownSlug(manifest, slug)) notFound();

  const data = await getLeagueData(slug);
  const leagueEntry = manifest.leagues.find((l) => l.slug === slug)!;
  const displayLeague = data?.league ?? leagueEntry.name;

  return (
    <>
      <SiteHeader
        leagues={manifest.leagues}
        leagueName={displayLeague}
        meta={data?.meta}
        generatedAt={data?.generatedAt}
      />
      <main className="container page-performance">
        <section className="hero">
          <h2>{displayLeague} — model evaluation</h2>
          <LeagueSubnav slug={slug} />
        </section>
        <PerformancePanel
          evaluation={data?.evaluation}
          metrics={data?.metrics}
          tiers={data?.confidenceTiers}
        />
        <PerformanceCharts
          weekly={data?.weeklyPerformance}
          timeline={data?.modelTimeline}
        />
      </main>

      <footer className="site-footer">
        <div className="container">
          <p>Not betting advice. Model updated weekly via automated pipeline.</p>
        </div>
      </footer>
    </>
  );
}
