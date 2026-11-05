import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";

import { FixtureCard } from "@/components/FixtureCard";
import { LeagueSubnav } from "@/components/LeagueSubnav";
import { SiteHeader } from "@/components/SiteHeader";
import { getLeagueData, getManifest, isKnownSlug } from "@/lib/data";

type PageProps = { params: Promise<{ league: string }> };

export async function generateStaticParams() {
  const manifest = await getManifest();
  return manifest.leagues.map((l) => ({ league: l.slug }));
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { league: slug } = await params;
  const data = await getLeagueData(slug);
  return { title: `${data?.league ?? slug} — confident picks` };
}

export default async function PicksPage({ params }: PageProps) {
  const { league: slug } = await params;
  const manifest = await getManifest();
  if (!isKnownSlug(manifest, slug)) notFound();
  const data = await getLeagueData(slug);
  const leagueEntry = manifest.leagues.find((l) => l.slug === slug)!;
  const picks = data?.confidentPicks ?? [];

  return (
    <>
      <SiteHeader
        leagues={manifest.leagues}
        leagueName={data?.league ?? leagueEntry.name}
        meta={data?.meta}
        generatedAt={data?.generatedAt}
      />
      <main className="container">
        <section className="hero">
          <h2>Confident picks</h2>
          <p className="hero-sub">
            Fixtures where the model&apos;s top outcome is at least 55% (configurable at
            export).
          </p>
          <LeagueSubnav slug={slug} />
        </section>
        <section className="fixture-grid">
          {picks.length === 0 ? (
            <p className="empty">No confident picks this week.</p>
          ) : (
            picks.map((item) => (
              <FixtureCard key={`${item.date}-${item.team}-${item.opponent}`} item={item} />
            ))
          )}
        </section>
        <p className="hero-sub">
          <Link href={`/${slug}`}>← All fixtures</Link>
        </p>
      </main>
    </>
  );
}
