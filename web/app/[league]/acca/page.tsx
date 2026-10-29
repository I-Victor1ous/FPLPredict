import type { Metadata } from "next";
import { notFound } from "next/navigation";

import { AccaBuilder } from "@/components/AccaBuilder";
import { LeagueSubnav } from "@/components/LeagueSubnav";
import { SiteHeader } from "@/components/SiteHeader";
import { getLeagueData, getManifest, isKnownSlug } from "@/lib/data";
import { sortUpcoming, upcomingItems } from "@/lib/fixtures";

type PageProps = { params: Promise<{ league: string }> };

export async function generateStaticParams() {
  const manifest = await getManifest();
  return manifest.leagues.map((l) => ({ league: l.slug }));
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { league: slug } = await params;
  const data = await getLeagueData(slug);
  return { title: `${data?.league ?? slug} — acca builder` };
}

export default async function AccaPage({ params }: PageProps) {
  const { league: slug } = await params;
  const manifest = await getManifest();
  if (!isKnownSlug(manifest, slug)) notFound();
  const data = await getLeagueData(slug);
  const leagueEntry = manifest.leagues.find((l) => l.slug === slug)!;
  const fixtures = sortUpcoming(upcomingItems(data?.upcoming));

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
          <h2>Accumulator builder</h2>
          <LeagueSubnav slug={slug} />
        </section>
        <AccaBuilder fixtures={fixtures} />
      </main>
    </>
  );
}
