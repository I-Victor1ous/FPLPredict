import { promises as fs } from "fs";
import path from "path";

import type { LeagueData, Manifest } from "@/lib/types";

const DATA_DIR = path.join(process.cwd(), "public", "data");

export async function getManifest(): Promise<Manifest> {
  const raw = await fs.readFile(path.join(DATA_DIR, "manifest.json"), "utf-8");
  return JSON.parse(raw) as Manifest;
}

export async function getLeagueData(slug: string): Promise<LeagueData | null> {
  const file = path.join(DATA_DIR, `${slug}.json`);
  try {
    const raw = await fs.readFile(file, "utf-8");
    return JSON.parse(raw) as LeagueData;
  } catch {
    return null;
  }
}

export function isKnownSlug(manifest: Manifest, slug: string): boolean {
  return manifest.leagues.some((l) => l.slug === slug);
}

export function defaultSlug(manifest: Manifest): string {
  const fromDefault = manifest.leagues.find(
    (l) => l.name === manifest.defaultLeague
  )?.slug;
  return fromDefault ?? manifest.leagues[0]?.slug ?? "premier-league";
}
