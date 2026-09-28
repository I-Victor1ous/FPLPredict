"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import type { LeagueEntry } from "@/lib/types";

type Props = {
  leagues: LeagueEntry[];
};

export function LeagueSwitcher({ leagues }: Props) {
  const pathname = usePathname();

  if (leagues.length <= 1) return null;

  return (
    <nav className="league-switcher" aria-label="Choose league">
      {leagues.map((entry) => {
        const href = `/${entry.slug}`;
        const active = pathname === href;
        return (
          <Link
            key={entry.slug}
            href={href}
            className={`league-tab${active ? " is-active" : ""}`}
            aria-current={active ? "page" : undefined}
          >
            {entry.name}
          </Link>
        );
      })}
    </nav>
  );
}
