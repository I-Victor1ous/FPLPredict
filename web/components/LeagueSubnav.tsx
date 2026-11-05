import Link from "next/link";

type Props = {
  slug: string;
};

export function LeagueSubnav({ slug }: Props) {
  return (
    <nav className="league-subnav" aria-label="League sections">
      <Link href={`/${slug}`}>Picks</Link>
      <Link href={`/${slug}/picks`}>Confident</Link>
      <Link href={`/${slug}/acca`}>Acca builder</Link>
      <Link href={`/${slug}/performance`}>Evaluation</Link>
    </nav>
  );
}
