import { redirect } from "next/navigation";

import { defaultSlug, getManifest } from "@/lib/data";

export default async function HomePage() {
  const manifest = await getManifest();
  redirect(`/${defaultSlug(manifest)}`);
}
