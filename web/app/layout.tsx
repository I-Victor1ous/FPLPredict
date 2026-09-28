import { Analytics } from "@vercel/analytics/react";
import type { Metadata } from "next";
import { DM_Sans, JetBrains_Mono } from "next/font/google";

import "./globals.css";

const dmSans = DM_Sans({
  subsets: ["latin"],
  variable: "--font-dm-sans",
});

const jetbrains = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains",
});

export const metadata: Metadata = {
  title: {
    default: "FPLPredict — Match Outcomes",
    template: "FPLPredict — %s",
  },
  description: "Win / draw / loss probabilities from rolling form and rating features.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${dmSans.variable} ${jetbrains.variable}`}>
      <body className={`${dmSans.className} ${jetbrains.variable}`}>
        {children}
        <Analytics />
      </body>
    </html>
  );
}
