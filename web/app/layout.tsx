import { Analytics } from "@vercel/analytics/react";
import type { Metadata } from "next";
import { DM_Sans, JetBrains_Mono } from "next/font/google";

import { ThemeInit } from "@/components/ThemeInit";

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
    default: "Football Predict — Match Outcomes",
    template: "Football Predict — %s",
  },
  description: "Win / draw / loss probabilities from rolling form and rating features.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${dmSans.variable} ${jetbrains.variable}`} data-theme="dark">
      <body className={`${dmSans.className} ${jetbrains.variable}`}>
        <ThemeInit />
        {children}
        <Analytics />
      </body>
    </html>
  );
}
