import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "HelioShield — Space Weather & Satellite Disruption Control Center",
  description: "Real-time AI & ML Solar Storm Forecasting & Operational Disruption Intelligence",
  icons: {
    icon: "/favicon.ico",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="bg-slate-950 text-slate-100 antialiased selection:bg-cyan-500/20 selection:text-cyan-300">
        {children}
      </body>
    </html>
  );
}
