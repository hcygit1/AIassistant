import type { Metadata } from "next";
import "./globals.css";
import { AppProvider } from "@/lib/store";

export const metadata: Metadata = {
  title: "PIPIXIA",
  description: "Local-first AI Agent with persistent memory and context management",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body className="antialiased">
        <AppProvider>{children}</AppProvider>
      </body>
    </html>
  );
}
