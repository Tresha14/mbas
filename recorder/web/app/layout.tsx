import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MBAS Recorder",
  description: "Dataset recording app for the Multimodal Behavioural Analysis System",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">{children}</body>
    </html>
  );
}
