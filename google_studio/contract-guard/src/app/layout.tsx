import type { Metadata } from "next";
import { ToastProvider } from '@/components/Toast';
import "./globals.css";

export const metadata: Metadata = {
  title: "ContractGuard - AI-Powered Smart Contract Security",
  description: "Analyze smart contracts for security risks using Gemini AI. Get human-readable insights, exploit narratives, and mitigation recommendations.",
  keywords: ["smart contract", "security", "audit", "blockchain", "AI", "Gemini", "Web3"],
  authors: [{ name: "ContractGuard" }],
  openGraph: {
    title: "ContractGuard - AI-Powered Smart Contract Security",
    description: "Analyze smart contracts for security risks using Gemini AI",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link 
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" 
          rel="stylesheet" 
        />
      </head>
      <body className="bg-premium antialiased">
        <ToastProvider>
          <div className="relative z-10">
            {children}
          </div>
        </ToastProvider>
      </body>
    </html>
  );
}
