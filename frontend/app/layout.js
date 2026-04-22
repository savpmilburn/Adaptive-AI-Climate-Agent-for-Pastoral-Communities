import { Archivo } from "next/font/google";
import "./globals.css";

const archivo = Archivo({
  subsets: ["latin"],
  weight: ["400", "600"],
  variable: "--font-archivo",
});

export const metadata = {
  title: "Adaptive Climate Agent",
  description: "WeatherRisk VIPR — UGA",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className={archivo.variable}>{children}</body>
    </html>
  );
}