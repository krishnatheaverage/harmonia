import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/harmonia/",
  plugins: [react()],
  build: {
    outDir: "gh-pages-dist",
    emptyOutDir: true,
  },
});
