# Privacy Certification — CERTIFIED WITH EXCEPTIONS

| Field | Value |
|---|---|
| Target Repo | `/Users/masa/Projects/tripbot7` |
| Commit SHA | `6e980ac0d90e5b96d9c3f5c0e2967a3d08431bd8` |
| Policy Path | `/Users/masa/Projects/tripbot7/content/privacy.md` |
| Policy SHA-256 | `5d445c9fee71e76a...` |
| Generated At | 2026-04-05 21:20:52 UTC |
| Generator Version | `3.0.76` |
| Auditor Model | `anthropic/claude-sonnet-4-5` |
| Schema Version | `1.0` |
| Content Hash | `70b07236fe6665696c25f296a584ae07` |

## Summary

| Status | Count |
|---|---|
| Pass | 0 |
| Fail | 0 |
| Insufficient Evidence | 18 |
| Manual Review | 2 |
| Ignored | 0 |
| Total | 20 |

## Claim Verdicts

### Claim `a47dd0f37640`

**Status:** [INSUFFICIENT EVIDENCE]
**Confidence:** 100%
**Evidence Count:** 10
**KG Path Present:** No

**Reasoning:** Knowledge graph path is required (require_kg_path=True) but no KG path was found in the evidence.

**Evidence:**

- **[search_hybrid]** `README.md` lines 1-1 (score: 0.000)
  - `# TripBot`
- **[search_hybrid]** `docs/research/tech-stack-validation-2026-04-04.md` lines 1-5 (score: 0.000)
  - `1. Replace Tavily with SerpAPI for event search (or use both: SerpAPI for events, Tavily for general context) 2. Install `next-auth@beta` and follow t`
- **[search_hybrid]** `docs/research/tech-stack-validation-2026-04-04.md` lines 1-1 (score: 0.000)
  - `**Tip:** Use `.describe()` on every Zod field for better extraction accuracy. AI SDK 5+ (released July 2025) introduced architectural changes -- check`
- **[search_hybrid]** `CLAUDE.md` lines 1-1 (score: 0.000)
  - `# TripBot7 — AI Travel Planner`
- **[search_hybrid]** `README.md` lines 1-8 (score: 0.000)
  - `| Method | Route | Description | |--------|-------|-------------| | POST | `/api/itinerary/parse` | Extract structured itinerary from raw text | | GET`

---

### Claim `212f4f503978`

**Status:** [INSUFFICIENT EVIDENCE]
**Confidence:** 100%
**Evidence Count:** 19
**KG Path Present:** No

**Reasoning:** Knowledge graph path is required (require_kg_path=True) but no KG path was found in the evidence.

**Evidence:**

- **[search_code]** `README.md` lines 27-27 (score: 0.000)
  - `5. Generate an AUTH_SECRET:`
- **[search_code]** `CLAUDE.md` lines 31-32 (score: 0.000)
  - `└── middleware.ts           # Auth middleware ````
- **[search_code]** `docs/research/tech-stack-validation-2026-04-04.md` lines 1-5 (score: 0.000)
  - `1. Replace Tavily with SerpAPI for event search (or use both: SerpAPI for events, Tavily for general context) 2. Install `next-auth@beta` and follow t`
- **[search_code]** `src/app/globals.css` lines 1-50 (score: 0.000)
  - `@import "tailwindcss"; @import "tw-animate-css"; @import "shadcn/tailwind.css";  @custom-variant dark (&:is(.dark *));  @theme inline {   --color-back`
- **[search_code]** `docs/research/tech-stack-validation-2026-04-04.md` lines 1-1 (score: 0.000)
  - `**Sources:** [Auth.js v5 migration guide](https://authjs.dev/getting-started/migrating-to-v5), [Next.js 15 + NextAuth v5 setup](https://codevoweb.com/`

---

### Claim `fe49cf0abab9`

**Status:** [INSUFFICIENT EVIDENCE]
**Confidence:** 100%
**Evidence Count:** 23
**KG Path Present:** No

**Reasoning:** Knowledge graph path is required (require_kg_path=True) but no KG path was found in the evidence.

**Evidence:**

- **[search_hybrid]** `docs/research/tech-stack-validation-2026-04-04.md` lines 1-5 (score: 0.000)
  - `1. Replace Tavily with SerpAPI for event search (or use both: SerpAPI for events, Tavily for general context) 2. Install `next-auth@beta` and follow t`
- **[search_hybrid]** `docs/research/tech-stack-validation-2026-04-04.md` lines 1-1 (score: 0.000)
  - `**Sources:** [AI SDK generateObject docs](https://ai-sdk.dev/docs/ai-sdk-core/generating-structured-data), [generateObject reference](https://ai-sdk.d`
- **[search_hybrid]** `CLAUDE.md` lines 1-9 (score: 0.000)
  - `## Tech Stack - **Framework**: Next.js 15 App Router (TypeScript, strict mode) - **API Layer**: Hono (mounted as catch-all in Next.js route handlers) `
- **[search_hybrid]** `README.md` lines 1-8 (score: 0.000)
  - `| Method | Route | Description | |--------|-------|-------------| | POST | `/api/itinerary/parse` | Extract structured itinerary from raw text | | GET`
- **[search_hybrid]** `docs/research/tech-stack-validation-2026-04-04.md` lines 13-13 (score: 0.000)
  - `const app = new Hono().basePath('/api')`

---

### Claim `8bab8faa0069`

**Status:** [INSUFFICIENT EVIDENCE]
**Confidence:** 100%
**Evidence Count:** 23
**KG Path Present:** No

**Reasoning:** Knowledge graph path is required (require_kg_path=True) but no KG path was found in the evidence.

**Evidence:**

- **[search_code]** `src/components/ui/dropdown-menu.tsx` lines 44-45 (score: 0.000)
  - `cn("z-50 max-h-(--available-height) w-(--anchor-width) min-w-32 origin-(--transform-origin) overflow-x-hidden overflow-y-auto rounded-lg bg-popover p-`
- **[search_code]** `src/app/globals.css` lines 51-100 (score: 0.000)
  - `:root {   --background: oklch(1 0 0);   --foreground: oklch(0.145 0 0);   --card: oklch(1 0 0);   --card-foreground: oklch(0.145 0 0);   --popover: ok`
- **[search_code]** `src/app/globals.css` lines 1-50 (score: 0.000)
  - `@import "tailwindcss"; @import "tw-animate-css"; @import "shadcn/tailwind.css";  @custom-variant dark (&:is(.dark *));  @theme inline {   --color-back`
- **[search_hybrid]** `README.md` lines 1-8 (score: 0.000)
  - `- **Next.js 15** (App Router, TypeScript, strict mode) - **Hono** mounted as API handler via catch-all route - **Auth.js v5** (NextAuth) with Google O`
- **[search_hybrid]** `docs/research/tech-stack-validation-2026-04-04.md` lines 1-1 (score: 0.000)
  - `**Tip:** Use `.describe()` on every Zod field for better extraction accuracy. AI SDK 5+ (released July 2025) introduced architectural changes -- check`

---

### Claim `24181d3ac96f`

**Status:** [INSUFFICIENT EVIDENCE]
**Confidence:** 100%
**Evidence Count:** 23
**KG Path Present:** No

**Reasoning:** Knowledge graph path is required (require_kg_path=True) but no KG path was found in the evidence.

**Evidence:**

- **[search_code]** `src/components/ui/dropdown-menu.tsx` lines 44-45 (score: 0.000)
  - `cn("z-50 max-h-(--available-height) w-(--anchor-width) min-w-32 origin-(--transform-origin) overflow-x-hidden overflow-y-auto rounded-lg bg-popover p-`
- **[search_code]** `src/app/globals.css` lines 51-100 (score: 0.000)
  - `:root {   --background: oklch(1 0 0);   --foreground: oklch(0.145 0 0);   --card: oklch(1 0 0);   --card-foreground: oklch(0.145 0 0);   --popover: ok`
- **[search_code]** `src/app/globals.css` lines 1-50 (score: 0.000)
  - `@import "tailwindcss"; @import "tw-animate-css"; @import "shadcn/tailwind.css";  @custom-variant dark (&:is(.dark *));  @theme inline {   --color-back`
- **[search_hybrid]** `README.md` lines 1-8 (score: 0.000)
  - `- **Next.js 15** (App Router, TypeScript, strict mode) - **Hono** mounted as API handler via catch-all route - **Auth.js v5** (NextAuth) with Google O`
- **[search_hybrid]** `docs/research/tech-stack-validation-2026-04-04.md` lines 1-1 (score: 0.000)
  - `**Tip:** Use `.describe()` on every Zod field for better extraction accuracy. AI SDK 5+ (released July 2025) introduced architectural changes -- check`

---

### Claim `60b654ec4220`

**Status:** [INSUFFICIENT EVIDENCE]
**Confidence:** 100%
**Evidence Count:** 11
**KG Path Present:** No

**Reasoning:** Knowledge graph path is required (require_kg_path=True) but no KG path was found in the evidence.

**Evidence:**

- **[search_hybrid]** `docs/research/tech-stack-validation-2026-04-04.md` lines 1-5 (score: 0.000)
  - `1. Replace Tavily with SerpAPI for event search (or use both: SerpAPI for events, Tavily for general context) 2. Install `next-auth@beta` and follow t`
- **[search_hybrid]** `docs/research/tech-stack-validation-2026-04-04.md` lines 1-1 (score: 0.000)
  - ``generateObject` with Zod schemas works with Anthropic models. The SDK handles schema validation and type safety automatically.`
- **[search_hybrid]** `README.md` lines 33-33 (score: 0.000)
  - `6. Run the development server:`
- **[search_hybrid]** `KUZU.md` lines 33-33 (score: 0.000)
  - `## Memory Guidelines`
- **[search_code]** `docs/research/tech-stack-validation-2026-04-04.md` lines 1-6 (score: 0.000)
  - `| API | Events Data | Geo Support | Cost | Verdict | |-----|------------|-------------|------|---------| | **SerpAPI Events** | Structured JSON | Yes `

---

### Claim `bfe93a245394`

**Status:** [INSUFFICIENT EVIDENCE]
**Confidence:** 100%
**Evidence Count:** 25
**KG Path Present:** No

**Reasoning:** Knowledge graph path is required (require_kg_path=True) but no KG path was found in the evidence.

**Evidence:**

- **[search_hybrid]** `docs/research/tech-stack-validation-2026-04-04.md` lines 1-1 (score: 0.000)
  - `**Design rationale:** Flat segments array (not nested by day) simplifies querying by date/city/country for event cache lookups. ISO country codes enab`
- **[search_hybrid]** `CLAUDE.md` lines 1-3 (score: 0.000)
  - `## Environment Variables Copy `.env.local.example` to `.env.local` and fill in values. Required: AUTH_SECRET, AUTH_GOOGLE_ID, AUTH_GOOGLE_SECRET, OPEN`
- **[search_hybrid]** `README.md` lines 1-4 (score: 0.000)
  - `4. Get API keys:    - **Anthropic**: https://console.anthropic.com/    - **Tavily**: https://app.tavily.com/    - **Vercel Blob/KV**: Create storage i`
- **[search_hybrid]** `README.md` lines 27-27 (score: 0.000)
  - `5. Generate an AUTH_SECRET:`
- **[search_code]** `docs/research/tech-stack-validation-2026-04-04.md` lines 1-5 (score: 0.000)
  - `export const GET = handle(app) export const POST = handle(app) export const PUT = handle(app) export const DELETE = handle(app) ````

---

### Claim `106a04abbf51`

**Status:** [INSUFFICIENT EVIDENCE]
**Confidence:** 100%
**Evidence Count:** 22
**KG Path Present:** No

**Reasoning:** Knowledge graph path is required (require_kg_path=True) but no KG path was found in the evidence.

**Evidence:**

- **[search_code]** `src/components/ui/dropdown-menu.tsx` lines 44-45 (score: 0.000)
  - `cn("z-50 max-h-(--available-height) w-(--anchor-width) min-w-32 origin-(--transform-origin) overflow-x-hidden overflow-y-auto rounded-lg bg-popover p-`
- **[search_code]** `src/app/globals.css` lines 51-100 (score: 0.000)
  - `:root {   --background: oklch(1 0 0);   --foreground: oklch(0.145 0 0);   --card: oklch(1 0 0);   --card-foreground: oklch(0.145 0 0);   --popover: ok`
- **[search_code]** `src/app/globals.css` lines 1-50 (score: 0.000)
  - `@import "tailwindcss"; @import "tw-animate-css"; @import "shadcn/tailwind.css";  @custom-variant dark (&:is(.dark *));  @theme inline {   --color-back`
- **[search_hybrid]** `README.md` lines 1-8 (score: 0.000)
  - `- **Next.js 15** (App Router, TypeScript, strict mode) - **Hono** mounted as API handler via catch-all route - **Auth.js v5** (NextAuth) with Google O`
- **[search_hybrid]** `docs/research/tech-stack-validation-2026-04-04.md` lines 1-1 (score: 0.000)
  - `**Tip:** Use `.describe()` on every Zod field for better extraction accuracy. AI SDK 5+ (released July 2025) introduced architectural changes -- check`

---

### Claim `e08257259bbf`

**Status:** [INSUFFICIENT EVIDENCE]
**Confidence:** 100%
**Evidence Count:** 20
**KG Path Present:** No

**Reasoning:** Knowledge graph path is required (require_kg_path=True) but no KG path was found in the evidence.

**Evidence:**

- **[search_code]** `README.md` lines 27-27 (score: 0.000)
  - `5. Generate an AUTH_SECRET:`
- **[search_code]** `CLAUDE.md` lines 31-32 (score: 0.000)
  - `└── middleware.ts           # Auth middleware ````
- **[search_code]** `docs/research/tech-stack-validation-2026-04-04.md` lines 1-5 (score: 0.000)
  - `1. Replace Tavily with SerpAPI for event search (or use both: SerpAPI for events, Tavily for general context) 2. Install `next-auth@beta` and follow t`
- **[search_code]** `src/app/globals.css` lines 1-50 (score: 0.000)
  - `@import "tailwindcss"; @import "tw-animate-css"; @import "shadcn/tailwind.css";  @custom-variant dark (&:is(.dark *));  @theme inline {   --color-back`
- **[search_code]** `docs/research/tech-stack-validation-2026-04-04.md` lines 1-1 (score: 0.000)
  - `**Sources:** [Auth.js v5 migration guide](https://authjs.dev/getting-started/migrating-to-v5), [Next.js 15 + NextAuth v5 setup](https://codevoweb.com/`

---

### Claim `f22fe10df1e8`

**Status:** [INSUFFICIENT EVIDENCE]
**Confidence:** 100%
**Evidence Count:** 22
**KG Path Present:** No

**Reasoning:** Knowledge graph path is required (require_kg_path=True) but no KG path was found in the evidence.

**Evidence:**

- **[search_code]** `src/components/ui/dropdown-menu.tsx` lines 44-45 (score: 0.000)
  - `cn("z-50 max-h-(--available-height) w-(--anchor-width) min-w-32 origin-(--transform-origin) overflow-x-hidden overflow-y-auto rounded-lg bg-popover p-`
- **[search_code]** `src/app/globals.css` lines 51-100 (score: 0.000)
  - `:root {   --background: oklch(1 0 0);   --foreground: oklch(0.145 0 0);   --card: oklch(1 0 0);   --card-foreground: oklch(0.145 0 0);   --popover: ok`
- **[search_code]** `src/app/globals.css` lines 1-50 (score: 0.000)
  - `@import "tailwindcss"; @import "tw-animate-css"; @import "shadcn/tailwind.css";  @custom-variant dark (&:is(.dark *));  @theme inline {   --color-back`
- **[search_hybrid]** `README.md` lines 1-8 (score: 0.000)
  - `- **Next.js 15** (App Router, TypeScript, strict mode) - **Hono** mounted as API handler via catch-all route - **Auth.js v5** (NextAuth) with Google O`
- **[search_hybrid]** `docs/research/tech-stack-validation-2026-04-04.md` lines 1-1 (score: 0.000)
  - `**Tip:** Use `.describe()` on every Zod field for better extraction accuracy. AI SDK 5+ (released July 2025) introduced architectural changes -- check`

---

### Claim `174679593c20`

**Status:** [INSUFFICIENT EVIDENCE]
**Confidence:** 100%
**Evidence Count:** 24
**KG Path Present:** No

**Reasoning:** Knowledge graph path is required (require_kg_path=True) but no KG path was found in the evidence.

**Evidence:**

- **[search_code]** `src/components/ui/dropdown-menu.tsx` lines 44-45 (score: 0.000)
  - `cn("z-50 max-h-(--available-height) w-(--anchor-width) min-w-32 origin-(--transform-origin) overflow-x-hidden overflow-y-auto rounded-lg bg-popover p-`
- **[search_code]** `src/app/globals.css` lines 51-100 (score: 0.000)
  - `:root {   --background: oklch(1 0 0);   --foreground: oklch(0.145 0 0);   --card: oklch(1 0 0);   --card-foreground: oklch(0.145 0 0);   --popover: ok`
- **[search_code]** `src/app/globals.css` lines 1-50 (score: 0.000)
  - `@import "tailwindcss"; @import "tw-animate-css"; @import "shadcn/tailwind.css";  @custom-variant dark (&:is(.dark *));  @theme inline {   --color-back`
- **[search_hybrid]** `README.md` lines 1-8 (score: 0.000)
  - `- **Next.js 15** (App Router, TypeScript, strict mode) - **Hono** mounted as API handler via catch-all route - **Auth.js v5** (NextAuth) with Google O`
- **[search_hybrid]** `docs/research/tech-stack-validation-2026-04-04.md` lines 1-1 (score: 0.000)
  - `**Tip:** Use `.describe()` on every Zod field for better extraction accuracy. AI SDK 5+ (released July 2025) introduced architectural changes -- check`

---

### Claim `a7ee56c348bb`

**Status:** [INSUFFICIENT EVIDENCE]
**Confidence:** 100%
**Evidence Count:** 23
**KG Path Present:** No

**Reasoning:** Knowledge graph path is required (require_kg_path=True) but no KG path was found in the evidence.

**Evidence:**

- **[search_code]** `src/components/ui/dropdown-menu.tsx` lines 44-45 (score: 0.000)
  - `cn("z-50 max-h-(--available-height) w-(--anchor-width) min-w-32 origin-(--transform-origin) overflow-x-hidden overflow-y-auto rounded-lg bg-popover p-`
- **[search_code]** `src/app/globals.css` lines 51-100 (score: 0.000)
  - `:root {   --background: oklch(1 0 0);   --foreground: oklch(0.145 0 0);   --card: oklch(1 0 0);   --card-foreground: oklch(0.145 0 0);   --popover: ok`
- **[search_code]** `src/app/globals.css` lines 1-50 (score: 0.000)
  - `@import "tailwindcss"; @import "tw-animate-css"; @import "shadcn/tailwind.css";  @custom-variant dark (&:is(.dark *));  @theme inline {   --color-back`
- **[search_hybrid]** `README.md` lines 1-8 (score: 0.000)
  - `- **Next.js 15** (App Router, TypeScript, strict mode) - **Hono** mounted as API handler via catch-all route - **Auth.js v5** (NextAuth) with Google O`
- **[search_hybrid]** `docs/research/tech-stack-validation-2026-04-04.md` lines 1-1 (score: 0.000)
  - `**Tip:** Use `.describe()` on every Zod field for better extraction accuracy. AI SDK 5+ (released July 2025) introduced architectural changes -- check`

---

### Claim `7ea956b47d69`

**Status:** [INSUFFICIENT EVIDENCE]
**Confidence:** 100%
**Evidence Count:** 25
**KG Path Present:** No

**Reasoning:** Knowledge graph path is required (require_kg_path=True) but no KG path was found in the evidence.

**Evidence:**

- **[search_code]** `src/components/ui/dropdown-menu.tsx` lines 44-45 (score: 0.000)
  - `cn("z-50 max-h-(--available-height) w-(--anchor-width) min-w-32 origin-(--transform-origin) overflow-x-hidden overflow-y-auto rounded-lg bg-popover p-`
- **[search_code]** `src/app/globals.css` lines 51-100 (score: 0.000)
  - `:root {   --background: oklch(1 0 0);   --foreground: oklch(0.145 0 0);   --card: oklch(1 0 0);   --card-foreground: oklch(0.145 0 0);   --popover: ok`
- **[search_code]** `src/app/globals.css` lines 1-50 (score: 0.000)
  - `@import "tailwindcss"; @import "tw-animate-css"; @import "shadcn/tailwind.css";  @custom-variant dark (&:is(.dark *));  @theme inline {   --color-back`
- **[search_hybrid]** `README.md` lines 1-8 (score: 0.000)
  - `- **Next.js 15** (App Router, TypeScript, strict mode) - **Hono** mounted as API handler via catch-all route - **Auth.js v5** (NextAuth) with Google O`
- **[search_hybrid]** `docs/research/tech-stack-validation-2026-04-04.md` lines 1-1 (score: 0.000)
  - `**Tip:** Use `.describe()` on every Zod field for better extraction accuracy. AI SDK 5+ (released July 2025) introduced architectural changes -- check`

---

### Claim `92fa3a460644`

**Status:** [INSUFFICIENT EVIDENCE]
**Confidence:** 100%
**Evidence Count:** 23
**KG Path Present:** No

**Reasoning:** Knowledge graph path is required (require_kg_path=True) but no KG path was found in the evidence.

**Evidence:**

- **[search_code]** `src/components/ui/dropdown-menu.tsx` lines 44-45 (score: 0.000)
  - `cn("z-50 max-h-(--available-height) w-(--anchor-width) min-w-32 origin-(--transform-origin) overflow-x-hidden overflow-y-auto rounded-lg bg-popover p-`
- **[search_code]** `src/app/globals.css` lines 51-100 (score: 0.000)
  - `:root {   --background: oklch(1 0 0);   --foreground: oklch(0.145 0 0);   --card: oklch(1 0 0);   --card-foreground: oklch(0.145 0 0);   --popover: ok`
- **[search_code]** `src/app/globals.css` lines 1-50 (score: 0.000)
  - `@import "tailwindcss"; @import "tw-animate-css"; @import "shadcn/tailwind.css";  @custom-variant dark (&:is(.dark *));  @theme inline {   --color-back`
- **[search_hybrid]** `README.md` lines 1-8 (score: 0.000)
  - `- **Next.js 15** (App Router, TypeScript, strict mode) - **Hono** mounted as API handler via catch-all route - **Auth.js v5** (NextAuth) with Google O`
- **[search_hybrid]** `docs/research/tech-stack-validation-2026-04-04.md` lines 1-1 (score: 0.000)
  - `**Tip:** Use `.describe()` on every Zod field for better extraction accuracy. AI SDK 5+ (released July 2025) introduced architectural changes -- check`

---

### Claim `3c52d6f6197b`

**Status:** [MANUAL REVIEW]
**Confidence:** 100%
**Evidence Count:** 22
**KG Path Present:** No

**Reasoning:** Claim is marked as non-testable — requires human review.

**Evidence:**

- **[search_hybrid]** `docs/research/tech-stack-validation-2026-04-04.md` lines 1-5 (score: 0.000)
  - `1. Replace Tavily with SerpAPI for event search (or use both: SerpAPI for events, Tavily for general context) 2. Install `next-auth@beta` and follow t`
- **[search_hybrid]** `docs/research/tech-stack-validation-2026-04-04.md` lines 1-1 (score: 0.000)
  - `**Sources:** [AI SDK generateObject docs](https://ai-sdk.dev/docs/ai-sdk-core/generating-structured-data), [generateObject reference](https://ai-sdk.d`
- **[search_hybrid]** `CLAUDE.md` lines 1-9 (score: 0.000)
  - `## Tech Stack - **Framework**: Next.js 15 App Router (TypeScript, strict mode) - **API Layer**: Hono (mounted as catch-all in Next.js route handlers) `
- **[search_hybrid]** `README.md` lines 1-8 (score: 0.000)
  - `| Method | Route | Description | |--------|-------|-------------| | POST | `/api/itinerary/parse` | Extract structured itinerary from raw text | | GET`
- **[search_hybrid]** `docs/research/tech-stack-validation-2026-04-04.md` lines 13-13 (score: 0.000)
  - `const app = new Hono().basePath('/api')`

---

### Claim `8d10d73fbfca`

**Status:** [INSUFFICIENT EVIDENCE]
**Confidence:** 100%
**Evidence Count:** 27
**KG Path Present:** No

**Reasoning:** Knowledge graph path is required (require_kg_path=True) but no KG path was found in the evidence.

**Evidence:**

- **[search_hybrid]** `docs/research/tech-stack-validation-2026-04-04.md` lines 1-1 (score: 0.000)
  - `**Design rationale:** Flat segments array (not nested by day) simplifies querying by date/city/country for event cache lookups. ISO country codes enab`
- **[search_hybrid]** `CLAUDE.md` lines 1-3 (score: 0.000)
  - `## Environment Variables Copy `.env.local.example` to `.env.local` and fill in values. Required: AUTH_SECRET, AUTH_GOOGLE_ID, AUTH_GOOGLE_SECRET, OPEN`
- **[search_hybrid]** `README.md` lines 1-4 (score: 0.000)
  - `4. Get API keys:    - **Anthropic**: https://console.anthropic.com/    - **Tavily**: https://app.tavily.com/    - **Vercel Blob/KV**: Create storage i`
- **[search_hybrid]** `README.md` lines 27-27 (score: 0.000)
  - `5. Generate an AUTH_SECRET:`
- **[search_code]** `docs/research/tech-stack-validation-2026-04-04.md` lines 1-5 (score: 0.000)
  - `export const GET = handle(app) export const POST = handle(app) export const PUT = handle(app) export const DELETE = handle(app) ````

---

### Claim `793a5a89f8cb`

**Status:** [INSUFFICIENT EVIDENCE]
**Confidence:** 100%
**Evidence Count:** 23
**KG Path Present:** No

**Reasoning:** Knowledge graph path is required (require_kg_path=True) but no KG path was found in the evidence.

**Evidence:**

- **[search_hybrid]** `docs/research/tech-stack-validation-2026-04-04.md` lines 1-1 (score: 0.000)
  - `**Design rationale:** Flat segments array (not nested by day) simplifies querying by date/city/country for event cache lookups. ISO country codes enab`
- **[search_hybrid]** `CLAUDE.md` lines 1-3 (score: 0.000)
  - `## Environment Variables Copy `.env.local.example` to `.env.local` and fill in values. Required: AUTH_SECRET, AUTH_GOOGLE_ID, AUTH_GOOGLE_SECRET, OPEN`
- **[search_hybrid]** `README.md` lines 1-4 (score: 0.000)
  - `4. Get API keys:    - **Anthropic**: https://console.anthropic.com/    - **Tavily**: https://app.tavily.com/    - **Vercel Blob/KV**: Create storage i`
- **[search_hybrid]** `README.md` lines 27-27 (score: 0.000)
  - `5. Generate an AUTH_SECRET:`
- **[search_code]** `docs/research/tech-stack-validation-2026-04-04.md` lines 1-5 (score: 0.000)
  - `export const GET = handle(app) export const POST = handle(app) export const PUT = handle(app) export const DELETE = handle(app) ````

---

### Claim `5b993f20ec0b`

**Status:** [INSUFFICIENT EVIDENCE]
**Confidence:** 100%
**Evidence Count:** 12
**KG Path Present:** No

**Reasoning:** Knowledge graph path is required (require_kg_path=True) but no KG path was found in the evidence.

**Evidence:**

- **[search_hybrid]** `CLAUDE.md` lines 31-32 (score: 0.000)
  - `└── middleware.ts           # Auth middleware ````
- **[search_hybrid]** `README.md` lines 27-27 (score: 0.000)
  - `5. Generate an AUTH_SECRET:`
- **[search_hybrid]** `README.md` lines 1-8 (score: 0.000)
  - `- **Next.js 15** (App Router, TypeScript, strict mode) - **Hono** mounted as API handler via catch-all route - **Auth.js v5** (NextAuth) with Google O`
- **[search_hybrid]** `README.md` lines 31-33 (score: 0.000)
  - ````bash npx auth secret ````
- **[search_code]** `src/lib/auth.ts` lines 28-33 (score: 0.000)
  - `session({ session, token }) {       if (token.sub) {         session.user.id = token.sub       }       return session     }`

---

### Claim `3b7688f266ed`

**Status:** [MANUAL REVIEW]
**Confidence:** 100%
**Evidence Count:** 23
**KG Path Present:** No

**Reasoning:** Claim is marked as non-testable — requires human review.

**Evidence:**

- **[search_hybrid]** `docs/research/tech-stack-validation-2026-04-04.md` lines 1-5 (score: 0.000)
  - `1. Replace Tavily with SerpAPI for event search (or use both: SerpAPI for events, Tavily for general context) 2. Install `next-auth@beta` and follow t`
- **[search_hybrid]** `docs/research/tech-stack-validation-2026-04-04.md` lines 1-1 (score: 0.000)
  - `**Sources:** [AI SDK generateObject docs](https://ai-sdk.dev/docs/ai-sdk-core/generating-structured-data), [generateObject reference](https://ai-sdk.d`
- **[search_hybrid]** `CLAUDE.md` lines 1-9 (score: 0.000)
  - `## Tech Stack - **Framework**: Next.js 15 App Router (TypeScript, strict mode) - **API Layer**: Hono (mounted as catch-all in Next.js route handlers) `
- **[search_hybrid]** `README.md` lines 1-8 (score: 0.000)
  - `| Method | Route | Description | |--------|-------|-------------| | POST | `/api/itinerary/parse` | Extract structured itinerary from raw text | | GET`
- **[search_hybrid]** `docs/research/tech-stack-validation-2026-04-04.md` lines 13-13 (score: 0.000)
  - `const app = new Hono().basePath('/api')`

---

### Claim `4756e9d3a81e`

**Status:** [INSUFFICIENT EVIDENCE]
**Confidence:** 100%
**Evidence Count:** 20
**KG Path Present:** No

**Reasoning:** Knowledge graph path is required (require_kg_path=True) but no KG path was found in the evidence.

**Evidence:**

- **[search_hybrid]** `CLAUDE.md` lines 1-6 (score: 0.000)
  - `## Key Patterns - Hono is mounted at `/api/[...route]` via `hono/vercel` handle() - AI extraction uses `generateObject` with Zod schemas for structure`
- **[search_hybrid]** `README.md` lines 50-52 (score: 0.000)
  - `## Deploy  ```bash vercel ````
- **[search_hybrid]** `KUZU.md` lines 1-5 (score: 0.000)
  - `### Available Commands: - `kuzu-memory enhance <prompt>` - Enhance prompts with project context - `kuzu-memory learn <content>` - Store learning from `
- **[search_hybrid]** `docs/research/tech-stack-validation-2026-04-04.md` lines 1-3 (score: 0.000)
  - `**Date:** 2026-04-04 **Status:** Complete **Type:** Informational`
- **[search_code]** `src/lib/storage/blob.ts` lines 53-63 (score: 0.000)
  - `async function deleteTrip(   userId: string,   tripId: string ): Promise<void> {   const path = tripPath(userId, tripId)   const { blobs } = await lis`

---
