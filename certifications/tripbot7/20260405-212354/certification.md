# Privacy Certification — CERTIFIED WITH EXCEPTIONS

| Field | Value |
|---|---|
| Target Repo | `/Users/masa/Projects/tripbot7` |
| Commit SHA | `6e980ac0d90e5b96d9c3f5c0e2967a3d08431bd8` |
| Policy Path | `/Users/masa/Projects/tripbot7/content/privacy.md` |
| Policy SHA-256 | `5d445c9fee71e76a...` |
| Generated At | 2026-04-05 21:23:54 UTC |
| Generator Version | `3.0.76` |
| Auditor Model | `anthropic/claude-sonnet-4-5` |
| Schema Version | `1.0` |
| Content Hash | `13df21c851c6553fea1ea9ce9472a8cf` |

## Summary

| Status | Count |
|---|---|
| Pass | 6 |
| Fail | 0 |
| Insufficient Evidence | 10 |
| Manual Review | 4 |
| Ignored | 0 |
| Total | 20 |

## Claim Verdicts

### Claim `a47dd0f37640`

**Status:** [INSUFFICIENT EVIDENCE]
**Confidence:** 30%
**Evidence Count:** 10
**KG Path Present:** No

**Reasoning:** The claim states the application collects Google Profile information (name, email, profile image) through Google OAuth authentication when using tripbot.tours. The evidence shows references to authentication setup (AUTH_SECRET, Auth.js v5, Google OAuth mentions, auth middleware) but does not contain actual code implementation showing what data is collected from Google OAuth or how it is stored. No code snippets demonstrate the OAuth callback handler, session management, or database schema that would confirm the specific data fields (name, email, profile image) are actually collected. Without seeing the actual authentication implementation code, user model/schema, or OAuth configuration details, there is insufficient evidence to verify this data collection claim.

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

**Status:** [MANUAL REVIEW]
**Confidence:** 30%
**Evidence Count:** 19
**KG Path Present:** No

**Reasoning:** The claim states that trip data and user preferences are stored as encrypted private JSON files in Vercel Blob. Evidence shows the system uses Vercel Blob for storage (CLAUDE.md mentions 'Blob storage paths: trips/{userId}/{tripId}.json' and 'Vercel Blob (private JSON for itineraries/user data)'). However, there is no code evidence demonstrating encryption implementation at the application level. The search queries for encryption libraries (AES, cryptography, hashlib, fernet, etc.) returned no relevant matches. While Vercel Blob may provide encryption at rest as a platform feature, the evidence does not confirm this, nor does it show application-level encryption being implemented in the codebase. The 'private' designation in the claim could refer to access control rather than encryption. Without access to actual storage configuration code or Vercel Blob setup, and with no encryption implementation visible in the source code, this requires manual review to verify whether Vercel Blob's platform-level encryption satisfies the claim or if additional application-level encryption was intended but not implemented.

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
**Confidence:** 30%
**Evidence Count:** 23
**KG Path Present:** No

**Reasoning:** The evidence shows the application uses third-party services (OpenRouter/Claude for AI, Tavily/SerpAPI for search, Vercel Blob/KV for storage, Google OAuth for authentication) and makes API calls to these services. However, there is no code evidence that directly demonstrates whether user data is 'sold' to third parties versus simply using third-party services for functionality. The distinction between selling data and using third-party APIs for service delivery requires examining business agreements, data processing terms, and monetization practices that are not visible in the source code. The evidence shows data is sent to external services (AI processing, event search), but cannot determine if this constitutes 'selling' data or legitimate service provision. This claim requires manual review of business practices, privacy policies, and third-party agreements beyond what source code can reveal.

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

**Status:** [PASS]
**Confidence:** 92%
**Evidence Count:** 23
**KG Path Present:** No

**Reasoning:** The claim states that the service integrates with Google OAuth for authentication and account management. Multiple pieces of evidence strongly support this: (1) README.md line 3 explicitly mentions 'Auth.js v5 (NextAuth) with Google OAuth', (2) CLAUDE.md lines 4 and 8 confirm 'Auth.js v5 (NextAuth) with Google OAuth' and show project structure with 'app/api/auth/[...nextauth]/' directory for Auth.js handler and 'app/login/' for Google OAuth login, (3) README.md line 27 references setting up Google OAuth credentials at Google Cloud Console, (4) CLAUDE.md line 3 lists AUTH_GOOGLE_ID and AUTH_GOOGLE_SECRET as required environment variables, (5) package.json shows '@auth/core' dependency. The evidence consistently demonstrates Google OAuth integration for authentication across multiple documentation files and code structure. No evidence of other third-party integrations contradicts the claim's scope.

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
**Confidence:** 30%
**Evidence Count:** 23
**KG Path Present:** No

**Reasoning:** The claim states that each third-party provider operates under their own privacy policy. The evidence shows the application integrates multiple third-party services (Auth.js/Google OAuth, Vercel Blob/KV, Tavily Search, Anthropic/OpenRouter AI SDK, etc.) as documented in README.md, CLAUDE.md, and package.json. However, none of the code snippets provided demonstrate whether the application acknowledges, links to, or discloses these third-party privacy policies to users. The evidence only confirms third-party integrations exist but does not show how privacy policies are communicated or handled. To verify this claim, we would need to see privacy policy pages, terms of service documentation, user-facing disclosures, or configuration that references third-party privacy policies. The current evidence is insufficient to determine compliance with this privacy claim.

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
**Confidence:** 30%
**Evidence Count:** 11
**KG Path Present:** No

**Reasoning:** The evidence provided does not contain relevant information about data minimization practices. The snippets show documentation about API selection, schema definitions, and storage operations, but do not demonstrate what data is actually sent to external services or whether it is minimized. To verify this claim, we would need to see actual API call implementations showing request payloads, data filtering logic, or explicit data minimization controls. The current evidence is tangential (API documentation, schema definitions) and does not address the core claim about sending minimum necessary data to services.

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
**Confidence:** 30%
**Evidence Count:** 25
**KG Path Present:** No

**Reasoning:** The claim states that the application allows users to request complete deletion of their account and all associated data. Evidence shows a `deleteTrip` function in blob.ts (lines 53-63) that can delete individual trips, but no evidence of account-level deletion functionality was found. The search results include API routes for trips (GET, POST for /api/trips) but no DELETE endpoint for user accounts. Auth configuration is present but doesn't show account deletion capabilities. Without evidence of a user account deletion endpoint or mechanism that removes all user data including authentication records, the claim cannot be verified. The presence of trip deletion is insufficient to confirm complete account deletion functionality.

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

**Status:** [PASS]
**Confidence:** 75%
**Evidence Count:** 22
**KG Path Present:** No

**Reasoning:** The evidence shows no presence of advertising, tracking, or third-party marketing cookies in the codebase. Multiple searches for common analytics/tracking services (analytics, segment, amplitude, mixpanel, google_analytics, stripe, twilio, sendgrid, datadog, sentry, facebook, twitter, linkedin) returned zero relevant matches. The package.json shows only functional third-party services: Auth.js for authentication, Vercel services for infrastructure, Tavily/SerpAPI for search functionality, and OpenRouter for AI - none of which are advertising or marketing tracking services. The only 'tracking' reference found was in UI styling code (tracking-widest for text spacing), not tracking cookies. The tech stack documentation confirms the application uses Google OAuth for authentication only, not for advertising purposes. While this is primarily absence of evidence, the comprehensive search queries across multiple relevant terms and the clean package.json dependencies provide reasonable confidence that no advertising or tracking cookies are implemented.

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

**Status:** [MANUAL REVIEW]
**Confidence:** 30%
**Evidence Count:** 20
**KG Path Present:** No

**Reasoning:** The claim states that trip data and user preferences stored in Vercel Blob are encrypted. Evidence shows the application uses Vercel Blob for storing private JSON files (trips/{userId}/{tripId}.json), but there is no code evidence demonstrating encryption implementation. The search queries for encryption-related terms (AES, TLS, cryptography, hashlib, fernet, etc.) returned no relevant matches in the codebase. While Vercel Blob may provide encryption at rest by default as a platform feature, this is a configuration/infrastructure concern outside the application code. The evidence shows storage paths and usage patterns but no explicit encryption implementation in the application layer. Without access to Vercel Blob configuration or documentation confirming default encryption, and without code implementing application-level encryption, there is insufficient evidence to confirm the claim. This requires manual review of Vercel Blob's security features and deployment configuration.

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

**Status:** [PASS]
**Confidence:** 92%
**Evidence Count:** 22
**KG Path Present:** No

**Reasoning:** Multiple pieces of evidence confirm Google OAuth is used for authentication and account management. Evidence 4 (README.md) explicitly states 'Auth.js v5 (NextAuth) with Google OAuth'. Evidence 6 (CLAUDE.md) confirms 'Auth: Auth.js v5 (NextAuth) with Google OAuth'. Evidence 21 shows setup instructions for Google OAuth credentials at Google Cloud Console. Evidence 12 shows the project structure includes 'app/api/auth/[...nextauth]/' for Auth.js handler and 'app/login/' for Google OAuth login. Evidence 13 lists required environment variables including AUTH_GOOGLE_ID and AUTH_GOOGLE_SECRET. Evidence 20 shows authentication middleware protecting routes. The consistent documentation across multiple files and the presence of Google OAuth-specific configuration variables provide strong direct evidence that Google OAuth is indeed used for authentication and account management as claimed.

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

**Status:** [PASS]
**Confidence:** 85%
**Evidence Count:** 24
**KG Path Present:** No

**Reasoning:** The claim states that itinerary text is sent to OpenRouter and Anthropic for AI processing. Evidence 6 (CLAUDE.md lines 1-9) explicitly documents 'AI: Vercel AI SDK + OpenRouter (Claude Haiku via OpenRouter)' as part of the tech stack. Evidence 8 (package.json) shows the dependency '@openrouter/ai-sdk-provider': '^2.3.3' confirming OpenRouter integration. Evidence 4 (README.md) states 'Vercel AI SDK (@ai-sdk/anthropic) for entity extraction with Claude Haiku', confirming Anthropic's Claude model is used. Evidence 21-22 describe the application as an 'AI-powered travel planner' where users 'paste your itinerary' for processing. The combination of documented architecture, installed dependencies, and stated functionality provides strong evidence that itinerary text is indeed sent to OpenRouter/Anthropic for AI processing as claimed.

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

**Status:** [PASS]
**Confidence:** 92%
**Evidence Count:** 23
**KG Path Present:** No

**Reasoning:** Multiple pieces of evidence confirm Tavily is used for event and activity search. Evidence 4 (README.md) explicitly states 'Tavily Search for hyperlocal event discovery'. Evidence 6 (CLAUDE.md) lists 'Search: Tavily Search API for hyperloca[l]'. Evidence 8 (package.json) shows '@tavily/core' as a dependency. Evidence 21 and 22 from tech-stack-validation document discuss 'Tavily for Hyperlocal Event Search' and 'Event Search Alternatives', confirming its intended use case. The claim that Tavily is used for real-time event and activity search is directly supported by the codebase documentation and dependencies.

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
**Confidence:** 95%
**Evidence Count:** 25
**KG Path Present:** No

**Reasoning:** The claim states that Mapbox is used for map display and travel time calculations. However, none of the 25 evidence snippets contain any references to Mapbox integration, Mapbox SDK imports, Mapbox API calls, or Mapbox configuration. The evidence shows the tech stack includes Next.js, Hono, Auth.js, Vercel services, Tavily Search, and OpenRouter/Claude for AI, but no Mapbox. Evidence 21-25 searched specifically for 'Mapbox map travel time third party integration vendor' but returned only generic trip-related UI components and navigation elements with no Mapbox references. The package.json (Evidence 8) lists dependencies but shows no Mapbox packages. Without any code evidence of Mapbox usage, we cannot confirm the claim is implemented in the codebase.

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

**Status:** [PASS]
**Confidence:** 92%
**Evidence Count:** 23
**KG Path Present:** No

**Reasoning:** Multiple pieces of evidence clearly demonstrate Vercel is used for all three claimed purposes: (1) Application hosting - Evidence 22 shows 'vercel' deployment command and Evidence 4,6 confirm Next.js deployment context; (2) Data storage - Evidence 4 explicitly states 'Vercel Blob for private JSON storage (itineraries, user profiles)' and Evidence 8 shows '@vercel/blob' package dependency, Evidence 23 confirms 'Blob storage paths: trips/{userId}/{tripId}.json'; (3) Caching - Evidence 4 states 'Vercel KV (Redis) for event cache', Evidence 8 shows '@vercel/kv' package dependency, and Evidence 23 confirms 'Events cached in Vercel KV with TTL, keyed as events:{date}:{country}:{city}'. The evidence directly supports all three aspects of the claim with concrete implementation details.

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
**Confidence:** 85%
**Evidence Count:** 27
**KG Path Present:** No

**Reasoning:** The evidence shows a deleteTrip function in blob.ts (lines 53-63) that can delete individual trips, but there is no evidence of functionality to delete a complete user account and all associated data. The code shows trip management (save, get, list, delete trips) and authentication, but no user account deletion endpoint or mechanism. The API routes documented include trip operations but no account deletion endpoint. To fulfill the claim, there should be evidence of: (1) an API endpoint for account deletion requests, (2) logic to delete all user data across all storage systems, and (3) cascading deletion of all user-associated records. None of this evidence is present in the provided code snippets.

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
**Confidence:** 85%
**Evidence Count:** 23
**KG Path Present:** No

**Reasoning:** The evidence shows API endpoints for trip management (GET /api/trips, GET /api/trips/:id, POST /api/trips) and storage functions (saveTrip, getTrip, listTrips, deleteTrip) but no implementation of data export functionality in a standard format. While the code can retrieve individual trips (GET /api/trips/:id returns JSON), there is no dedicated endpoint or function for users to request a comprehensive export of all their trip data in a portable standard format (e.g., JSON, CSV, XML download). The claim specifically mentions users can 'request a copy' which implies a self-service export feature, but no such feature is evident in the codebase. The existing GET endpoints appear to be for application functionality rather than data portability/export purposes.

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
**Confidence:** 30%
**Evidence Count:** 12
**KG Path Present:** No

**Reasoning:** The evidence shows the application uses Auth.js v5 (NextAuth) for authentication with Google OAuth, and there are references to session management in auth.ts. However, there is no direct evidence about cookie configuration, cookie types, or explicit exclusion of advertising/tracking/third-party marketing cookies. The search results do not include cookie configuration code, middleware that sets cookies, or any code that would demonstrate the absence of tracking cookies. Without examining actual cookie-setting code, HTTP headers, or explicit cookie configuration (e.g., session cookie settings, cookie attributes, or absence of third-party cookie integrations), we cannot verify the claim that only authentication session cookies are used and that no advertising, tracking, or marketing cookies exist.

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
**Confidence:** 45%
**Evidence Count:** 20
**KG Path Present:** No

**Reasoning:** Evidence 1 confirms events are cached in Vercel KV with TTL (keyed as 'events:{date}:{country}:{city}'), and Evidence 8 shows cache retrieval logic that filters expired events based on 'expiresAt' timestamps. However, critical information is missing: (1) No code showing the actual TTL value or setCachedEvents implementation to verify automatic expiration is configured, (2) No evidence examining the cached event data structure to confirm it contains no personally identifiable information (PII). The claim has two parts - automatic expiration and non-PII data - but we only have partial evidence for expiration mechanism and no evidence about data content. The cache key structure suggests location-based data rather than user-specific data, but this is circumstantial. More code evidence is needed to verify both TTL configuration and data anonymization.

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
