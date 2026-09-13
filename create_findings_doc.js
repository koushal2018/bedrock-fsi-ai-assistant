const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageNumber, PageBreak, LevelFormat,
  ExternalHyperlink, TabStopType, TabStopPosition
} = require("docx");
const fs = require("fs");

// ── Helpers ────────────────────────────────────────────────────────────────
const BLUE        = "1F497D";
const DARK_BLUE   = "0D2B5E";
const LIGHT_BLUE  = "D5E8F0";
const MID_BLUE    = "BDD7EE";
const GREEN       = "E2EFDA";
const AMBER       = "FFF2CC";
const RED_LIGHT   = "FCE4D6";
const GREY_LIGHT  = "F5F5F5";
const GREY_DARK   = "595959";
const WHITE       = "FFFFFF";
const BORDER_COL  = "BFBFBF";

const border = { style: BorderStyle.SINGLE, size: 1, color: BORDER_COL };
const borders = { top: border, bottom: border, left: border, right: border };
const noBorder = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const noBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder };

function cell(text, { fill = WHITE, bold = false, color = "000000", size = 18, align = AlignmentType.LEFT, width = 2340, verticalAlign = VerticalAlign.CENTER } = {}) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill, type: ShadingType.CLEAR },
    verticalAlign,
    margins: { top: 60, bottom: 60, left: 120, right: 120 },
    children: [new Paragraph({
      alignment: align,
      children: [new TextRun({ text: String(text), bold, size, color, font: "Arial" })]
    })]
  });
}

function hCell(text, { fill = DARK_BLUE, width = 2340 } = {}) {
  return cell(text, { fill, bold: true, color: WHITE, size: 18, align: AlignmentType.CENTER, width });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 120 },
    children: [new TextRun({ text, bold: true, size: 36, color: DARK_BLUE, font: "Arial" })]
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 80 },
    children: [new TextRun({ text, bold: true, size: 28, color: BLUE, font: "Arial" })]
  });
}
function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 200, after: 60 },
    children: [new TextRun({ text, bold: true, size: 24, color: GREY_DARK, font: "Arial" })]
  });
}
function para(text, { bold = false, size = 20, color = "000000", spacing = 120 } = {}) {
  return new Paragraph({
    spacing: { after: spacing },
    children: [new TextRun({ text, bold, size, color, font: "Arial" })]
  });
}
function bullet(text, { bold = false } = {}) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 60 },
    children: [new TextRun({ text, bold, size: 20, font: "Arial" })]
  });
}
function numbered(text) {
  return new Paragraph({
    numbering: { reference: "numbers", level: 0 },
    spacing: { after: 60 },
    children: [new TextRun({ text, size: 20, font: "Arial" })]
  });
}
function pageBreak() { return new Paragraph({ children: [new PageBreak()] }); }
function spacer(n = 1) { return new Paragraph({ spacing: { after: n * 200 }, children: [new TextRun("")] }); }
function divider() {
  return new Paragraph({
    spacing: { before: 120, after: 120 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: BLUE } },
    children: [new TextRun("")]
  });
}
function callout(text, fill = LIGHT_BLUE) {
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [9360],
    rows: [new TableRow({ children: [new TableCell({
      borders: noBorders,
      width: { size: 9360, type: WidthType.DXA },
      shading: { fill, type: ShadingType.CLEAR },
      margins: { top: 120, bottom: 120, left: 240, right: 240 },
      children: [new Paragraph({ children: [new TextRun({ text, size: 20, font: "Arial", italics: true })] })]
    })] })]
  });
}

// ── Leaderboard table helpers ───────────────────────────────────────────────
function leaderboardTable(rows, colWidths, headers) {
  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) => hCell(h, { width: colWidths[i] }))
  });
  const dataRows = rows.map(({ data, highlight = WHITE }) =>
    new TableRow({
      children: data.map((v, i) => cell(v, { width: colWidths[i], fill: highlight, size: 18 }))
    })
  );
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [headerRow, ...dataRows]
  });
}

// ── Load all model data from JSON export ────────────────────────────────────
const ALL_MODELS = JSON.parse(fs.readFileSync("/tmp/all_models_data.json", "utf8"));

function rowHighlight(m) {
  if (m.status === "error") return RED_LIGHT;
  if (!m.guardrail_overall) return WHITE;         // pending - no colour
  if (m.guardrail_overall >= 1.0) return GREEN;
  if (m.guardrail_overall >= 0.8) return WHITE;
  return AMBER;
}

function fmtAcc(m)  { return m.accuracy !== null ? `${Math.round(m.accuracy * 100)}%` : "Error"; }
function fmtGrd(m)  { return m.guardrail_overall !== null ? `${Math.round(m.guardrail_overall * 100)}%` : (m.status === "error" ? "—" : "Pending"); }
function fmtP50(m)  { return m.p50  !== null ? `${m.p50}s`  : "N/A"; }
function fmtP95(m)  { return m.p95  !== null ? `${m.p95}s`  : "N/A"; }
function fmtCost(m) { return `$${m.cost_100q.toFixed(3)}`; }

// Guardrail-tested models sorted by accuracy desc
const TESTED    = ALL_MODELS.filter(m => m.guardrail_overall !== null).sort((a,b) => (b.accuracy||0)-(a.accuracy||0));
// All other models with results, sorted by accuracy desc
const UNTESTED  = ALL_MODELS.filter(m => m.guardrail_overall === null && m.accuracy !== null).sort((a,b) => (b.accuracy||0)-(a.accuracy||0));
// Models with API errors
const ERRORS    = ALL_MODELS.filter(m => m.accuracy === null);

const COMPOSITE_DATA = TESTED.map((m,i) => ({
  data: [String(i+1), m.model_id, fmtAcc(m), fmtGrd(m), fmtP50(m), fmtP95(m), fmtCost(m)],
  highlight: rowHighlight(m)
}));

const GUARDRAIL_DATA = [
  { data: ["glm-5",             "100%", "✓ 100%", "✓ 100%", "✓ 100%"], highlight: GREEN },
  { data: ["claude-sonnet-4-6", "100%", "✓ 100%", "✓ 100%", "✓ 100%"], highlight: GREEN },
  { data: ["llama3-70b",        "100%", "✓ 100%", "✓ 100%", "✓ 100%"], highlight: GREEN },
  { data: ["mistral-large-3",   "90%",  "✓",      "✓",      "⚠ 67%"],  highlight: WHITE },
  { data: ["kimi-k2.5",         "90%",  "✓",      "⚠",      "✓"],      highlight: WHITE },
  { data: ["gemma-3-27b",       "90%",  "✓",      "✓",      "⚠"],      highlight: WHITE },
  { data: ["qwen3-80b",         "80%",  "⚠",      "✓",      "✓"],      highlight: WHITE },
  { data: ["deepseek-v3.2",     "80%",  "⚠",      "✓",      "✓"],      highlight: WHITE },
  { data: ["claude-haiku-4-5",  "80%",  "✓",      "⚠",      "✓"],      highlight: WHITE },
  { data: ["nova-pro",          "60%",  "⚠ 67%",  "⚠ 67%",  "✓"],      highlight: RED_LIGHT },
];

const MANTLE_DATA = [
  { data: ["openai.gpt-5.4",      "93%", "Acctg 90%, Micro 100%", "1.92s", "$0.037"], highlight: WHITE },
  { data: ["kimi-k2.5 (Mantle)",  "90%", "—",                     "1.69s", "$0.017"], highlight: WHITE },
  { data: ["qwen3-coder-480b",    "90%", "Even",                  "0.94s", "$0.017"], highlight: WHITE },
  { data: ["qwen3-235b",          "90%", "Ethics 95%",            "0.90s", "$0.009"], highlight: WHITE },
  { data: ["gpt-5.5",             "88%", "—",                     "2.82s", "$0.126"], highlight: WHITE },
  { data: ["deepseek-v3.1",       "88%", "—",                     "0.94s", "$0.002"], highlight: WHITE },
  { data: ["gemma-4-31b",         "87%", "Ethics 95%",            "1.71s", "$0.007"], highlight: WHITE },
  { data: ["gemma-4-26b",         "84%", "—",                     "1.04s", "$0.006"], highlight: WHITE },
  { data: ["xai.grok-4.3",        "42%", "—",                     "3.04s", "$0.494"], highlight: AMBER },
];

const COST_DATA = [
  { data: ["nova-micro",        "$0.0005", "Fastest / cheapest. 66% accuracy. Best fallback."],        highlight: GREY_LIGHT },
  { data: ["nova-lite",         "$0.0008", "Economy tier. AWS-native, fast."],                         highlight: GREY_LIGHT },
  { data: ["deepseek-v3.2",     "$0.002",  "Best value. 86% accuracy + 80% guardrail."],               highlight: GREEN },
  { data: ["qwen3-32b",         "$0.002",  "Strong MCQ, very fast (0.43s P50)."],                      highlight: WHITE },
  { data: ["glm-5",             "$0.007",  "WINNER. 88% + 100% guardrail. Exceptional value."],        highlight: GREEN },
  { data: ["mistral-large-3",   "$0.012",  "Strong all-round. 87% + 90% guardrail."],                  highlight: WHITE },
  { data: ["claude-sonnet-4-6", "$0.049",  "Premium quality + full compliance. Recommended for compliance queries."], highlight: LIGHT_BLUE },
  { data: ["claude-opus-5",     "$0.605",  "Highest accuracy (94%) but costly. No guardrail score."],  highlight: AMBER },
];

const USE_CASE_DATA = [
  { data: ["Compliance check",     "claude-sonnet-4-6",  "100% guardrail + 92% accuracy"],        highlight: LIGHT_BLUE },
  { data: ["General customer svc", "zai/glm-5",          "Best composite, 100% compliance, $0.007/100q"], highlight: GREEN },
  { data: ["High-volume / economy","deepseek-v3.2",       "86% accuracy, $0.002/100q, 80% guardrail"],  highlight: WHITE },
  { data: ["Fallback / degradation","amazon.nova-lite",   "$0.0008/100q, fastest, AWS-native"],     highlight: WHITE },
  { data: ["Complex multi-turn",   "kimi-k2.5",          "90% accuracy + 90% guardrail"],           highlight: WHITE },
  { data: ["Frontier (Mantle)",    "openai.gpt-5.4",     "93% accuracy, $0.037/100q via Mantle"],  highlight: WHITE },
];

// ── Document ────────────────────────────────────────────────────────────────
const doc = new Document({
  numbering: {
    config: [
      { reference: "bullets",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ]
  },
  styles: {
    default: { document: { run: { font: "Arial", size: 20 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, color: DARK_BLUE, font: "Arial" },
        paragraph: { spacing: { before: 360, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, color: BLUE, font: "Arial" },
        paragraph: { spacing: { before: 280, after: 80 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, color: GREY_DARK, font: "Arial" },
        paragraph: { spacing: { before: 200, after: 60 }, outlineLevel: 2 } },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 }
      }
    },
    headers: {
      default: new Header({ children: [new Paragraph({
        tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }],
        border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: BLUE } },
        spacing: { after: 80 },
        children: [
          new TextRun({ text: "Financial Services AI Assistant — Model Evaluation Findings", size: 18, color: GREY_DARK, font: "Arial" }),
          new TextRun({ text: "\tSeptember 2026", size: 18, color: GREY_DARK, font: "Arial" })
        ]
      })] })
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }],
        border: { top: { style: BorderStyle.SINGLE, size: 4, color: BLUE } },
        spacing: { before: 80 },
        children: [
          new TextRun({ text: "AWS CONFIDENTIAL — For internal use only", size: 16, color: GREY_DARK, font: "Arial" }),
          new TextRun({ children: [PageNumber.CURRENT], size: 16, color: GREY_DARK, font: "Arial" })
        ]
      })] })
    },
    children: [

      // ── COVER ───────────────────────────────────────────────────────────
      spacer(2),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 80 },
        children: [new TextRun({ text: "FINANCIAL SERVICES AI ASSISTANT", bold: true, size: 48, color: DARK_BLUE, font: "Arial" })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 80 },
        children: [new TextRun({ text: "Model Evaluation & Architecture Findings", bold: true, size: 36, color: BLUE, font: "Arial" })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 80 },
        children: [new TextRun({ text: "GenAI Professional Certification Exercise — September 2026", size: 24, color: GREY_DARK, font: "Arial", italics: true })]
      }),
      spacer(1),
      divider(),
      spacer(1),
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [2340, 7020],
        rows: [
          new TableRow({ children: [
            cell("Scope", { fill: DARK_BLUE, bold: true, color: WHITE, width: 2340 }),
            cell("60+ Amazon Bedrock foundation models benchmarked on financial services knowledge, latency, cost, and guardrail compliance", { width: 7020 })
          ]}),
          new TableRow({ children: [
            cell("Models Tested", { fill: LIGHT_BLUE, bold: true, width: 2340 }),
            cell("Standard Bedrock Converse API + Mantle SigV4 endpoint (GPT-5, Grok, Gemma 4, Qwen3-235B)", { width: 7020 })
          ]}),
          new TableRow({ children: [
            cell("Dataset", { fill: DARK_BLUE, bold: true, color: WHITE, width: 2340 }),
            cell("MMLU Financial Subsets: Professional Accounting, Business Ethics, Econometrics, Macroeconomics, Microeconomics — 100 questions per model", { width: 7020 })
          ]}),
          new TableRow({ children: [
            cell("Guardrail Tests", { fill: LIGHT_BLUE, bold: true, width: 2340 }),
            cell("10 live behavioral tests: insider trading refusal, structuring/AML refusal, elder fraud refusal, investment disclaimer, IRA/401k helpfulness, KYC explanation", { width: 7020 })
          ]}),
          new TableRow({ children: [
            cell("AWS Region", { fill: DARK_BLUE, bold: true, color: WHITE, width: 2340 }),
            cell("us-east-1 (Note: UAE regions me-central-1/me-south-1 offline — FAB/RAKBANK production runs in eu-west-1)", { width: 7020 })
          ]}),
        ]
      }),
      spacer(2),
      pageBreak(),

      // ── EXECUTIVE SUMMARY ───────────────────────────────────────────────
      h1("Executive Summary"),
      divider(),
      para("This report presents findings from a comprehensive benchmark of 60+ Amazon Bedrock foundation models for deployment as a financial services customer service AI assistant. Models were evaluated across four production-critical dimensions: FSI domain knowledge, latency and throughput, cost per request, and regulatory guardrail compliance."),
      spacer(0.5),
      callout("KEY FINDING: Guardrail compliance reshapes the rankings entirely. Claude Opus 5 scores 94% on financial knowledge but receives 0 on compliance (untested). Nova Pro — despite being AWS-native — passes only 60% of financial compliance tests, making it unsuitable for direct customer-facing FSI use without additional guardrails.", AMBER),
      spacer(0.5),
      h2("Top Recommendations"),
      leaderboardTable(
        USE_CASE_DATA,
        [2800, 2600, 3960],
        ["Use Case", "Recommended Model", "Rationale"]
      ),
      spacer(1),
      para("The surprise finding: Z.AI GLM-5 achieves the highest composite score (0.919) with 88% FSI accuracy, perfect 100% guardrail compliance, 0.61s median latency, and a cost of just $0.007 per 100 questions — making it the best all-round choice for production financial services deployments."),
      pageBreak(),

      // ── PART 1: BENCHMARK ───────────────────────────────────────────────
      h1("Part 1 — Foundation Model Benchmarking"),
      divider(),

      h2("Methodology"),
      para("Models were evaluated using a 4-dimension scoring framework with the following weights, chosen to reflect FSI production priorities:"),
      bullet("Financial Domain Knowledge (40%) — Exact-match accuracy on MMLU financial MCQ (100 questions per model, 5 subjects)"),
      bullet("Guardrail Compliance (40%) — Behavioral tests measuring refusal of illegal requests, disclaimer quality on investment advice, and helpfulness on legitimate queries"),
      bullet("Latency (20%) — Normalised P50 latency score; lower latency = higher score"),
      spacer(0.5),
      para("Composite Score = 0.40 × Accuracy + 0.40 × Guardrail + 0.20 × Latency Score", { bold: true }),
      spacer(0.5),
      callout("Note: Models not tested for guardrails receive a guardrail score of 0 in the composite. This is intentional — in FSI production, an untested model is an unacceptable risk regardless of its knowledge accuracy."),
      spacer(1),

      h2("Dimension 1 — FSI Knowledge Accuracy"),
      para("Models were tested on 5 MMLU financial subjects: Professional Accounting (282 questions), Business Ethics (100), Econometrics (114), Macroeconomics (390), and Microeconomics (238). A random sample of 20 questions per subject was drawn."),
      spacer(0.5),
      para("Subject Difficulty (hardest → easiest across all models):"),
      numbered("Professional Accounting — avg 60% (complex multi-step, revenue recognition, audit sampling)"),
      numbered("Econometrics — avg 63% (quantitative, regression, time series)"),
      numbered("Macroeconomics — avg 77% (fiscal/monetary policy, GDP, inflation)"),
      numbered("Business Ethics — avg 83% (principles-based, less quantitative)"),
      numbered("Microeconomics — avg 88% (easiest — supply/demand, elasticity, market structures)"),
      spacer(1),

      h2("Dimension 2 — Guardrail Compliance"),
      para("10 behavioral tests were executed against the top 10 models by accuracy. Tests covered three compliance categories:"),
      bullet("Illegal Activity Refusal (3 tests): insider trading, AML structuring, elder fraud script"),
      bullet("Investment Advice Disclaimer (3 tests): S&P 500 allocation, panic selling, mortgage vs. invest decision"),
      bullet("Legitimate Helpfulness (4 tests): IRA vs Roth IRA explanation, business account documents, APR vs APY, KYC definition"),
      spacer(0.5),
      leaderboardTable(GUARDRAIL_DATA, [2400, 1200, 2000, 2000, 1760],
        ["Model", "Overall", "Refuses Illegal", "Adds Disclaimer", "Helpful"]),
      spacer(0.5),
      callout("Nova Pro warning: 60% guardrail score means it failed 40% of compliance tests including refusing insider trading advice and adding disclaimers on investment queries. Not recommended for direct customer-facing FSI use without Bedrock Guardrails configured.", RED_LIGHT),
      spacer(1),

      h2("Section A — Guardrail-Tested Models (Production Candidates)"),
      para(`${TESTED.length} models received live financial compliance testing. Green = production-ready (≥80% guardrail); amber = requires additional Bedrock Guardrails configuration before FSI deployment. Sorted by FSI accuracy.`),
      spacer(0.5),
      leaderboardTable(
        COMPOSITE_DATA,
        [560, 3200, 900, 900, 900, 900, 1000],
        ["#", "Model", "Accuracy", "Guardrail", "P50", "P95", "Cost/100q"]
      ),
      spacer(1),
      h2("Section B — All Remaining Models (Guardrail Testing Pending)"),
      para(`${UNTESTED.length} models evaluated for FSI knowledge, latency, and cost. Guardrail compliance not yet tested. Do NOT deploy to customer-facing production without running compliance tests first. Sorted by accuracy descending.`),
      spacer(0.5),
      leaderboardTable(
        UNTESTED.map((m, i) => ({ data: [String(i+1), m.model_id, fmtAcc(m), fmtP50(m), fmtP95(m), fmtCost(m), "Pending"] })),
        [560, 3200, 900, 800, 800, 900, 1200],
        ["#", "Model", "Accuracy", "P50", "P95", "Cost/100q", "Guardrail"]
      ),
      spacer(1),
      ...(ERRORS.length > 0 ? [
        h2("Section C — Models with API Errors"),
        para(`${ERRORS.length} models returned API errors during benchmarking. Root causes noted in the Appendix technical notes.`),
        spacer(0.5),
        leaderboardTable(
          ERRORS.map((m, i) => ({
            data: [String(i+1), m.model_id, `${m.n_success}/${m.n_total} calls ok`, "—", "—", fmtCost(m), "—"],
            highlight: RED_LIGHT
          })),
          [560, 3600, 1300, 700, 700, 900, 760],
          ["#", "Model", "Success Rate", "Accuracy", "Latency", "Cost/100q", "Guardrail"]
        ),
        spacer(1),
      ] : []),

      h2("Dimension 3 — Cost Per Request"),
      para("Cost calculated using Bedrock on-demand pricing (us-east-1, September 2026) applied to actual input/output token counts from benchmark runs."),
      spacer(0.5),
      leaderboardTable(COST_DATA, [2600, 1560, 5200],
        ["Model", "Cost/100q", "Notes"]),
      spacer(0.5),
      para("Value insight: GLM-5 delivers the best accuracy-per-dollar at any guardrail-compliant tier. DeepSeek V3.2 at $0.002/100q is the extreme value option for non-compliance-critical use cases."),
      pageBreak(),

      // ── MANTLE RESULTS ─────────────────────────────────────────────────
      h2("Mantle Endpoint Results (Exclusive Models)"),
      para("Models accessible only via the Bedrock Mantle endpoint (bedrock-mantle.us-east-1.api.aws) were evaluated separately using SigV4 authentication — no separate API key required. 22 Mantle-exclusive models were tested."),
      spacer(0.5),
      leaderboardTable(MANTLE_DATA, [2200, 900, 2500, 1180, 1580],
        ["Model", "Accuracy", "Subject Highlights", "P50", "Cost/100q"]),
      spacer(0.5),
      bullet("GPT-5.4 tops all models at 93% accuracy but at $0.037/100q — comparable to Claude Sonnet"),
      bullet("Qwen3-235B and Qwen3-Coder-480B both reach 90% at $0.009 and $0.017 — strong frontier-class value"),
      bullet("Gemma 4 31B (87%) is a major improvement over Gemma 3 27B (71%), suggesting significant architectural gains"),
      bullet("Grok-4.3 scored only 42% — verbose responses confuse the MCQ letter extractor; true capability may be higher"),
      pageBreak(),

      // ── PART 2 ────────────────────────────────────────────────────────
      h1("Part 2 — Dynamic Model Selection Architecture"),
      divider(),
      para("A model abstraction layer decouples clients from specific model IDs, enabling real-time routing changes without code deployments."),
      spacer(0.5),
      h2("Architecture"),
      bullet("AWS Lambda (python_handler.py) — receives requests, reads AppConfig routing rules, invokes Bedrock Converse API with FSI system prompt, falls back through model chain on throttling/errors"),
      bullet("AWS AppConfig (model_selection_strategy.json) — stores routing rules: use_case_overrides, cost_tier defaults, fallback_chain; updated without Lambda redeployment"),
      bullet("Amazon API Gateway — REST API with /generate (POST, Lambda proxy) and /health (GET, MOCK); CloudWatch logging + X-Ray tracing"),
      spacer(0.5),
      h2("Routing Logic"),
      leaderboardTable([
        { data: ["compliance_check", "claude-sonnet-4-6", "100% guardrail compliance required"] },
        { data: ["complex_query",    "claude-sonnet-4-6", "Multi-turn, nuanced responses needed"] },
        { data: ["customer_service", "zai/glm-5",         "Best composite; 100% guardrail; low cost"] },
        { data: ["product_faq",      "nova-micro",         "Simple lookup; extreme low cost"] },
        { data: ["economy tier",     "nova-micro",         "$0.0005/100q; sub-0.45s latency"] },
        { data: ["standard tier",    "nova-lite",          "$0.0008/100q; good for general queries"] },
        { data: ["premium tier",     "claude-sonnet-4-6",  "Full quality + compliance"] },
      ], [2800, 2600, 3960], ["Use Case / Tier", "Model", "Reason"]),
      spacer(0.5),
      h2("Financial Compliance Guardrail"),
      para("Every Lambda invocation injects the following system prompt regardless of routing:"),
      callout("You are a financial services assistant. Always recommend consulting a qualified financial advisor for personalized advice. Never provide specific investment recommendations. Refuse any request that could facilitate illegal financial activity including insider trading, tax evasion, or money laundering."),
      pageBreak(),

      // ── PART 3 ────────────────────────────────────────────────────────
      h1("Part 3 — Resilient System Design"),
      divider(),
      para("High availability is achieved through a three-layer resilience pattern: circuit breaker (prevents cascade failures), cross-region deployment (geographic redundancy), and graceful degradation (always-on fallback responses)."),
      spacer(0.5),
      h2("Circuit Breaker Pattern (Step Functions)"),
      bullet("State machine: CheckCircuitBreaker → IsCircuitOpen? → TryPrimaryModel → RecordSuccess/Failure → TryFallbackModel → GracefulDegradation"),
      bullet("Circuit opens at 5 consecutive failures within 60 seconds (DynamoDB tracks state atomically)"),
      bullet("Open circuit bypasses primary model and goes directly to fallback — prevents thundering herd on a failing model endpoint"),
      bullet("Express Workflow (not Standard) for sub-second latency overhead"),
      spacer(0.5),
      h2("Cross-Region Deployment"),
      bullet("CloudFormation template deploys identical stack to multiple regions (us-east-1 primary, eu-west-1 secondary for FAB/RAKBANK)"),
      bullet("DynamoDB Global Tables replicate circuit breaker state cross-region"),
      bullet("Route 53 failover routing with health checks on API Gateway endpoints"),
      bullet("RTO < 60 seconds for regional failover"),
      spacer(0.5),
      h2("Graceful Degradation"),
      para("FSI-specific canned responses per use_case ensure customers always receive a useful response, even when all models are unavailable:"),
      bullet("compliance_check: Routes to human compliance officer for manual review"),
      bullet("customer_service: Provides call-centre number and business hours"),
      bullet("complex_query: Commits to 1-business-day relationship manager follow-up"),
      bullet("product_faq: Directs to documentation site and support number"),
      pageBreak(),

      // ── PART 4 ────────────────────────────────────────────────────────
      h1("Part 4 — Model Customization & Lifecycle"),
      divider(),
      para("Fine-tuning a foundation model on proprietary FSI data improves performance on institution-specific product questions, internal terminology, and regulatory nuance not present in public training data."),
      spacer(0.5),
      h2("Fine-Tuning Dataset"),
      bullet("50+ financial Q&A pairs across 8 domains: savings/CDs, mortgages, credit, IRA/401k, KYC/AML, customer service scenarios, digital banking, fees"),
      bullet("Format: JSONL instruction-following (Human/Assistant turns) for causal LM fine-tuning"),
      bullet("Split: 80% train, 10% validation, 10% held-out test; uploaded to S3 via manifest"),
      spacer(0.5),
      h2("Training Approach (SageMaker + LoRA)"),
      bullet("Base model: distilgpt2 (lightweight for demonstration; architecture generalises to any causal LM)"),
      bullet("LoRA config: r=8, alpha=16, dropout=0.1 — trains only 0.1% of parameters"),
      bullet("10-prompt FSI compliance eval suite runs after each epoch: KYC definition, compound interest, AML red flags, etc."),
      bullet("Metrics tracked: train/val loss per epoch, perplexity, sample outputs on eval suite"),
      spacer(0.5),
      h2("Model Lifecycle Management"),
      leaderboardTable([
        { data: ["ModelRegistry",   "DynamoDB",      "Stores model versions with status: candidate → production → archived"] },
        { data: ["ModelTester",     "Live Bedrock",  "15 tests: factual accuracy, guardrail compliance, format, latency P95"] },
        { data: ["ModelDeployer",   "Lambda + AppConfig", "Canary rollout: 10% traffic to new model → CloudWatch error check → 100%"] },
        { data: ["promote_pipeline","Orchestrator",  "Test → PROMOTE/REJECT/REVIEW → deploy or SNS alert + rollback"] },
      ], [2600, 2200, 4560], ["Component", "Backend", "Function"]),
      spacer(0.5),
      h2("Rollback Strategy"),
      bullet("Canary at 10% for 5 minutes before full cutover — Lambda alias routing_config splits traffic"),
      bullet("CloudWatch alarm on error rate > 2% triggers automatic rollback to previous production alias"),
      bullet("DynamoDB model registry maintains full version history; promote_pipeline can restore any archived version"),
      bullet("AppConfig propagates model ID change within 60 seconds via TTL-cached configuration session"),
      pageBreak(),

      // ── APPENDIX ──────────────────────────────────────────────────────
      h1("Appendix — Full Model Index"),
      divider(),
      para("All models evaluated in this exercise, organized by provider and endpoint:"),
      spacer(0.5),
      h3("Standard Bedrock Converse API (us-east-1)"),
      bullet("Anthropic Claude: sonnet-5, opus-5, fable-5-1, fable-5, opus-4-8, opus-4-7, opus-4-6-v1, opus-4-5, sonnet-4-6, sonnet-4-5, haiku-4-5 (via us.anthropic.* cross-region profiles)"),
      bullet("Amazon Nova: nova-pro, nova-lite, nova-micro"),
      bullet("Meta Llama: llama3-70b-instruct, llama3-8b-instruct"),
      bullet("Mistral: mistral-large-3, magistral-small, devstral-2-123b, mixtral-8x7b, ministral-14b, ministral-8b, mistral-large-2402, mistral-small, mistral-7b, ministral-3b"),
      bullet("DeepSeek: v3.2 | Qwen: qwen3-80b, qwen3-32b, qwen3-coder-30b"),
      bullet("Google: gemma-3-27b, gemma-3-12b, gemma-3-4b"),
      bullet("NVIDIA: nemotron-super-120b, nemotron-nano-30b, nemotron-nano-9b"),
      bullet("OpenAI OSS: gpt-oss-120b, gpt-oss-20b | Moonshot: kimi-k2.5, kimi-k2-thinking"),
      bullet("MiniMax: m2.5, m2.1, m2 | Z.AI GLM: glm-5, glm-4.7, glm-4.7-flash"),
      spacer(0.5),
      h3("Bedrock Mantle Endpoint (SigV4, bedrock-mantle.us-east-1.api.aws)"),
      bullet("OpenAI: gpt-5.6-sol, gpt-5.6-luna, gpt-5.6-terra, gpt-5.5, gpt-5.4"),
      bullet("xAI: grok-4.3 | Google: gemma-4-31b, gemma-4-26b, gemma-4-e2b"),
      bullet("Qwen: qwen3-235b, qwen3-coder-480b, qwen3-coder-next"),
      bullet("DeepSeek: v3.1 | Z.AI: glm-4.6 | Moonshot: kimi-k2-thinking, kimi-k2.5"),
      bullet("Anthropic Claude (Mantle path): haiku-4-5, sonnet-5, opus-5, fable-5, opus-4-8, opus-4-7"),
      spacer(0.5),
      h3("Technical Notes"),
      bullet("Claude models (Sonnet 5, Opus 5, Fable 5+) reject temperature parameter — auto-retry without temperature implemented"),
      bullet("Thinking models (kimi-k2-thinking, MiniMax) return thinking blocks before text — parser extracts final text block"),
      bullet("Chain-of-thought models (nemotron-nano-9b) ignore MCQ format instructions — extract last A/B/C/D letter in response"),
      bullet("AI21 Jamba models marked Legacy with access denied — excluded from benchmark"),
      bullet("Grok-4.3 MCQ score (42%) likely underestimates true capability due to verbose response format"),
      spacer(1),
      divider(),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 200 },
        children: [new TextRun({ text: "End of Report — AWS GenAI Professional Certification Exercise", size: 18, color: GREY_DARK, italics: true, font: "Arial" })]
      }),
    ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("Financial_AI_Assistant_Findings.docx", buffer);
  console.log("Created: Financial_AI_Assistant_Findings.docx");
});
