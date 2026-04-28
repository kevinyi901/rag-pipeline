import React, { useState, useEffect, useRef, useCallback, createContext, useContext } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  HelpCircle, Wand2, X, Send, FileText,
  Filter, ChevronDown, ChevronRight, Upload, AlertCircle,
  BookOpen, Info, Shield, ExternalLink, Search,
  Quote, Moon, Sun,
} from "lucide-react";

// ─── Theme ────────────────────────────────────────────────────────────────────
const ThemeCtx = createContext(null);
const useTheme = () => useContext(ThemeCtx);

const LIGHT = {
  bg:           "#ffffff",
  panel:        "#f0f5f9",
  panelSoft:    "#f8fafd",
  ink:          "#041E42",
  inkSoft:      "#4a6a8a",
  line:         "#041E4218",
  lineStrong:   "#041E4235",
  accent:       "#003865",
  accentSoft:   "#00386510",
  highlight:    "#00C1D5",
  highlightSoft:"#00C1D518",
  error:        "#BE5400",
  green:        "#4A6A1D",
  navy:         "#041E42",
  headerBg:     "#041E42",
  headerText:   "#ffffff",
};

const DARK = {
  bg:           "#0a1628",
  panel:        "#0f1e35",
  panelSoft:    "#0c1a2e",
  ink:          "#cce0f5",
  inkSoft:      "#6a90b8",
  line:         "#ffffff10",
  lineStrong:   "#ffffff1e",
  accent:       "#1a5285",
  accentSoft:   "#1a528520",
  highlight:    "#00C1D5",
  highlightSoft:"#00C1D520",
  error:        "#d4622a",
  green:        "#5d8424",
  navy:         "#041E42",
  headerBg:     "#060e1c",
  headerText:   "#d4e8f8",
};

// ─── Derived API URLs ─────────────────────────────────────────────────────────
const API_BASE        = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/+$/, "");
const QUERY_URL       = `${API_BASE}/query`;
const HEALTH_URL      = `${API_BASE}/health`;
const LOGIN_URL       = `${API_BASE}/auth/login`;
const UPLOAD_URL      = `${API_BASE}/auth/upload`;
const FALLBACK_LOGIN_PASS  = import.meta.env.VITE_LOGIN_PASS  || "";
const FALLBACK_UPLOAD_PASS = import.meta.env.VITE_UPLOAD_PASS || "";

// ─── Helpers ─────────────────────────────────────────────────────────────────
async function apiPost(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return { ok: r.ok, status: r.status, data: await r.json().catch(() => ({})) };
}

function buildFilters(state) {
  const f = {};
  if (state.states.length)   f.state  = state.states;
  if (state.counties.length) f.county = state.counties;
  if (state.penalty     !== "all") f.penalty     = state.penalty;
  if (state.obligation  !== "all") f.obligation  = state.obligation;
  if (state.permission  !== "all") f.permission  = state.permission;
  if (state.prohibition !== "all") f.prohibition = state.prohibition;
  const addRange = (key, min, max, lo, hi) => {
    const r = {};
    if (min > lo) r.min = min;
    if (max < hi) r.max = max;
    if (Object.keys(r).length) f[key] = r;
  };
  addRange("fk_grade",    state.fkMin,  state.fkMax,  0,   20);
  addRange("fre",         state.freMin, state.freMax, 0,   100);
  addRange("wc",          state.wcMin,  state.wcMax,  0,   5000);
  addRange("pct_complex", state.pcMin,  state.pcMax,  0,   100);
  return f;
}

function chunksToSources(chunks) {
  return chunks.map((c, i) => ({
    id:          c.id || String(i),
    title:       c.source_name || c.doc_id || `Document ${i + 1}`,
    authors:     c.pi_name || c.author || c.investigator || "",
    year:        c.year   || c.date   || "",
    type:        c.doc_type || "PDF",
    institution: c.institution || c.grantee || c.university || "",
    excerpt:     c.chunk_text  || c.text || "",
    score:       c.rerank_score != null
                   ? Number(c.rerank_score).toFixed(3)
                   : c.score != null ? Number(c.score).toFixed(3) : "",
    page:        c.page != null ? c.page : "",
    state:       c.state  || "",
    county:      c.county || "",
  }));
}

// ─── Prompt library ───────────────────────────────────────────────────────────
const PROMPT_CATEGORIES = [
  {
    id: "grants", label: "Grant Search",
    desc: "Find specific grants, funding decisions, and awardees.",
    prompts: [
      "Have we ever funded research on gene BEST1?",
      "Show all grants related to retinitis pigmentosa.",
      "Which institutions received the most funding?",
    ],
  },
  {
    id: "research", label: "Research",
    desc: "Surface scientific findings and clinical outcomes.",
    prompts: [
      "What progress has been made in gene therapy for LCA?",
      "Summarize findings on RPE65 treatments.",
      "What clinical trials are currently active?",
    ],
  },
  {
    id: "compare", label: "Comparison",
    desc: "Compare projects, years, or investigators.",
    prompts: [
      "Compare Year 2 to Year 1 for this project.",
      "Compare the proposal to the latest progress report.",
      "Show funding trends over the past five years.",
    ],
  },
  {
    id: "audience", label: "Audience",
    desc: "Reshape evidence for a different reader.",
    prompts: [
      "Summarize this for a donor unfamiliar with genetics.",
      "Write a patient-friendly explanation.",
      "Draft a board update on recent breakthroughs.",
    ],
  },
];

// ─── Dual range slider ────────────────────────────────────────────────────────
function DualRangeSlider({ label, mn, mx, lo, hi, step = 1, fmt = v => v, onChange }) {
  const C = useTheme();
  const containerRef = useRef(null);
  const dragging     = useRef(null);

  // Keep a live ref so the effect closure never goes stale
  const live = useRef({ mn, mx, onChange, lo, hi, step });
  useEffect(() => { live.current = { mn, mx, onChange, lo, hi, step }; });

  // Convert clientX to a snapped value
  const valueFromX = (clientX) => {
    if (!containerRef.current) return live.current.mn;
    const { lo, hi, step } = live.current;
    const rect  = containerRef.current.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    const raw   = lo + ratio * (hi - lo);
    return Math.round(raw / step) * step;
  };

  useEffect(() => {
    const onMove = (e) => {
      if (!dragging.current) return;
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      const v = valueFromX(clientX);
      const { mn, mx, onChange } = live.current;
      if (dragging.current === "min") onChange(Math.min(v, mx), mx);
      else                            onChange(mn, Math.max(v, mn));
    };
    const onUp = () => { dragging.current = null; };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup",   onUp);
    document.addEventListener("touchmove", onMove, { passive: true });
    document.addEventListener("touchend",  onUp);
    return () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup",   onUp);
      document.removeEventListener("touchmove", onMove);
      document.removeEventListener("touchend",  onUp);
    };
  }, []); // empty — live ref keeps callbacks current

  const startDrag = (handle) => (e) => {
    e.preventDefault();
    e.stopPropagation();
    dragging.current = handle;
  };

  const onTrackDown = (e) => {
    const v = valueFromX(e.clientX);
    const { mn, mx, onChange } = live.current;
    const handle = Math.abs(v - mn) <= Math.abs(v - mx) ? "min" : "max";
    dragging.current = handle;
    if (handle === "min") onChange(Math.min(v, mx), mx);
    else                  onChange(mn, Math.max(v, mn));
  };

  const pct     = v => ((v - lo) / (hi - lo)) * 100;
  const leftPct  = pct(mn);
  const rightPct = 100 - pct(mx);

  return (
    <div>
      <div className="flex justify-between text-[10px] mb-2" style={{ color: C.inkSoft }}>
        <span>{label}</span>
        <span style={{ fontFamily: "'JetBrains Mono', monospace", color: C.highlight }}>
          {fmt(mn)} – {fmt(mx)}
        </span>
      </div>
      <div
        ref={containerRef}
        className="relative select-none"
        style={{ height: "20px", cursor: "pointer" }}
        onMouseDown={onTrackDown}
      >
        {/* Background track */}
        <div style={{
          position: "absolute", top: "7px", left: 0, right: 0,
          height: "6px", borderRadius: "9999px",
          backgroundColor: C.lineStrong, pointerEvents: "none",
        }} />
        {/* Filled range */}
        <div style={{
          position: "absolute", top: "7px",
          left: `${leftPct}%`, right: `${rightPct}%`,
          height: "6px", borderRadius: "9999px",
          backgroundColor: C.highlight, pointerEvents: "none",
        }} />
        {/* Min thumb */}
        <div
          style={{
            position: "absolute", top: "2px",
            left: `calc(${leftPct}% - 8px)`,
            width: "16px", height: "16px",
            borderRadius: "50%",
            backgroundColor: C.highlight,
            border: "2px solid #fff",
            boxShadow: "0 1px 4px rgba(4,30,66,0.4)",
            cursor: "grab",
            zIndex: leftPct > 90 ? 3 : 1,
          }}
          onMouseDown={startDrag("min")}
          onTouchStart={startDrag("min")}
        />
        {/* Max thumb */}
        <div
          style={{
            position: "absolute", top: "2px",
            left: `calc(${pct(mx)}% - 8px)`,
            width: "16px", height: "16px",
            borderRadius: "50%",
            backgroundColor: C.accent,
            border: "2px solid #fff",
            boxShadow: "0 1px 4px rgba(4,30,66,0.4)",
            cursor: "grab",
            zIndex: 2,
          }}
          onMouseDown={startDrag("max")}
          onTouchStart={startDrag("max")}
        />
      </div>
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────
export default function SightReader() {
  useEffect(() => {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href =
      "https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Roboto+Condensed:wght@400;700&family=JetBrains+Mono:wght@400;500&display=swap";
    document.head.appendChild(link);
  }, []);

  const [darkMode, setDarkMode] = useState(false);
  const C = darkMode ? DARK : LIGHT;

  // ── Auth ──
  const [authed, setAuthed]       = useState(false);
  const [pwInput, setPwInput]     = useState("");
  const [pwError, setPwError]     = useState("");
  const [pwLoading, setPwLoading] = useState(false);

  // ── Upload panel ──
  const [activePanel, setActivePanel]     = useState(null);
  const [uploadAuthed, setUploadAuthed]   = useState(false);
  const [uploadPw, setUploadPw]           = useState("");
  const [uploadPwError, setUploadPwError] = useState("");
  const [uploadLoading, setUploadLoading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [dragOver, setDragOver]           = useState(false);

  // ── Layout ──
  const [menuOpen, setMenuOpen]       = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [activeTab, setActiveTab]     = useState("filters");
  const [expandedSource, setExpandedSource] = useState(null);


  // ── Filter state ──
  const [filterState, setFilterState] = useState({
    states: [], counties: [],
    penalty: "all", obligation: "all", permission: "all", prohibition: "all",
    fkMin: 0,  fkMax: 20,
    freMin: 0, freMax: 100,
    wcMin: 0,  wcMax: 5000,
    pcMin: 0,  pcMax: 100,
  });
  const [stateInput,  setStateInput]  = useState("");
  const [countyInput, setCountyInput] = useState("");

  const addTag = (field, val) => {
    const v = val.trim().toLowerCase();
    if (!v) return;
    setFilterState(s => ({ ...s, [field]: [...new Set([...s[field], v])] }));
  };
  const removeTag = (field, val) =>
    setFilterState(s => ({ ...s, [field]: s[field].filter(x => x !== val) }));

  const resetFilters = () => {
    setFilterState({
      states: [], counties: [],
      penalty: "all", obligation: "all", permission: "all", prohibition: "all",
      fkMin: 0, fkMax: 20, freMin: 0, freMax: 100, wcMin: 0, wcMax: 5000, pcMin: 0, pcMax: 100,
    });
    setStateInput(""); setCountyInput("");
  };

  // ── Session ──
  const [sessionId] = useState(() => crypto.randomUUID());

  // ── Chat ──
  const [messages, setMessages]       = useState([]);
  const [input, setInput]             = useState("");
  const [thinking, setThinking]       = useState(false);
  const [promptOpen, setPromptOpen]   = useState(false);
  const [promptCat, setPromptCat]     = useState("grants");
  const [promptSearch, setPromptSearch] = useState("");

  const inputRef   = useRef(null);
  const chatEndRef = useRef(null);
  const menuRef    = useRef(null);
  const menuBtnRef = useRef(null);

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, thinking]);

  useEffect(() => {
    if (!menuOpen) return;
    const onClick = e => {
      if (menuRef.current && !menuRef.current.contains(e.target) &&
          menuBtnRef.current && !menuBtnRef.current.contains(e.target))
        setMenuOpen(false);
    };
    const onKey = e => e.key === "Escape" && setMenuOpen(false);
    document.addEventListener("mousedown", onClick);
    document.addEventListener("keydown", onKey);
    return () => { document.removeEventListener("mousedown", onClick); document.removeEventListener("keydown", onKey); };
  }, [menuOpen]);

  // ── Auth ──
  const tryLogin = async () => {
    const pw = pwInput.trim();
    if (!pw) return;
    setPwLoading(true); setPwError("");
    try {
      if (FALLBACK_LOGIN_PASS) {
        if (pw === FALLBACK_LOGIN_PASS) { setAuthed(true); return; }
        setPwError("Incorrect password."); return;
      }
      const { ok, status, data } = await apiPost(LOGIN_URL, { password: pw });
      if (ok && data.ok) { setAuthed(true); return; }
      if (status === 401) { setPwError("Incorrect password."); return; }
      setPwError("Cannot reach server. Is the API running?");
    } catch {
      setPwError("Cannot reach server. Is the API running?");
    } finally { setPwLoading(false); }
  };

  const tryUploadAuth = async () => {
    const pw = uploadPw.trim();
    if (!pw) return;
    setUploadLoading(true); setUploadPwError("");
    try {
      if (FALLBACK_UPLOAD_PASS) {
        if (pw === FALLBACK_UPLOAD_PASS) { setUploadAuthed(true); return; }
        setUploadPwError("Incorrect upload password."); return;
      }
      const { ok, status, data } = await apiPost(UPLOAD_URL, { password: pw });
      if (ok && data.ok) { setUploadAuthed(true); return; }
      if (status === 401) { setUploadPwError("Incorrect upload password."); return; }
      setUploadPwError("Cannot reach server.");
    } catch {
      setUploadPwError("Cannot reach server.");
    } finally { setUploadLoading(false); }
  };

  // ── Query ──
  const send = useCallback(async () => {
    const q = input.trim();
    if (!q || thinking) return;
    const filters = buildFilters(filterState);
    setMessages(m => [...m, { role: "user", text: q, id: Date.now() }]);
    setInput("");
    setThinking(true);
    setActiveTab("sources");
    try {
      const r = await fetch(QUERY_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, filters, mode: "hybrid", session_id: sessionId }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.error || `API ${r.status}`);
      setMessages(m => [...m, {
        role: "ai", id: Date.now() + 1,
        text: data.response || "No response received.",
        sources: chunksToSources(data.chunks || []),
      }]);
    } catch (err) {
      setMessages(m => [...m, { role: "ai", id: Date.now() + 1, text: `Error: ${err.message}`, sources: [] }]);
    } finally { setThinking(false); }
  }, [input, thinking, filterState]);

  const insertPrompt = text => { setInput(text); setPromptOpen(false); setTimeout(() => inputRef.current?.focus(), 50); };

  const handleDrop = e => {
    e.preventDefault(); setDragOver(false);
    const files = Array.from(e.dataTransfer.files).map(f => ({ name: f.name, size: (f.size / 1024).toFixed(1) + " KB" }));
    setUploadedFiles(p => [...p, ...files]);
  };

  const latestSources = [...messages].reverse().find(m => m.role === "ai")?.sources || [];

  // ══════════════════════════════════════════════════════════════════════════════
  // LOGIN SCREEN
  // ══════════════════════════════════════════════════════════════════════════════
  if (!authed) {
    return (
      <ThemeCtx.Provider value={C}>
        <div className="min-h-screen w-full flex items-center justify-center relative overflow-hidden"
          style={{ fontFamily: "'Roboto', sans-serif", backgroundColor: C.navy }}>
          <div className="absolute bottom-0 right-0 w-full h-48 pointer-events-none overflow-hidden">
            <div className="absolute bottom-[-60px] right-[-40px] w-[140%] h-40"
              style={{ backgroundColor: C.highlight, transform: "rotate(-6deg) translateY(20px)" }} />
          </div>
          <div className="relative z-10 w-full max-w-sm px-8">
            <div className="mb-8 text-center">
              <div className="inline-flex items-center gap-1 mb-1">
                <span className="text-xs tracking-[0.35em] uppercase font-bold" style={{ color: C.highlight }}>
                  FOUNDATION
                </span>
              </div>
              <h1 className="leading-none"
                style={{ fontFamily: "'Roboto Condensed', sans-serif", fontWeight: 700,
                  fontSize: "52px", color: "#ffffff", letterSpacing: "-0.01em", lineHeight: 1 }}>
                FIGHTING<br />BLINDNESS
              </h1>
              <div className="mt-4 h-px w-16 mx-auto" style={{ backgroundColor: C.highlight }} />
              <p className="mt-4 text-sm leading-relaxed" style={{ color: "#a0b8d0" }}>
                Research intelligence portal — sign in to continue.
              </p>
            </div>
            <div className="space-y-4">
              <input
                type="password" value={pwInput}
                onChange={e => { setPwInput(e.target.value); setPwError(""); }}
                onKeyDown={e => e.key === "Enter" && tryLogin()}
                placeholder="Password"
                className="w-full px-4 py-3 text-sm outline-none rounded border transition"
                style={{ backgroundColor: "#ffffff18", borderColor: "#ffffff30", color: "#ffffff",
                  fontFamily: "'JetBrains Mono', monospace" }}
                onFocus={e => (e.target.style.borderColor = C.highlight)}
                onBlur={e => (e.target.style.borderColor = "#ffffff30")}
              />
              {pwError && (
                <div className="flex items-center gap-2 text-xs" style={{ color: "#ffaa80" }}>
                  <AlertCircle size={12} />{pwError}
                </div>
              )}
              <button onClick={tryLogin} disabled={pwLoading || !pwInput.trim()}
                className="w-full py-3 text-sm font-bold tracking-[0.2em] uppercase rounded transition disabled:opacity-40"
                style={{ backgroundColor: C.accent, color: "#ffffff" }}
                onMouseEnter={e => (e.currentTarget.style.backgroundColor = C.highlight)}
                onMouseLeave={e => (e.currentTarget.style.backgroundColor = C.accent)}>
                {pwLoading ? "Checking…" : "Sign In →"}
              </button>
            </div>
          </div>
        </div>
      </ThemeCtx.Provider>
    );
  }

  // ══════════════════════════════════════════════════════════════════════════════
  // MAIN APP
  // ══════════════════════════════════════════════════════════════════════════════
  return (
    <ThemeCtx.Provider value={C}>
      <div className="h-screen w-full flex flex-col overflow-hidden"
        style={{ fontFamily: "'Roboto', sans-serif", backgroundColor: C.bg, color: C.ink }}>

        {/* ── Header ── */}
        <header className="grid grid-cols-2 items-center px-5 py-3 border-b shrink-0"
          style={{ borderColor: C.line, backgroundColor: C.headerBg, color: C.headerText }}>
{/*           <div className="flex items-center gap-2 hidden"> */}
{/*             <button onClick={() => setSidebarOpen(v => !v)} */}
{/*               className="p-2 rounded transition hidden md:flex items-center gap-1 text-xs tracking-widest uppercase hover:opacity-80" */}
{/*               style={{ color: C.highlight }}> */}
{/*               <Filter size={14} /> */}
{/*             </button> */}
{/*           </div> */}

          <div className="flex items-center justify-start gap-3">
            <span style={{ fontFamily: "'Roboto Condensed', sans-serif", fontWeight: 700,
              fontSize: "20px", letterSpacing: "0.02em", color: C.headerText }}>
              FOUNDATION FIGHTING BLINDNESS
            </span>
            <span className="text-xs px-2 py-0.5 rounded font-medium hidden sm:inline"
              style={{ backgroundColor: C.highlight, color: C.navy }}>
              Research Portal
            </span>
          </div>

          <div className="flex items-center justify-end gap-2">
            {/* Dark mode toggle */}
            <button
              onClick={() => setDarkMode(v => !v)}
              className="p-2 rounded transition hover:opacity-80"
              style={{ color: C.highlight }}
              title={darkMode ? "Switch to light mode" : "Switch to dark mode"}>
              {darkMode ? <Sun size={17} /> : <Moon size={17} />}
            </button>
            <button ref={menuBtnRef} onClick={() => setMenuOpen(v => !v)}
              className="p-2 rounded transition hover:opacity-80" style={{ color: C.highlight }}>
              <HelpCircle size={17} />
            </button>
          </div>
        </header>

        {/* ── Help menu ── */}
        {menuOpen && (
          <div ref={menuRef}
            className="fixed right-4 top-[57px] w-72 z-[200] border shadow-xl overflow-hidden rounded"
            style={{ backgroundColor: C.bg, borderColor: C.lineStrong,
              boxShadow: "0 12px 40px rgba(4, 30, 66, 0.18)" }}>
            <div className="px-4 py-2.5 border-b" style={{ borderColor: C.line, backgroundColor: C.panel }}>
              <span className="text-[10px] tracking-[0.3em] uppercase font-bold" style={{ color: C.inkSoft }}>
                Help & More
              </span>
            </div>
            {[
              { key: "how",    icon: Info,        title: "How it works",    desc: "The RAG pipeline, briefly" },
              { key: "limits", icon: AlertCircle, title: "Limitations",     desc: "What this tool will not do" },
              { key: "upload", icon: Upload,      title: "Add to Database", desc: "Protected · password required", top: true },
            ].map(item => (
              <button key={item.key}
                onClick={() => { setActivePanel(item.key); setMenuOpen(false); }}
                className="w-full flex items-center gap-3 px-4 py-3 transition text-left text-sm"
                style={item.top ? { borderTop: `1px solid ${C.line}` } : {}}
                onMouseEnter={e => (e.currentTarget.style.backgroundColor = C.highlightSoft)}
                onMouseLeave={e => (e.currentTarget.style.backgroundColor = "transparent")}>
                <item.icon size={16} style={{ color: C.highlight }} />
                <div>
                  <div className="font-medium" style={{ color: C.ink }}>{item.title}</div>
                  <div className="text-xs" style={{ color: C.inkSoft }}>{item.desc}</div>
                </div>
              </button>
            ))}
          </div>
        )}

        <div className="flex-1 flex overflow-hidden">
          {/* ── Sidebar ── */}
          {sidebarOpen && (
            <aside className="w-80 border-r flex flex-col overflow-hidden shrink-0"
              style={{ borderColor: C.line, backgroundColor: C.panel }}>
              <div className="flex border-b shrink-0" style={{ borderColor: C.line }}>
                {["filters", "sources"].map(tab => (
                  <button key={tab} onClick={() => setActiveTab(tab)}
                    className="flex-1 py-3 text-[10px] tracking-[0.25em] uppercase font-bold transition"
                    style={activeTab === tab
                      ? { backgroundColor: C.highlightSoft, color: C.highlight, borderBottom: `2px solid ${C.highlight}` }
                      : { color: C.inkSoft }}>
                    {tab}
                    {tab === "sources" && latestSources.length > 0 && (
                      <span className="ml-2 inline-block px-1.5 py-0.5 text-[9px] rounded-full font-bold"
                        style={{ backgroundColor: C.highlight, color: C.navy }}>
                        {latestSources.length}
                      </span>
                    )}
                  </button>
                ))}
              </div>
              <div className="flex-1 overflow-y-auto">
                {activeTab === "filters" ? (
                  <FiltersPanel
                    filterState={filterState} setFilterState={setFilterState}
                    stateInput={stateInput} setStateInput={setStateInput}
                    countyInput={countyInput} setCountyInput={setCountyInput}
                    addTag={addTag} removeTag={removeTag}
                    onReset={resetFilters}
                  />
                ) : (
                  <SourcesList
                    sources={latestSources}
                    expandedSource={expandedSource}
                    setExpandedSource={setExpandedSource}
                  />
                )}
              </div>
            </aside>
          )}

          {/* ── Chat ── */}
          <main className="flex-1 flex flex-col overflow-hidden">
            <div className="flex-1 overflow-y-auto">
              {messages.length === 0
                ? <EmptyState />
                : (
                  <div className="max-w-3xl mx-auto px-6 py-8 space-y-8">
                    {messages.map(m =>
                      m.role === "user" ? (
                        <div key={m.id} className="flex justify-end">
                          <div className="max-w-[85%] px-5 py-3 rounded-2xl rounded-tr-sm text-sm leading-relaxed"
                            style={{ backgroundColor: C.accent, color: "#ffffff" }}>
                            {m.text}
                          </div>
                        </div>
                      ) : (
                        <AIMessage key={m.id} message={m} />
                      )
                    )}
                    {thinking && <ThinkingIndicator />}
                    <div ref={chatEndRef} />
                  </div>
                )}
            </div>

            {promptOpen && (
              <PromptMenu
                categories={PROMPT_CATEGORIES}
                activeCategory={promptCat}
                setActiveCategory={setPromptCat}
                search={promptSearch}
                setSearch={setPromptSearch}
                onInsert={insertPrompt}
                onClose={() => setPromptOpen(false)}
              />
            )}

            {/* Input bar */}
            <div className="border-t px-6 py-4 shrink-0"
              style={{ borderColor: C.line, backgroundColor: C.bg }}>
              <div className="max-w-3xl mx-auto">
                <div className="flex items-end gap-3 border rounded-xl pl-3 pr-3 py-2 min-h-[48px]"
                  style={{ backgroundColor: C.panel, borderColor: C.lineStrong }}>
                  <button onClick={() => setPromptOpen(v => !v)}
                    className="shrink-0 p-2 rounded-lg transition self-end mb-0.5"
                    style={{ backgroundColor: promptOpen ? C.highlight : "transparent",
                      color: promptOpen ? C.navy : C.inkSoft }}
                    onMouseEnter={e => { if (!promptOpen) { e.currentTarget.style.backgroundColor = C.highlightSoft; e.currentTarget.style.color = C.highlight; } }}
                    onMouseLeave={e => { if (!promptOpen) { e.currentTarget.style.backgroundColor = "transparent"; e.currentTarget.style.color = C.inkSoft; } }}
                    title="Prompt library">
                    <Wand2 size={17} />
                  </button>
                  <textarea
                    ref={inputRef}
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
                    rows={1}
                    placeholder="Ask a question about the research corpus…"
                    className="flex-1 bg-transparent outline-none resize-none text-sm leading-relaxed max-h-32 py-1"
                    style={{ color: C.ink }}
                  />
                  <button onClick={send}
                    disabled={!input.trim() || thinking}
                    className="shrink-0 p-2.5 rounded-lg transition disabled:opacity-30 self-end"
                    style={{ backgroundColor: C.accent, color: "#ffffff" }}
                    onMouseEnter={e => { if (input.trim() && !thinking) e.currentTarget.style.backgroundColor = C.highlight; }}
                    onMouseLeave={e => (e.currentTarget.style.backgroundColor = C.accent)}>
                    <Send size={16} />
                  </button>
                </div>
                <p className="text-[10px] mt-2 text-center" style={{ color: C.inkSoft }}>
                  Filters applied: {Object.keys(buildFilters(filterState)).length > 0
                    ? Object.keys(buildFilters(filterState)).join(", ")
                    : "none"}
                </p>
              </div>
            </div>
          </main>
        </div>

        {/* ── Modals ── */}
        {activePanel && (
          <Modal onClose={() => {
            setActivePanel(null); setUploadAuthed(false);
            setUploadPw(""); setUploadPwError("");
          }}>
            {activePanel === "how"    && <HowItWorks />}
            {activePanel === "limits" && <Limitations />}
            {activePanel === "upload" && (
              <UploadPanel
                authed={uploadAuthed} pw={uploadPw} setPw={setUploadPw}
                error={uploadPwError} onAuth={tryUploadAuth} loading={uploadLoading}
                files={uploadedFiles} onDrop={handleDrop}
                dragOver={dragOver} setDragOver={setDragOver}
              />
            )}
          </Modal>
        )}
      </div>
    </ThemeCtx.Provider>
  );
}

// ─── Filters panel ────────────────────────────────────────────────────────────
function FiltersPanel({
  filterState, setFilterState,
  stateInput, setStateInput,
  countyInput, setCountyInput,
  addTag, removeTag, onReset,
}) {
  const C = useTheme();

  const Binary = ({ label, field }) => {
    const styles = {
      all: { backgroundColor: C.navy,  color: "#fff", borderColor: C.navy  },
      Y:   { backgroundColor: C.green, color: "#fff", borderColor: C.green },
      N:   { backgroundColor: C.error, color: "#fff", borderColor: C.error },
    };
    return (
      <div className="flex items-center justify-between">
        <span className="text-xs" style={{ color: C.ink }}>{label}</span>
        <div className="flex gap-1">
          {["all", "Y", "N"].map(v => (
            <button key={v}
              onClick={() => setFilterState(s => ({ ...s, [field]: v }))}
              className="px-2.5 py-0.5 text-[10px] rounded border font-bold transition"
              style={filterState[field] === v
                ? styles[v]
                : { borderColor: C.lineStrong, color: C.inkSoft, backgroundColor: "transparent" }}>
              {v === "all" ? "Any" : v}
            </button>
          ))}
        </div>
      </div>
    );
  };

  const TagInput = ({ label, field, inputVal, setInputVal }) => (
    <div>
      <label className="text-[10px] tracking-[0.2em] uppercase font-bold block mb-2"
        style={{ color: C.inkSoft }}>{label}</label>
      <div className="flex flex-wrap gap-1 mb-2">
        {filterState[field].map(t => (
          <span key={t} className="flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium"
            style={{ backgroundColor: C.highlightSoft, color: C.ink, border: `1px solid ${C.highlight}` }}>
            {t}
            <button onClick={() => removeTag(field, t)} className="hover:opacity-70 ml-0.5">
              <X size={10} />
            </button>
          </span>
        ))}
      </div>
      <input
        value={inputVal}
        onChange={e => setInputVal(e.target.value)}
        onKeyDown={e => {
          if (e.key === "Enter" || e.key === ",") {
            e.preventDefault(); addTag(field, inputVal); setInputVal("");
          }
        }}
        placeholder="Type + Enter to add"
        className="w-full px-3 py-2 text-xs border rounded outline-none"
        style={{ borderColor: C.lineStrong, color: C.ink, backgroundColor: C.bg }}
        onFocus={e => (e.target.style.borderColor = C.highlight)}
        onBlur={e => (e.target.style.borderColor = C.lineStrong)}
      />
    </div>
  );

  return (
    <div className="p-5 space-y-5">
      <TagInput label="State"  field="states"   inputVal={stateInput}  setInputVal={setStateInput} />
      <TagInput label="County" field="counties" inputVal={countyInput} setInputVal={setCountyInput} />

      <div className="rounded-lg p-3" style={{ backgroundColor: C.panelSoft, border: `1px solid ${C.lineStrong}` }}>
        <label className="text-[10px] tracking-[0.2em] uppercase font-bold block mb-3"
          style={{ color: C.accent }}>Document Flags</label>
        <div className="space-y-2.5">
          <Binary label="Penalty"     field="penalty" />
          <Binary label="Obligation"  field="obligation" />
          <Binary label="Permission"  field="permission" />
          <Binary label="Prohibition" field="prohibition" />
        </div>
      </div>

      <div className="rounded-lg p-3" style={{ backgroundColor: C.panelSoft, border: `1px solid ${C.lineStrong}` }}>
        <label className="text-[10px] tracking-[0.2em] uppercase font-bold block mb-3"
          style={{ color: C.accent }}>Readability Metrics</label>
        <div className="space-y-5">
          <DualRangeSlider label="FK Grade" mn={filterState.fkMin} mx={filterState.fkMax}
            lo={0} hi={20} step={0.5} fmt={v => v.toFixed(1)}
            onChange={(mn, mx) => setFilterState(s => ({ ...s, fkMin: mn, fkMax: mx }))} />
          <DualRangeSlider label="Flesch Reading Ease" mn={filterState.freMin} mx={filterState.freMax}
            lo={0} hi={100}
            onChange={(mn, mx) => setFilterState(s => ({ ...s, freMin: mn, freMax: mx }))} />
          <DualRangeSlider label="Word Count" mn={filterState.wcMin} mx={filterState.wcMax}
            lo={0} hi={5000} step={50}
            onChange={(mn, mx) => setFilterState(s => ({ ...s, wcMin: mn, wcMax: mx }))} />
          <DualRangeSlider label="% Complex Words" mn={filterState.pcMin} mx={filterState.pcMax}
            lo={0} hi={100}
            onChange={(mn, mx) => setFilterState(s => ({ ...s, pcMin: mn, pcMax: mx }))} />
        </div>
      </div>

      <div className="pt-2">
        <button onClick={onReset}
          className="w-full text-[10px] tracking-[0.25em] uppercase py-2 border rounded transition font-bold"
          style={{ borderColor: C.lineStrong, color: C.inkSoft }}
          onMouseEnter={e => { e.currentTarget.style.borderColor = C.error; e.currentTarget.style.color = C.error; }}
          onMouseLeave={e => { e.currentTarget.style.borderColor = C.lineStrong; e.currentTarget.style.color = C.inkSoft; }}>
          Reset all filters
        </button>
      </div>
    </div>
  );
}

// ─── Sources list in sidebar ──────────────────────────────────────────────────
function SourcesList({ sources, expandedSource, setExpandedSource }) {
  const C = useTheme();
  if (sources.length === 0) {
    return (
      <div className="text-center text-sm py-12 px-4" style={{ color: C.inkSoft }}>
        <BookOpen size={28} className="mx-auto mb-3 opacity-40" />
        Retrieved sources will appear here after you ask a question.
      </div>
    );
  }
  return (
    <div className="p-4 space-y-3">
      <p className="text-[10px] tracking-[0.25em] uppercase font-bold mb-2" style={{ color: C.inkSoft }}>
        Retrieved · {sources.length} documents
      </p>
      {sources.map((s, i) => {
        const key  = s.id || i;
        const open = expandedSource === key;
        return (
          <div key={key}
            onClick={() => setExpandedSource(open ? null : key)}
            className="p-3 cursor-pointer border rounded transition"
            style={{ backgroundColor: C.bg, borderColor: open ? C.highlight : C.line }}>
            <div className="flex items-start gap-2 mb-1">
              <span className="text-[10px] mt-0.5 px-1.5 py-0.5 rounded shrink-0 font-bold"
                style={{ backgroundColor: C.accent, color: "#fff", fontFamily: "'JetBrains Mono', monospace" }}>
                {String(i + 1).padStart(2, "0")}
              </span>
              <h4 className="text-sm font-medium leading-snug flex-1" style={{ color: C.ink }}>{s.title}</h4>
            </div>
            <div className="text-xs flex flex-wrap gap-1.5 pl-7" style={{ color: C.inkSoft }}>
              {s.page && <span>p.{s.page}</span>}
              {s.state && <span className="px-1 py-0.5 rounded text-[10px]"
                style={{ backgroundColor: C.highlightSoft, color: C.highlight }}>{s.state}</span>}
            </div>
            {open && s.excerpt && (
              <div className="mt-2 pt-2 border-t pl-7" style={{ borderColor: C.line }}>
                <p className="text-xs leading-relaxed p-2 border-l-2 italic"
                  style={{ backgroundColor: C.highlightSoft, borderColor: C.highlight, color: C.ink }}>
                  "{s.excerpt.slice(0, 300)}{s.excerpt.length > 300 ? "…" : ""}"
                </p>
                <div className="flex justify-between mt-1 text-[10px]" style={{ color: C.inkSoft }}>
                  {s.institution && <span>{s.institution}</span>}
                  {s.score && <span style={{ fontFamily: "'JetBrains Mono', monospace" }}>score {s.score}</span>}
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ─── Empty state ──────────────────────────────────────────────────────────────
function EmptyState() {
  const C = useTheme();
  return (
    <div className="h-full flex flex-col items-center justify-center px-6 max-w-2xl mx-auto text-center">
      <div className="w-16 h-1 rounded mb-8" style={{ backgroundColor: C.highlight }} />
      <h2 className="text-4xl md:text-5xl mb-4 leading-tight font-bold"
        style={{ fontFamily: "'Roboto Condensed', sans-serif", color: C.green }}>
        What would you<br />like to know?
      </h2>
      <p className="max-w-md leading-relaxed text-sm" style={{ color: C.inkSoft }}>
        Ask a question about the research corpus.<br />
        Every answer is grounded in retrieved documents.
      </p>
      <div className="w-16 h-1 rounded mt-8" style={{ backgroundColor: C.highlight }} />
    </div>
  );
}

// ─── AI message ───────────────────────────────────────────────────────────────
function AIMessage({ message }) {
  const C = useTheme();
  const [showSources, setShowSources] = useState(false);
  return (
    <div className="flex gap-4">
      <div className="shrink-0 mt-1 w-7 h-7 rounded flex items-center justify-center"
        style={{ backgroundColor: C.navy }}>
        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: C.highlight }} />
      </div>
      <div className="flex-1 min-w-0">
        <div className="text-[10px] tracking-[0.25em] uppercase font-bold mb-2" style={{ color: C.highlight }}>
          Response
        </div>
        <div className="text-sm leading-relaxed mb-5 prose prose-sm max-w-none prose-headings:font-bold prose-h1:text-2xl prose-h2:text-xl prose-h3:text-lg" style={{ color: C.ink }}>
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.text}</ReactMarkdown>
        </div>
        {message.sources && message.sources.length > 0 && (
          <div className="border-t pt-4" style={{ borderColor: C.line }}>
            <button onClick={() => setShowSources(v => !v)}
              className="flex items-center gap-2 text-[10px] tracking-[0.2em] uppercase font-bold transition mb-3"
              style={{ color: C.inkSoft }}>
              {showSources ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
              <Quote size={12} />
              Sources · {message.sources.length} documents cited
            </button>
            {showSources && (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {message.sources.map((s, i) => <SourceCard key={s.id || i} source={s} index={i + 1} />)}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Source card (inline in chat) ─────────────────────────────────────────────
function SourceCard({ source, index }) {
  const C = useTheme();
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="p-3 border rounded transition"
      style={{ backgroundColor: C.panel, borderColor: expanded ? C.highlight : C.line }}
      onMouseEnter={e => { if (!expanded) e.currentTarget.style.borderColor = C.highlight; }}
      onMouseLeave={e => { if (!expanded) e.currentTarget.style.borderColor = C.line; }}>
      <div className="flex items-start gap-2 mb-1">
        <span className="text-[10px] px-1.5 py-0.5 rounded shrink-0 mt-0.5 font-bold"
          style={{ backgroundColor: C.accent, color: "#fff", fontFamily: "'JetBrains Mono', monospace" }}>
          [{index}]
        </span>
        <a href="#" onClick={e => { e.preventDefault(); setExpanded(v => !v); }}
          className="text-sm font-medium leading-snug flex-1 hover:underline" style={{ color: C.ink }}>
          {source.title}
          <ExternalLink size={10} className="inline ml-1 opacity-40" />
        </a>
      </div>
{/*        TODO: need to display PI name correctly here */}
      <div className="text-xs flex flex-wrap gap-1.5 pl-7" style={{ color: C.inkSoft }}>
        {source.authors && <span>{source.authors}</span>}
        {source.year && <span style={{ fontFamily: "'JetBrains Mono', monospace" }}>{source.year}</span>}
        {source.type && <span>{source.type}</span>}
        {source.page && <span>p.{source.page}</span>}
      </div>
      {expanded && source.excerpt && (
        <div className="mt-3 pl-7">
          <p className="text-xs italic leading-relaxed p-2.5 border-l-2"
            style={{ backgroundColor: C.highlightSoft, borderColor: C.highlight, color: C.ink }}>
            "{source.excerpt.slice(0, 400)}{source.excerpt.length > 400 ? "…" : ""}"
          </p>
          <div className="text-[10px] mt-1.5 flex justify-between" style={{ color: C.inkSoft }}>
            {source.institution && <span>{source.institution}</span>}
            {source.score && <span style={{ fontFamily: "'JetBrains Mono', monospace" }}>cosine · {source.score}</span>}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Thinking indicator ───────────────────────────────────────────────────────
function ThinkingIndicator() {
  const C = useTheme();
  return (
    <div className="flex gap-4">
      <div className="shrink-0 mt-1 w-7 h-7 rounded flex items-center justify-center animate-pulse"
        style={{ backgroundColor: C.navy }}>
        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: C.highlight }} />
      </div>
      <div className="flex-1">
        <div className="text-[10px] tracking-[0.25em] uppercase font-bold mb-2" style={{ color: C.highlight }}>
          Retrieving sources…
        </div>
        <div className="flex gap-1.5">
          {[0, 1, 2].map(i => (
            <span key={i} className="w-2 h-2 rounded-full animate-bounce"
              style={{ backgroundColor: C.highlight, animationDelay: `${i * 0.15}s` }} />
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── Modal wrapper ────────────────────────────────────────────────────────────
function Modal({ children, onClose }) {
  const C = useTheme();
  return (
    <div className="fixed inset-0 z-[200] flex items-center justify-center p-4"
      style={{ backgroundColor: "#041E42cc", backdropFilter: "blur(4px)" }}>
      <div className="max-w-lg w-full max-h-[85vh] overflow-y-auto shadow-2xl border rounded"
        style={{ backgroundColor: C.bg, borderColor: C.lineStrong }}>
        <div className="flex justify-end p-3">
          <button onClick={onClose} className="p-1 rounded transition hover:opacity-70">
            <X size={18} style={{ color: C.ink }} />
          </button>
        </div>
        <div className="px-8 pb-8">{children}</div>
      </div>
    </div>
  );
}

// ─── How it works ─────────────────────────────────────────────────────────────
function HowItWorks() {
  const C = useTheme();
  const steps = [
    { n: "01", title: "Retrieval",    desc: "Your query is embedded and matched against an indexed corpus of research documents — grants, reports, and publications." },
    { n: "02", title: "Augmentation", desc: "The most relevant excerpts are passed to the language model as grounding context, alongside your question." },
    { n: "03", title: "Generation",   desc: "A response is composed from retrieved evidence, with every claim traceable to a cited source." },
  ];
  return (
    <div>
      <span className="text-[10px] tracking-[0.3em] uppercase font-bold" style={{ color: C.highlight }}>The pipeline</span>
      <h2 className="text-3xl mt-2 mb-6 font-bold" style={{ fontFamily: "'Roboto Condensed', sans-serif", color: C.green }}>
        How it works
      </h2>
      <div className="space-y-5">
        {steps.map(s => (
          <div key={s.n} className="flex gap-4">
            <span className="text-3xl shrink-0 font-bold" style={{ fontFamily: "'Roboto Condensed', sans-serif", color: C.highlight }}>
              {s.n}
            </span>
            <div>
              <h3 className="font-bold mb-1 text-sm" style={{ color: C.accent }}>{s.title}</h3>
              <p className="text-sm leading-relaxed" style={{ color: C.inkSoft }}>{s.desc}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Limitations ─────────────────────────────────────────────────────────────
function Limitations() {
  const C = useTheme();
  const items = [
    "AI may miss nuance or contextual subtlety that a domain expert would catch.",
    "Answers are bounded by the documents available in the indexed corpus.",
    "This tool is not a replacement for peer-reviewed expert review or clinical judgment.",
    "Confidence scores are heuristic and should be interpreted with care.",
  ];
  return (
    <div>
      <span className="text-[10px] tracking-[0.3em] uppercase font-bold" style={{ color: C.highlight }}>Honesty</span>
      <h2 className="text-3xl mt-2 mb-6 font-bold" style={{ fontFamily: "'Roboto Condensed', sans-serif", color: C.green }}>
        Limitations
      </h2>
      <ul className="space-y-3">
        {items.map((it, i) => (
          <li key={i} className="flex gap-3 text-sm leading-relaxed">
            <span className="shrink-0 font-bold" style={{ color: C.highlight }}>—</span>
            <span style={{ color: C.inkSoft }}>{it}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

// ─── Upload panel ─────────────────────────────────────────────────────────────
function UploadPanel({ authed, pw, setPw, error, onAuth, loading, files, onDrop, dragOver, setDragOver }) {
  const C = useTheme();
  if (!authed) {
    return (
      <div>
        <span className="text-[10px] tracking-[0.3em] uppercase font-bold" style={{ color: C.highlight }}>Protected</span>
        <h2 className="text-3xl mt-2 mb-2 font-bold" style={{ fontFamily: "'Roboto Condensed', sans-serif", color: C.green }}>
          Add to Database
        </h2>
        <p className="text-sm mb-6" style={{ color: C.inkSoft }}>
          Database submission is access-controlled. Enter the upload password to continue.
        </p>
        <input
          type="password" value={pw}
          onChange={e => setPw(e.target.value)}
          onKeyDown={e => e.key === "Enter" && onAuth()}
          placeholder="Upload password"
          className="w-full px-4 py-3 text-sm border rounded outline-none tracking-widest text-center mb-3"
          style={{ fontFamily: "'JetBrains Mono', monospace", borderColor: C.lineStrong, color: C.ink }}
          onFocus={e => (e.target.style.borderColor = C.highlight)}
          onBlur={e => (e.target.style.borderColor = C.lineStrong)}
        />
        {error && (
          <p className="text-xs mb-3 flex items-center justify-center gap-2" style={{ color: C.error }}>
            <AlertCircle size={12} /> {error}
          </p>
        )}
        <button onClick={onAuth} disabled={loading}
          className="w-full py-3 text-xs tracking-[0.25em] uppercase font-bold rounded transition disabled:opacity-40"
          style={{ backgroundColor: C.accent, color: "#ffffff" }}
          onMouseEnter={e => (e.currentTarget.style.backgroundColor = C.highlight)}
          onMouseLeave={e => (e.currentTarget.style.backgroundColor = C.accent)}>
          {loading ? "Checking…" : "Unlock →"}
        </button>
      </div>
    );
  }
  return (
    <div>
      <span className="text-[10px] tracking-[0.3em] uppercase font-bold" style={{ color: C.highlight }}>Unlocked</span>
      <h2 className="text-3xl mt-2 mb-6 font-bold" style={{ fontFamily: "'Roboto Condensed', sans-serif", color: C.green }}>
        Add to corpus
      </h2>
      <div
        onDragOver={e => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        className="border-2 border-dashed rounded-lg p-10 text-center transition"
        style={{ borderColor: dragOver ? C.highlight : C.lineStrong,
          backgroundColor: dragOver ? C.highlightSoft : "transparent" }}>
        <Upload size={32} className="mx-auto mb-3" style={{ color: C.inkSoft }} />
        <p className="text-sm font-medium mb-1" style={{ color: C.ink }}>Drop PDFs, slides, or docs here</p>
        <p className="text-xs" style={{ color: C.inkSoft }}>max 50 MB per file</p>
      </div>
      {files.length > 0 && (
        <div className="mt-4 space-y-1.5">
          <p className="text-[10px] tracking-[0.25em] uppercase font-bold mb-2" style={{ color: C.inkSoft }}>
            Queued · {files.length}
          </p>
          {files.map((f, i) => (
            <div key={i} className="flex items-center gap-2 text-sm border rounded px-3 py-2"
              style={{ borderColor: C.line }}>
              <FileText size={14} style={{ color: C.highlight }} />
              <span className="flex-1 truncate" style={{ color: C.ink }}>{f.name}</span>
              <span className="text-xs" style={{ color: C.inkSoft, fontFamily: "'JetBrains Mono', monospace" }}>{f.size}</span>
            </div>
          ))}
        </div>
      )}
      <p className="text-[10px] mt-5 leading-relaxed" style={{ color: C.inkSoft }}>
        <Shield size={10} className="inline mr-1" />
        Submitted documents are stored privately and are never used to train models.
      </p>
    </div>
  );
}

// ─── Prompt menu ─────────────────────────────────────────────────────────────
function PromptMenu({ categories, activeCategory, setActiveCategory, search, setSearch, onInsert, onClose }) {
  const C = useTheme();
  const low      = search.trim().toLowerCase();
  const searching = low.length > 0;
  const visible  = searching
    ? categories.flatMap(c => c.prompts.filter(p => p.toLowerCase().includes(low))
        .map(p => ({ text: p, category: c.label })))
    : (categories.find(c => c.id === activeCategory)?.prompts || []).map(p => ({ text: p, category: null }));
  const activeDesc = categories.find(c => c.id === activeCategory)?.desc;

  return (
    <div className="border-t" style={{ borderColor: C.line, backgroundColor: C.panel }}>
      <div className="max-w-3xl mx-auto px-6 pt-5 pb-4">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Wand2 size={14} style={{ color: C.highlight }} />
            <span className="text-[11px] tracking-[0.3em] uppercase font-bold" style={{ color: C.inkSoft }}>
              Prompt Library
            </span>
          </div>
          <button onClick={onClose} className="p-1 rounded transition hover:opacity-70">
            <X size={15} style={{ color: C.inkSoft }} />
          </button>
        </div>

        <div className="relative mb-4">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2" style={{ color: C.inkSoft }} />
          <input value={search} onChange={e => setSearch(e.target.value)}
            placeholder="Search prompts…"
            className="w-full pl-9 pr-3 py-2 text-sm outline-none border rounded"
            style={{ borderColor: C.line, color: C.ink, backgroundColor: C.bg }}
            onFocus={e => (e.target.style.borderColor = C.highlight)}
            onBlur={e => (e.target.style.borderColor = C.line)} />
        </div>

        {!searching && (
          <>
            <div className="flex flex-wrap gap-1 mb-3">
              {categories.map(c => {
                const on = c.id === activeCategory;
                return (
                  <button key={c.id} onClick={() => setActiveCategory(c.id)}
                    className="px-3 py-1.5 text-xs border rounded-full transition font-bold"
                    style={on
                      ? { backgroundColor: C.highlight, color: C.navy, borderColor: C.highlight }
                      : { backgroundColor: "transparent", borderColor: C.lineStrong, color: C.ink }}
                    onMouseEnter={e => { if (!on) e.currentTarget.style.borderColor = C.highlight; }}
                    onMouseLeave={e => { if (!on) e.currentTarget.style.borderColor = C.lineStrong; }}>
                    {c.label}
                  </button>
                );
              })}
            </div>
            {activeDesc && (
              <p className="text-xs italic mb-3 pl-1" style={{ color: C.inkSoft }}>{activeDesc}</p>
            )}
          </>
        )}

        {visible.length === 0 ? (
          <div className="text-center text-xs py-8" style={{ color: C.inkSoft }}>
            No prompts match "{search}"
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-h-64 overflow-y-auto pr-1">
            {visible.map((p, i) => (
              <button key={`${p.text}-${i}`}
                onClick={() => onInsert(p.text)}
                className="group flex items-start gap-2 p-3 border rounded text-left transition"
                style={{ backgroundColor: C.bg, borderColor: C.line }}
                onMouseEnter={e => { e.currentTarget.style.borderColor = C.highlight; }}
                onMouseLeave={e => { e.currentTarget.style.borderColor = C.line; }}>
                <div className="flex-1 min-w-0">
                  {p.category && (
                    <div className="text-[9px] tracking-[0.25em] uppercase font-bold mb-1"
                      style={{ color: C.highlight }}>{p.category}</div>
                  )}
                  <p className="text-sm leading-snug" style={{ color: C.ink }}>{p.text}</p>
                </div>
                <ChevronRight size={14} className="opacity-0 group-hover:opacity-100 transition shrink-0 mt-0.5"
                  style={{ color: C.highlight }} />
              </button>
            ))}
          </div>
        )}

        <p className="text-[10px] mt-3 text-center" style={{ color: C.inkSoft }}>
          Click a prompt to paste it into the input · edit before sending
        </p>
      </div>
    </div>
  );
}
