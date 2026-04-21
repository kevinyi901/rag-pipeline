import React, { useState, useEffect, useRef } from "react";
import {
  HelpCircle, Wand2, X, Send, FileText,
  Filter, ChevronDown, ChevronRight, Upload, Lock, AlertCircle,
  BookOpen, Info, Shield, Trash2, ExternalLink, Search,
  Quote, Building2,
} from "lucide-react";

const C = {
  bg:            "#ecebe3",
  panel:         "#e1dfd2",
  panelSoft:     "#e7e5d8",
  ink:           "#1a1d18",
  inkSoft:       "#55584e",
  line:          "#1a1d1822",
  lineStrong:    "#1a1d1844",
  accent:        "#2f5d4e",
  accentSoft:    "#2f5d4e15",
  highlight:     "#b8571f",
  highlightSoft: "#b8571f15",
  error:         "#8a2a1f",
};

const API_URL      = import.meta.env.VITE_API_URL || "";
const API_KEY      = import.meta.env.VITE_API_KEY || "";
const APP_PASSWORD = import.meta.env.VITE_PASSWORD || "1234";

async function queryBackend(question) {
  if (!API_URL) throw new Error("VITE_API_URL is not set.");
  const headers = { "Content-Type": "application/json" };
  if (API_KEY) headers["Authorization"] = `Bearer ${API_KEY}`;
  const r = await fetch(API_URL, {
    method: "POST",
    headers,
    body: JSON.stringify({ query: question }),
  });
  if (!r.ok) {
    const body = await r.text().catch(() => "");
    throw new Error(`API ${r.status}: ${body}`);
  }
  return r.json();
}

export default function SightReader() {
  useEffect(() => {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href =
      "https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,400;9..144,500;9..144,600;9..144,700;9..144,900&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap";
    document.head.appendChild(link);
  }, []);

  const [authed, setAuthed] = useState(false);
  const [pwInput, setPwInput] = useState("");
  const [pwError, setPwError] = useState(false);

  const [menuOpen, setMenuOpen] = useState(false);
  const [activePanel, setActivePanel] = useState(null);
  const [uploadAuthed, setUploadAuthed] = useState(false);
  const [uploadPw, setUploadPw] = useState("");
  const [uploadPwError, setUploadPwError] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [dragOver, setDragOver] = useState(false);

  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [activeTab, setActiveTab] = useState("filters");
  const [expandedSource, setExpandedSource] = useState(null);

  const [yearRange, setYearRange] = useState(2018);
  const [selectedReqTypes, setSelectedReqTypes] = useState([]);
  const [selectedPrograms, setSelectedPrograms] = useState(["Vision Science"]);
  const [selectedAwardTypes, setSelectedAwardTypes] = useState([]);
  const [budgetMax, setBudgetMax] = useState(2000000);
  const [selectedInst, setSelectedInst] = useState("All institutions");
  const [selectedCountry, setSelectedCountry] = useState("All countries");

  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [thinking, setThinking] = useState(false);

  const [promptOpen, setPromptOpen] = useState(false);
  const [promptCategory, setPromptCategory] = useState("semantic");
  const [promptSearch, setPromptSearch] = useState("");

  const inputRef  = useRef(null);
  const chatEndRef = useRef(null);
  const menuRef   = useRef(null);
  const menuBtnRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, thinking]);

  useEffect(() => {
    if (!menuOpen) return;
    const onClick = (e) => {
      if (
        menuRef.current && !menuRef.current.contains(e.target) &&
        menuBtnRef.current && !menuBtnRef.current.contains(e.target)
      ) setMenuOpen(false);
    };
    const onKey = (e) => e.key === "Escape" && setMenuOpen(false);
    document.addEventListener("mousedown", onClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [menuOpen]);

  const REQUIREMENT_TYPES = ["Open Call", "RFA", "Strategic Initiative", "Fellowship", "Pilot"];
  const PROGRAM_AREAS     = ["Vision Science", "Neurodegeneration", "Gene Therapy", "Translational", "Clinical Research", "Basic Science"];
  const AWARD_TYPES       = ["R01", "R21", "K99", "Fellowship", "Seed Grant", "Program Project"];
  const INSTITUTIONS      = ["All institutions", "Johns Hopkins", "Moorfields", "NEI", "NIH", "Mass Eye & Ear"];
  const COUNTRIES         = ["All countries", "United States", "United Kingdom", "Germany", "France", "Japan", "Canada", "Australia", "Netherlands"];

  const PROMPT_CATEGORIES = [
    {
      id: "semantic",
      label: "Semantic Search",
      desc: "Pinpoint mentions, funded work, and entities across the corpus.",
      prompts: [
        "Have we ever funded research on gene BEST1?",
        "Show every mention of [gene] with context.",
        "Find all companies in Phase 1 trials.",
      ],
    },
    {
      id: "outcome",
      label: "Outcome",
      desc: "Surface what was achieved, what stalled, and what came out of a project.",
      prompts: [
        "What did this project ultimately achieve?",
        "What barriers are described in this report?",
        "What outputs or tools came from this project?",
      ],
    },
    {
      id: "compare",
      label: "Cross-Comparison",
      desc: "Hold two things next to each other — years, documents, trajectories.",
      prompts: [
        "Compare Year 2 to Year 1 for this project.",
        "Compare the proposal to the latest progress report.",
        "Show progress over time for this PI or project.",
      ],
    },
    {
      id: "audience",
      label: "Audience",
      desc: "Reshape the same evidence for a different reader.",
      prompts: [
        "Summarize this for a layperson.",
        "Write a donor update.",
        "Explain key technical terms simply.",
      ],
    },
  ];

  const insertPrompt = (text) => {
    setInput(text);
    setPromptOpen(false);
    setTimeout(() => inputRef.current?.focus(), 50);
  };

  const toggleSet = (val, list, setter) =>
    setter(list.includes(val) ? list.filter((x) => x !== val) : [...list, val]);

  const tryLogin = () => {
    if (pwInput.trim() === APP_PASSWORD) { setAuthed(true); setPwError(false); }
    else setPwError(true);
  };

  const tryUploadAuth = () => {
    if (uploadPw.trim() === APP_PASSWORD) { setUploadAuthed(true); setUploadPwError(false); }
    else setUploadPwError(true);
  };

  const send = async () => {
    const q = input.trim();
    if (!q || thinking) return;
    setMessages((m) => [...m, { role: "user", text: q, id: Date.now() }]);
    setInput("");
    setThinking(true);
    setActiveTab("sources");

    try {
      const data = await queryBackend(q);
      setMessages((m) => [...m, {
        role: "ai",
        id: Date.now() + 1,
        text: data.response || "No response received.",
        sources: data.sources || [],
      }]);
    } catch (err) {
      setMessages((m) => [...m, {
        role: "ai",
        id: Date.now() + 1,
        text: `Error: ${err.message}`,
        sources: [],
      }]);
    } finally {
      setThinking(false);
    }
  };

  const clearConvo = () => { setMessages([]); setExpandedSource(null); };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const files = Array.from(e.dataTransfer.files).map((f) => ({
      name: f.name,
      size: (f.size / 1024).toFixed(1) + " KB",
    }));
    setUploadedFiles((prev) => [...prev, ...files]);
  };

  const latestSources =
    [...messages].reverse().find((m) => m.role === "ai")?.sources || [];

  if (!authed) {
    return (
      <div
        className="min-h-screen w-full flex items-center justify-center relative overflow-hidden"
        style={{
          fontFamily: "'Inter', sans-serif",
          background: `radial-gradient(ellipse at center, ${C.bg} 0%, #e3e0d2 60%, #d4d1c0 100%)`,
          color: C.ink,
        }}
      >
        <div
          className="absolute inset-0 opacity-25 pointer-events-none mix-blend-multiply"
          style={{
            backgroundImage:
              "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='200' height='200'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2'/><feColorMatrix values='0 0 0 0 0.1 0 0 0 0 0.1 0 0 0 0 0.08 0 0 0 0.4 0'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>\")",
          }}
        />
        <div className="relative z-10 w-full max-w-md px-8">
          <div className="flex justify-center mb-10">
            <EyeLogo size={88} />
          </div>
          <h1
            className="text-center leading-none tracking-tight"
            style={{
              fontFamily: "'Fraunces', serif",
              fontWeight: 400,
              fontSize: "68px",
              fontVariationSettings: "'opsz' 144",
              fontStyle: "italic",
            }}
          >
            SightReader
          </h1>
          <p
            className="text-center mt-4 text-sm tracking-wide max-w-xs mx-auto leading-relaxed"
            style={{ color: C.inkSoft }}
          >
            An AI-powered research assistant for grounded scientific insight, citation-first.
          </p>
          <div className="mt-10 space-y-4">
            <input
              type="password"
              value={pwInput}
              onChange={(e) => { setPwInput(e.target.value); setPwError(false); }}
              onKeyDown={(e) => e.key === "Enter" && tryLogin()}
              placeholder="Enter password"
              className="w-full bg-transparent border-b px-1 py-3 outline-none transition text-center tracking-widest"
              style={{
                fontFamily: "'JetBrains Mono', monospace",
                borderColor: C.lineStrong,
                color: C.ink,
              }}
              onFocus={(e) => (e.target.style.borderColor = C.ink)}
              onBlur={(e) => (e.target.style.borderColor = C.lineStrong)}
            />
            {pwError && (
              <div className="flex items-center justify-center gap-2 text-xs" style={{ color: C.error }}>
                <AlertCircle size={12} />
                Incorrect password. Try again.
              </div>
            )}
            <button
              onClick={tryLogin}
              className="w-full py-3.5 tracking-[0.25em] text-xs uppercase transition-colors duration-300"
              style={{ backgroundColor: C.ink, color: C.bg }}
              onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = C.accent)}
              onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = C.ink)}
            >
              Enter →
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      className="h-screen w-full flex flex-col overflow-hidden"
      style={{ fontFamily: "'Inter', sans-serif", backgroundColor: C.bg, color: C.ink }}
    >
      {/* Header */}
      <header
        className="flex items-center justify-between px-5 py-3 border-b"
        style={{ borderColor: C.line, backgroundColor: C.bg }}
      >
        <div className="flex items-center gap-1">
          <button
            onClick={() => setSidebarOpen((v) => !v)}
            className="p-2 rounded transition hidden md:block hover:bg-[#1a1d1810]"
            aria-label="Toggle filters"
            title="Filters"
          >
            <Filter size={16} />
          </button>
        </div>

        <span
          style={{
            fontFamily: "'Fraunces', serif",
            fontStyle: "italic",
            fontSize: "24px",
            fontWeight: 500,
            letterSpacing: "-0.01em",
          }}
        >
          SightReader
        </span>

        <div className="flex items-center gap-1">
          <button
            ref={menuBtnRef}
            onClick={() => setMenuOpen((v) => !v)}
            className="p-2 rounded transition hover:bg-[#1a1d1810]"
            style={{ color: menuOpen ? C.accent : C.inkSoft }}
            aria-label="Help"
            title="Help"
          >
            <HelpCircle size={17} />
          </button>
          <button
            onClick={clearConvo}
            className="flex items-center gap-1.5 text-xs tracking-wider uppercase px-3 py-2 rounded transition hover:bg-[#1a1d1810]"
            style={{ color: C.inkSoft }}
          >
            <Trash2 size={13} />
            <span className="hidden sm:inline">Clear</span>
          </button>
        </div>
      </header>

      {/* Help menu */}
      {menuOpen && (
        <div
          ref={menuRef}
          className="fixed right-4 top-[56px] w-72 z-[200] border shadow-xl overflow-hidden"
          style={{
            backgroundColor: C.bg,
            borderColor: C.lineStrong,
            boxShadow: "0 12px 40px rgba(26, 29, 24, 0.18)",
          }}
        >
          <div className="px-4 py-2.5 border-b" style={{ borderColor: C.line, backgroundColor: C.panelSoft }}>
            <span className="text-[10px] tracking-[0.3em] uppercase" style={{ color: C.inkSoft }}>
              Help & More
            </span>
          </div>
          {[
            { key: "how",    icon: Info,        title: "How it works",      desc: "The RAG pipeline, briefly" },
            { key: "limits", icon: AlertCircle, title: "Limitations",       desc: "What this tool will not do" },
            { key: "upload", icon: Lock,        title: "Add to Database",   desc: "Protected · password required", top: true },
          ].map((item) => (
            <button
              key={item.key}
              onClick={() => { setActivePanel(item.key); setMenuOpen(false); }}
              className="w-full flex items-center gap-3 px-4 py-3 transition text-left hover:bg-[#2f5d4e10]"
              style={item.top ? { borderTop: `1px solid ${C.line}` } : {}}
            >
              <item.icon size={16} style={{ color: C.accent }} />
              <div>
                <div className="text-sm font-medium">{item.title}</div>
                <div className="text-xs" style={{ color: C.inkSoft }}>{item.desc}</div>
              </div>
            </button>
          ))}
        </div>
      )}

      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        {sidebarOpen && (
          <aside
            className="w-80 border-r flex flex-col overflow-hidden"
            style={{ borderColor: C.line, backgroundColor: C.panel }}
          >
            <div className="flex border-b" style={{ borderColor: C.line }}>
              {["filters", "sources"].map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className="flex-1 py-3 text-[10px] tracking-[0.25em] uppercase transition relative"
                  style={
                    activeTab === tab
                      ? { backgroundColor: C.accentSoft, color: C.accent, borderBottom: `2px solid ${C.accent}` }
                      : { color: C.inkSoft }
                  }
                  onMouseEnter={(e) => { if (activeTab !== tab) e.currentTarget.style.backgroundColor = "#1a1d1808"; }}
                  onMouseLeave={(e) => { if (activeTab !== tab) e.currentTarget.style.backgroundColor = "transparent"; }}
                >
                  {tab}
                  {tab === "sources" && latestSources.length > 0 && (
                    <span
                      className="ml-2 inline-block px-1.5 py-0.5 text-[9px] rounded-full"
                      style={{ backgroundColor: C.highlight, color: C.bg }}
                    >
                      {latestSources.length}
                    </span>
                  )}
                </button>
              ))}
            </div>

            <div className="flex-1 overflow-y-auto">
              {activeTab === "filters" ? (
                <div className="p-5 space-y-6">
                  <ChipGroup
                    label="Requirement Type"
                    options={REQUIREMENT_TYPES}
                    selected={selectedReqTypes}
                    onToggle={(v) => toggleSet(v, selectedReqTypes, setSelectedReqTypes)}
                  />

                  <div>
                    <label className="text-[10px] tracking-[0.25em] uppercase block mb-3" style={{ color: C.inkSoft }}>
                      Date
                      <span className="float-right normal-case tracking-normal" style={{ color: C.ink, fontFamily: "'JetBrains Mono', monospace" }}>
                        ≥ {yearRange}
                      </span>
                    </label>
                    <input
                      type="range" min="2000" max="2026" value={yearRange}
                      onChange={(e) => setYearRange(e.target.value)}
                      className="w-full"
                      style={{ accentColor: C.accent }}
                    />
                    <div className="flex justify-between text-[10px] mt-1" style={{ color: C.inkSoft, fontFamily: "'JetBrains Mono', monospace" }}>
                      <span>2000</span><span>2026</span>
                    </div>
                  </div>

                  <ChipGroup
                    label="Program Area"
                    options={PROGRAM_AREAS}
                    selected={selectedPrograms}
                    onToggle={(v) => toggleSet(v, selectedPrograms, setSelectedPrograms)}
                  />

                  <ChipGroup
                    label="Award Type"
                    options={AWARD_TYPES}
                    selected={selectedAwardTypes}
                    onToggle={(v) => toggleSet(v, selectedAwardTypes, setSelectedAwardTypes)}
                  />

                  <div>
                    <label className="text-[10px] tracking-[0.25em] uppercase block mb-3" style={{ color: C.inkSoft }}>
                      Actual Award Budget
                      <span className="float-right normal-case tracking-normal" style={{ color: C.ink, fontFamily: "'JetBrains Mono', monospace" }}>
                        ≤ ${(budgetMax / 1000).toLocaleString()}k
                      </span>
                    </label>
                    <input
                      type="range" min="50000" max="5000000" step="50000" value={budgetMax}
                      onChange={(e) => setBudgetMax(Number(e.target.value))}
                      className="w-full"
                      style={{ accentColor: C.accent }}
                    />
                    <div className="flex justify-between text-[10px] mt-1" style={{ color: C.inkSoft, fontFamily: "'JetBrains Mono', monospace" }}>
                      <span>$50k</span><span>$5M</span>
                    </div>
                  </div>

                  <div>
                    <label className="text-[10px] tracking-[0.25em] uppercase block mb-3" style={{ color: C.inkSoft }}>
                      Institution
                    </label>
                    <select
                      value={selectedInst}
                      onChange={(e) => setSelectedInst(e.target.value)}
                      className="w-full bg-transparent px-3 py-2 text-sm outline-none border"
                      style={{ borderColor: C.lineStrong, color: C.ink }}
                    >
                      {INSTITUTIONS.map((i) => <option key={i}>{i}</option>)}
                    </select>
                  </div>

                  <div>
                    <label className="text-[10px] tracking-[0.25em] uppercase block mb-3" style={{ color: C.inkSoft }}>
                      Country
                    </label>
                    <select
                      value={selectedCountry}
                      onChange={(e) => setSelectedCountry(e.target.value)}
                      className="w-full bg-transparent px-3 py-2 text-sm outline-none border"
                      style={{ borderColor: C.lineStrong, color: C.ink }}
                    >
                      {COUNTRIES.map((c) => <option key={c}>{c}</option>)}
                    </select>
                  </div>

                  <div className="pt-4 border-t" style={{ borderColor: C.line }}>
                    <button
                      onClick={() => {
                        setSelectedReqTypes([]);
                        setYearRange(2018);
                        setSelectedPrograms([]);
                        setSelectedAwardTypes([]);
                        setBudgetMax(2000000);
                        setSelectedInst("All institutions");
                        setSelectedCountry("All countries");
                      }}
                      className="w-full text-[10px] tracking-[0.25em] uppercase py-2 border transition"
                      style={{ borderColor: C.lineStrong, color: C.inkSoft }}
                      onMouseEnter={(e) => { e.currentTarget.style.borderColor = C.highlight; e.currentTarget.style.color = C.highlight; }}
                      onMouseLeave={(e) => { e.currentTarget.style.borderColor = C.lineStrong; e.currentTarget.style.color = C.inkSoft; }}
                    >
                      Reset all filters
                    </button>
                  </div>
                </div>
              ) : (
                <div className="p-4 space-y-3">
                  {latestSources.length === 0 ? (
                    <div className="text-center text-sm py-12" style={{ color: C.inkSoft }}>
                      <BookOpen size={28} className="mx-auto mb-3 opacity-40" />
                      Retrieved sources will appear here after you ask a question.
                    </div>
                  ) : (
                    <>
                      <p className="text-[10px] tracking-[0.25em] uppercase mb-2" style={{ color: C.inkSoft }}>
                        Retrieved · {latestSources.length} documents
                      </p>
                      {latestSources.map((s, i) => (
                        <div
                          key={s.id || i}
                          onClick={() => setExpandedSource(expandedSource === (s.id || i) ? null : (s.id || i))}
                          className="p-3 transition cursor-pointer border"
                          style={{
                            backgroundColor: C.bg,
                            borderColor: expandedSource === (s.id || i) ? C.accent : C.line,
                          }}
                        >
                          <div className="flex items-start gap-2 mb-1.5">
                            <span
                              className="text-[10px] mt-0.5 px-1.5 py-0.5 rounded shrink-0"
                              style={{ backgroundColor: C.ink, color: C.bg, fontFamily: "'JetBrains Mono', monospace" }}
                            >
                              {String(i + 1).padStart(2, "0")}
                            </span>
                            <h4 className="text-sm font-medium leading-snug flex-1">{s.title}</h4>
                          </div>
                          <div className="text-xs flex items-center gap-2 flex-wrap mb-1" style={{ color: C.inkSoft }}>
                            {s.authors && <><span>{s.authors}</span><span>·</span></>}
                            {s.year && <><span>{s.year}</span><span>·</span></>}
                            {s.type && <span className="px-1.5 py-0.5 text-[10px]" style={{ backgroundColor: C.line }}>{s.type}</span>}
                          </div>
                          {expandedSource === (s.id || i) && s.excerpt && (
                            <div className="mt-2 pt-2 border-t" style={{ borderColor: C.line }}>
                              <p
                                className="text-xs italic leading-relaxed p-2 border-l-2"
                                style={{ fontFamily: "'Fraunces', serif", backgroundColor: C.accentSoft, borderColor: C.accent }}
                              >
                                "{s.excerpt}"
                              </p>
                              <div className="flex items-center justify-between mt-2 text-[10px]" style={{ color: C.inkSoft }}>
                                {s.institution && (
                                  <span className="flex items-center gap-1">
                                    <Building2 size={10} />{s.institution}
                                  </span>
                                )}
                                {s.score != null && (
                                  <span style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                                    score {s.score}
                                  </span>
                                )}
                              </div>
                            </div>
                          )}
                        </div>
                      ))}
                    </>
                  )}
                </div>
              )}
            </div>
          </aside>
        )}

        {/* Main chat area */}
        <main className="flex-1 flex flex-col overflow-hidden">
          <div className="flex-1 overflow-y-auto">
            {messages.length === 0 ? (
              <EmptyState />
            ) : (
              <div className="max-w-3xl mx-auto px-6 py-8 space-y-8">
                {messages.map((m) =>
                  m.role === "user" ? (
                    <div key={m.id} className="flex justify-end">
                      <div
                        className="max-w-[85%] px-5 py-3 rounded-2xl rounded-tr-sm"
                        style={{ backgroundColor: C.ink, color: C.bg }}
                      >
                        <p className="leading-relaxed">{m.text}</p>
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

          {/* Prompt library drawer */}
          {promptOpen && (
            <PromptMenu
              categories={PROMPT_CATEGORIES}
              activeCategory={promptCategory}
              setActiveCategory={setPromptCategory}
              search={promptSearch}
              setSearch={setPromptSearch}
              onInsert={insertPrompt}
              onClose={() => setPromptOpen(false)}
            />
          )}

          {/* Input bar */}
          <div className="border-t px-6 py-5" style={{ borderColor: C.line, backgroundColor: C.bg }}>
            <div className="max-w-3xl mx-auto">
              <div
                className="flex items-end gap-3 border rounded-2xl pl-3 pr-3 py-3 transition min-h-[68px]"
                style={{ backgroundColor: C.panelSoft, borderColor: C.lineStrong }}
              >
                <button
                  onClick={() => setPromptOpen((v) => !v)}
                  className="shrink-0 p-2 rounded-xl transition self-end mb-0.5"
                  style={{
                    backgroundColor: promptOpen ? C.accent : "transparent",
                    color: promptOpen ? C.bg : C.inkSoft,
                  }}
                  onMouseEnter={(e) => { if (!promptOpen) { e.currentTarget.style.backgroundColor = C.accentSoft; e.currentTarget.style.color = C.accent; } }}
                  onMouseLeave={(e) => { if (!promptOpen) { e.currentTarget.style.backgroundColor = "transparent"; e.currentTarget.style.color = C.inkSoft; } }}
                  aria-label="Prompt library"
                  title="Prompt library"
                >
                  <Wand2 size={17} />
                </button>
                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
                  rows={2}
                  placeholder="Ask a scientific question…"
                  className="flex-1 bg-transparent outline-none resize-none text-[15px] leading-[1.55] max-h-40 py-1"
                  style={{ fontFamily: "'Inter', sans-serif", color: C.ink }}
                />
                <button
                  onClick={send}
                  disabled={!input.trim() || thinking}
                  className="shrink-0 p-2.5 rounded-xl transition disabled:opacity-30 disabled:cursor-not-allowed self-end"
                  style={{ backgroundColor: C.ink, color: C.bg }}
                  onMouseEnter={(e) => { if (input.trim() && !thinking) e.currentTarget.style.backgroundColor = C.accent; }}
                  onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = C.ink)}
                >
                  <Send size={16} />
                </button>
              </div>
            </div>
          </div>
        </main>
      </div>

      {/* Modals */}
      {activePanel && (
        <Modal onClose={() => {
          setActivePanel(null);
          setUploadAuthed(false);
          setUploadPw("");
          setUploadPwError(false);
        }}>
          {activePanel === "how"    && <HowItWorks />}
          {activePanel === "limits" && <Limitations />}
          {activePanel === "upload" && (
            <UploadPanel
              authed={uploadAuthed} pw={uploadPw} setPw={setUploadPw}
              error={uploadPwError} onAuth={tryUploadAuth}
              files={uploadedFiles} onDrop={handleDrop}
              dragOver={dragOver} setDragOver={setDragOver}
            />
          )}
        </Modal>
      )}
    </div>
  );
}

// ─── Sub-components ──────────────────────────────────────────────────────────

function EyeLogo({ size = 88 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 88 88" fill="none">
      <ellipse cx="44" cy="44" rx="40" ry="22" stroke={C.ink} strokeWidth="1.5" fill="none" />
      <circle cx="44" cy="44" r="16" fill={C.ink} />
      <circle cx="44" cy="44" r="7" fill={C.accent} />
      <circle cx="48" cy="40" r="2.5" fill={C.bg} />
      <line x1="44" y1="22" x2="44" y2="14" stroke={C.ink} strokeWidth="1.5" strokeLinecap="round" />
      <line x1="30" y1="25" x2="26" y2="18" stroke={C.ink} strokeWidth="1.5" strokeLinecap="round" />
      <line x1="58" y1="25" x2="62" y2="18" stroke={C.ink} strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

function ChipGroup({ label, options, selected, onToggle }) {
  return (
    <div>
      <label className="text-[10px] tracking-[0.25em] uppercase block mb-3" style={{ color: C.inkSoft }}>
        {label}
      </label>
      <div className="flex flex-wrap gap-1.5">
        {options.map((t) => {
          const on = selected.includes(t);
          return (
            <button
              key={t}
              onClick={() => onToggle(t)}
              className="px-2.5 py-1 text-xs border transition"
              style={on
                ? { backgroundColor: C.accentSoft, color: C.accent, borderColor: C.accent }
                : { borderColor: C.lineStrong, color: C.ink }
              }
              onMouseEnter={(e) => { if (!on) e.currentTarget.style.borderColor = C.accent; }}
              onMouseLeave={(e) => { if (!on) e.currentTarget.style.borderColor = C.lineStrong; }}
            >
              {t}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="h-full flex flex-col items-center justify-center px-6 max-w-2xl mx-auto text-center">
      <div className="mb-6">
        <svg width="64" height="64" viewBox="0 0 88 88" fill="none">
          <ellipse cx="44" cy="44" rx="40" ry="22" stroke={C.ink} strokeWidth="2" fill="none" />
          <circle cx="44" cy="44" r="16" fill={C.ink} />
          <circle cx="44" cy="44" r="7" fill={C.accent} />
          <circle cx="48" cy="40" r="2.5" fill={C.bg} />
        </svg>
      </div>
      <h2
        className="text-4xl md:text-5xl mb-4 leading-tight"
        style={{ fontFamily: "'Fraunces', serif", fontStyle: "italic", fontWeight: 400 }}
      >
        What would you like to know?
      </h2>
      <p className="max-w-md leading-relaxed" style={{ color: C.inkSoft }}>
        Ask a scientific question. SightReader retrieves relevant documents and grounds every claim in cited sources.
      </p>
    </div>
  );
}

function AIMessage({ message }) {
  const [showSources, setShowSources] = useState(true);
  return (
    <div className="flex gap-4">
      <div className="shrink-0 mt-1">
        <svg width="28" height="28" viewBox="0 0 88 88" fill="none">
          <ellipse cx="44" cy="44" rx="40" ry="22" stroke={C.ink} strokeWidth="3" fill="none" />
          <circle cx="44" cy="44" r="16" fill={C.ink} />
          <circle cx="44" cy="44" r="7" fill={C.accent} />
        </svg>
      </div>
      <div className="flex-1 min-w-0">
        <div className="text-[10px] tracking-[0.25em] uppercase mb-2" style={{ color: C.inkSoft }}>
          SightReader · Response
        </div>
        <p
          className="text-[16px] leading-[1.75] mb-5 whitespace-pre-wrap"
          style={{ fontFamily: "'Fraunces', serif", fontWeight: 400 }}
        >
          {message.text}
        </p>

        {message.sources && message.sources.length > 0 && (
          <div className="border-t pt-4" style={{ borderColor: C.line }}>
            <button
              onClick={() => setShowSources((v) => !v)}
              className="flex items-center gap-2 text-[10px] tracking-[0.25em] uppercase transition mb-3"
              style={{ color: C.inkSoft }}
            >
              {showSources ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
              <Quote size={12} />
              Sources · {message.sources.length} documents cited
            </button>
            {showSources && (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {message.sources.map((s, i) => (
                  <SourceCard key={s.id || i} source={s} index={i + 1} />
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function SourceCard({ source, index }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div
      className="p-3 transition group border"
      style={{ backgroundColor: C.panelSoft, borderColor: expanded ? C.accent : C.line }}
      onMouseEnter={(e) => { if (!expanded) e.currentTarget.style.borderColor = C.accent; }}
      onMouseLeave={(e) => { if (!expanded) e.currentTarget.style.borderColor = C.line; }}
    >
      <div className="flex items-start gap-2 mb-1">
        <span
          className="text-[10px] px-1.5 py-0.5 rounded shrink-0 mt-0.5"
          style={{ backgroundColor: C.ink, color: C.bg, fontFamily: "'JetBrains Mono', monospace" }}
        >
          [{index}]
        </span>
        <a
          href="#"
          onClick={(e) => { e.preventDefault(); setExpanded((v) => !v); }}
          className="text-sm font-medium leading-snug hover:underline flex-1"
        >
          {source.title}
          <ExternalLink size={10} className="inline ml-1 opacity-50" />
        </a>
      </div>
      <div className="text-xs flex items-center gap-1.5 flex-wrap pl-7" style={{ color: C.inkSoft }}>
        {source.authors && <><span>{source.authors}</span><span>·</span></>}
        {source.year && <><span style={{ fontFamily: "'JetBrains Mono', monospace" }}>{source.year}</span><span>·</span></>}
        {source.type && <span>{source.type}</span>}
      </div>
      {expanded && source.excerpt && (
        <div className="mt-3 pl-7">
          <p
            className="text-xs italic leading-relaxed p-2.5 border-l-2"
            style={{ fontFamily: "'Fraunces', serif", backgroundColor: C.accentSoft, borderColor: C.accent }}
          >
            "{source.excerpt}"
          </p>
          <div className="text-[10px] mt-1.5 flex justify-between" style={{ color: C.inkSoft }}>
            {source.institution && <span>{source.institution}</span>}
            {source.score != null && (
              <span style={{ fontFamily: "'JetBrains Mono', monospace" }}>cosine · {source.score}</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function ThinkingIndicator() {
  return (
    <div className="flex gap-4">
      <div className="shrink-0 mt-1">
        <svg width="28" height="28" viewBox="0 0 88 88" fill="none" className="animate-pulse">
          <ellipse cx="44" cy="44" rx="40" ry="22" stroke={C.ink} strokeWidth="3" fill="none" />
          <circle cx="44" cy="44" r="16" fill={C.ink} />
          <circle cx="44" cy="44" r="7" fill={C.accent} />
        </svg>
      </div>
      <div className="flex-1">
        <div className="text-[10px] tracking-[0.25em] uppercase mb-2" style={{ color: C.inkSoft }}>
          Retrieving sources…
        </div>
        <div className="flex gap-1.5">
          {[0, 1, 2].map((i) => (
            <span
              key={i}
              className="w-2 h-2 rounded-full animate-bounce"
              style={{ backgroundColor: C.accent, animationDelay: `${i * 0.15}s` }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

function Modal({ children, onClose }) {
  return (
    <div
      className="fixed inset-0 z-[200] flex items-center justify-center p-4"
      style={{ backgroundColor: "#1a1d1888", backdropFilter: "blur(4px)" }}
    >
      <div
        className="max-w-lg w-full max-h-[85vh] overflow-y-auto shadow-2xl border"
        style={{ backgroundColor: C.bg, borderColor: C.lineStrong }}
      >
        <div className="flex justify-end p-3">
          <button onClick={onClose} className="p-1 rounded transition hover:bg-[#1a1d1810]">
            <X size={18} />
          </button>
        </div>
        <div className="px-8 pb-8">{children}</div>
      </div>
    </div>
  );
}

function HowItWorks() {
  const steps = [
    { n: "01", title: "Retrieval",    desc: "Your query is embedded and matched against an indexed corpus of scientific documents — papers, reports, slide decks." },
    { n: "02", title: "Augmentation", desc: "The most relevant excerpts are passed into the language model as grounding context, alongside your question." },
    { n: "03", title: "Generation",   desc: "A response is composed strictly from retrieved evidence, with every claim traceable to a cited source." },
  ];
  return (
    <div>
      <span className="text-[10px] tracking-[0.3em] uppercase" style={{ color: C.highlight }}>The pipeline</span>
      <h2 className="text-4xl mt-2 mb-6" style={{ fontFamily: "'Fraunces', serif", fontStyle: "italic", fontWeight: 400 }}>
        How it works
      </h2>
      <div className="space-y-5">
        {steps.map((s) => (
          <div key={s.n} className="flex gap-4">
            <span className="text-3xl shrink-0" style={{ fontFamily: "'Fraunces', serif", color: C.accent }}>{s.n}</span>
            <div>
              <h3 className="font-semibold mb-1">{s.title}</h3>
              <p className="text-sm leading-relaxed" style={{ color: C.inkSoft }}>{s.desc}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function Limitations() {
  const items = [
    "AI may miss nuance or contextual subtlety that a domain expert would catch.",
    "Answers are bounded by the documents available in the indexed corpus.",
    "This tool is not a replacement for peer-reviewed expert review or clinical judgment.",
    "Confidence scores are heuristic and should be interpreted with care.",
  ];
  return (
    <div>
      <span className="text-[10px] tracking-[0.3em] uppercase" style={{ color: C.highlight }}>Honesty</span>
      <h2 className="text-4xl mt-2 mb-6" style={{ fontFamily: "'Fraunces', serif", fontStyle: "italic", fontWeight: 400 }}>
        Limitations
      </h2>
      <ul className="space-y-3">
        {items.map((it, i) => (
          <li key={i} className="flex gap-3 text-sm leading-relaxed">
            <span className="shrink-0" style={{ color: C.accent }}>—</span>
            <span>{it}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function UploadPanel({ authed, pw, setPw, error, onAuth, files, onDrop, dragOver, setDragOver }) {
  if (!authed) {
    return (
      <div>
        <span className="text-[10px] tracking-[0.3em] uppercase" style={{ color: C.highlight }}>Protected</span>
        <h2 className="text-4xl mt-2 mb-2" style={{ fontFamily: "'Fraunces', serif", fontStyle: "italic", fontWeight: 400 }}>
          Add to Database
        </h2>
        <p className="text-sm mb-6" style={{ color: C.inkSoft }}>
          Database submission is access-controlled. Enter the upload password to continue.
        </p>
        <input
          type="password"
          value={pw}
          onChange={(e) => setPw(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && onAuth()}
          placeholder="Upload password"
          className="w-full bg-transparent border-b px-1 py-3 outline-none tracking-widest text-center mb-3"
          style={{ fontFamily: "'JetBrains Mono', monospace", borderColor: C.lineStrong, color: C.ink }}
        />
        {error && (
          <p className="text-xs mb-3 flex items-center justify-center gap-2" style={{ color: C.error }}>
            <AlertCircle size={12} /> Incorrect upload password.
          </p>
        )}
        <button
          onClick={onAuth}
          className="w-full py-3 tracking-[0.25em] text-xs uppercase transition"
          style={{ backgroundColor: C.ink, color: C.bg }}
          onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = C.accent)}
          onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = C.ink)}
        >
          Unlock →
        </button>
      </div>
    );
  }

  return (
    <div>
      <span className="text-[10px] tracking-[0.3em] uppercase" style={{ color: C.highlight }}>Unlocked</span>
      <h2 className="text-4xl mt-2 mb-6" style={{ fontFamily: "'Fraunces', serif", fontStyle: "italic", fontWeight: 400 }}>
        Add to corpus
      </h2>
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        className="border-2 border-dashed rounded-lg p-10 text-center transition"
        style={{ borderColor: dragOver ? C.accent : C.lineStrong, backgroundColor: dragOver ? C.accentSoft : "transparent" }}
      >
        <Upload size={32} className="mx-auto mb-3" style={{ color: C.inkSoft }} />
        <p className="text-sm font-medium mb-1">Drop PDFs, slides, or docs here</p>
        <p className="text-xs" style={{ color: C.inkSoft }}>or click to browse · max 50 MB per file</p>
      </div>
      {files.length > 0 && (
        <div className="mt-4 space-y-1.5">
          <p className="text-[10px] tracking-[0.25em] uppercase mb-2" style={{ color: C.inkSoft }}>
            Queued · {files.length}
          </p>
          {files.map((f, i) => (
            <div key={i} className="flex items-center gap-2 text-sm border px-3 py-2" style={{ borderColor: C.line }}>
              <FileText size={14} style={{ color: C.accent }} />
              <span className="flex-1 truncate">{f.name}</span>
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

function PromptMenu({ categories, activeCategory, setActiveCategory, search, setSearch, onInsert, onClose }) {
  const searchLower = search.trim().toLowerCase();
  const searching = searchLower.length > 0;
  const visiblePrompts = searching
    ? categories.flatMap((c) =>
        c.prompts
          .filter((p) => p.toLowerCase().includes(searchLower))
          .map((p) => ({ text: p, category: c.label }))
      )
    : (categories.find((c) => c.id === activeCategory)?.prompts || []).map((p) => ({ text: p, category: null }));

  const activeDesc = categories.find((c) => c.id === activeCategory)?.desc;

  return (
    <div className="border-t" style={{ borderColor: C.line, backgroundColor: C.panelSoft }}>
      <div className="max-w-3xl mx-auto px-6 pt-5 pb-4">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Wand2 size={14} style={{ color: C.highlight }} />
            <span className="text-[11px] tracking-[0.3em] uppercase" style={{ color: C.inkSoft }}>
              Prompt Library
            </span>
          </div>
          <button onClick={onClose} className="p-1 rounded transition hover:bg-[#1a1d1810]" aria-label="Close prompt library">
            <X size={15} style={{ color: C.inkSoft }} />
          </button>
        </div>

        <div className="relative mb-4">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2" style={{ color: C.inkSoft }} />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search prompts…"
            className="w-full bg-[#ecebe3] pl-9 pr-3 py-2 text-sm outline-none border rounded-lg"
            style={{ borderColor: C.line, color: C.ink }}
            onFocus={(e) => (e.target.style.borderColor = C.accent)}
            onBlur={(e) => (e.target.style.borderColor = C.line)}
          />
        </div>

        {!searching && (
          <>
            <div className="flex flex-wrap gap-1 mb-3">
              {categories.map((c) => {
                const on = c.id === activeCategory;
                return (
                  <button
                    key={c.id}
                    onClick={() => setActiveCategory(c.id)}
                    className="px-3 py-1.5 text-xs border rounded-full transition"
                    style={on
                      ? { backgroundColor: C.accent, color: C.bg, borderColor: C.accent }
                      : { backgroundColor: "transparent", borderColor: C.lineStrong, color: C.ink }
                    }
                    onMouseEnter={(e) => { if (!on) e.currentTarget.style.borderColor = C.accent; }}
                    onMouseLeave={(e) => { if (!on) e.currentTarget.style.borderColor = C.lineStrong; }}
                  >
                    {c.label}
                  </button>
                );
              })}
            </div>
            {activeDesc && (
              <p className="text-xs italic mb-3 pl-1" style={{ color: C.inkSoft, fontFamily: "'Fraunces', serif" }}>
                {activeDesc}
              </p>
            )}
          </>
        )}

        {visiblePrompts.length === 0 ? (
          <div className="text-center text-xs py-8" style={{ color: C.inkSoft }}>
            No prompts match "{search}"
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-h-64 overflow-y-auto pr-1">
            {visiblePrompts.map((p, i) => (
              <PromptCard key={`${p.text}-${i}`} text={p.text} category={p.category} onInsert={() => onInsert(p.text)} />
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

function PromptCard({ text, category, onInsert }) {
  return (
    <div
      className="group relative flex items-start gap-2 p-3 border rounded-lg transition cursor-pointer"
      style={{ backgroundColor: C.bg, borderColor: C.line }}
      onClick={onInsert}
      onMouseEnter={(e) => { e.currentTarget.style.borderColor = C.accent; e.currentTarget.style.boxShadow = "0 2px 8px rgba(47,93,78,0.08)"; }}
      onMouseLeave={(e) => { e.currentTarget.style.borderColor = C.line; e.currentTarget.style.boxShadow = "none"; }}
    >
      <div className="flex-1 min-w-0">
        {category && (
          <div className="text-[9px] tracking-[0.25em] uppercase mb-1" style={{ color: C.highlight }}>
            {category}
          </div>
        )}
        <p className="text-sm leading-snug" style={{ color: C.ink }}>{text}</p>
      </div>
      <div className="opacity-0 group-hover:opacity-100 transition shrink-0 mt-0.5" style={{ color: C.accent }} aria-hidden="true">
        <ChevronRight size={14} />
      </div>
    </div>
  );
}
