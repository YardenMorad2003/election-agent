"use client";

import {
  BarChart3,
  Bot,
  ChevronDown,
  Clock3,
  GitCompareArrows,
  Loader2,
  MessageSquarePlus,
  Send,
  Sparkles,
  Trash2,
  UserRound
} from "lucide-react";
import Image from "next/image";
import { FormEvent, Fragment, useEffect, useMemo, useRef, useState } from "react";

type Role = "user" | "assistant";

type ChatMessage = {
  role: Role;
  content: string;
  tools_used?: string[];
  trace?: string[];
  chart_urls?: string[];
  time?: number;
  comparison?: Record<string, ComparisonResult>;
};

type ComparisonResult = {
  answer: string;
  tools_used?: string[];
  trace?: string[];
  chart_urls?: string[];
};

type AskResponse = {
  answer: string;
  tools_used: string[];
  trace: string[];
  chart_urls: string[];
  time: number;
  comparison?: Record<string, ComparisonResult> | null;
  suggestions: string[];
};

const models = ["gpt-4o-mini", "gpt-4o", "gpt-4.1"];

const usExamples = [
  "How did Biden perform in suburban counties in 2020?",
  "Which state had the highest Republican vote share in 2024?",
  "Compare urban vs rural voting trends from 2000 to 2024",
  "Which counties flipped from R to D between 2016 and 2020?"
];

const israelExamples = [
  "How many seats did Likud win in Knesset 25?",
  "List all 3-party coalitions reaching 61 seats in K25",
  "How did right-bloc share change from K14 to K25?",
  "Which locality had the highest turnout in K25?",
  "Who is the current Prime Minister of Israel?",
  "Give me background on the Joint List party from the web"
];

const comparisonLabels: Record<string, string> = {
  single_pass: "Single-Pass LLM",
  rag_only: "RAG-Only",
  fixed_routing: "Fixed Routing",
  dynamic_routing: "Dynamic Routing"
};

function setupNoticeFor(answer: string) {
  if (answer.includes("ChromaDB not built")) {
    return {
      title: "Vector store not installed",
      detail: "RAG-only needs the ChromaDB folder before it can answer.",
      command: "tar -xzf chroma_db.tar.gz"
    };
  }

  if (answer.includes("Fine-tuned model not found")) {
    return {
      title: "Fine-tuned router not installed",
      detail: "Fixed routing can run after the DistilBERT router is expanded.",
      command: "mkdir -p models && tar -xzf distilbert-router.tar.gz -C models/"
    };
  }

  return null;
}

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [model, setModel] = useState(models[0]);
  const [compare, setCompare] = useState(false);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const compactMessages = useMemo(
    () => messages.map(({ role, content }) => ({ role, content })),
    [messages]
  );

  async function ask(question: string) {
    const trimmed = question.trim();
    if (!trimmed || isLoading) return;

    const nextMessages: ChatMessage[] = [
      ...messages,
      { role: "user", content: trimmed }
    ];
    setMessages(nextMessages);
    setInput("");
    setSuggestions([]);
    setError(null);
    setIsLoading(true);

    try {
      const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: [...compactMessages, { role: "user", content: trimmed }],
          model,
          compare
        })
      });

      if (!response.ok) {
        throw new Error(`API request failed with ${response.status}`);
      }

      const data = (await response.json()) as AskResponse;
      setMessages([
        ...nextMessages,
        {
          role: "assistant",
          content: data.answer,
          tools_used: data.tools_used,
          trace: data.trace,
          chart_urls: data.chart_urls,
          time: data.time,
          comparison: data.comparison ?? undefined
        }
      ]);
      setSuggestions(data.suggestions ?? []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
      setMessages(nextMessages);
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void ask(input);
  }

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  }, [messages, isLoading, suggestions]);

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brandMark">
            <BarChart3 size={22} />
          </div>
          <div>
            <h1>Electoral Analyst</h1>
            <p>U.S. and Israeli election intelligence</p>
          </div>
        </div>

        <section className="controlGroup">
          <label htmlFor="model">Model</label>
          <div className="selectWrap">
            <select
              id="model"
              value={model}
              onChange={(event) => setModel(event.target.value)}
            >
              {models.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
            <ChevronDown size={16} />
          </div>
        </section>

        <label className="toggleRow">
          <span>
            <GitCompareArrows size={17} />
            Compare configs
          </span>
          <input
            type="checkbox"
            checked={compare}
            onChange={(event) => setCompare(event.target.checked)}
          />
        </label>

        <ExampleGroup title="U.S. Elections" examples={usExamples} onPick={ask} />
        <ExampleGroup title="Israeli Elections" examples={israelExamples} onPick={ask} />

        <button
          className="clearButton"
          type="button"
          onClick={() => {
            setMessages([]);
            setSuggestions([]);
            setError(null);
          }}
        >
          <Trash2 size={16} />
          Clear chat
        </button>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Agentic Electoral Analyst</p>
            <h2>Ask about elections, coalitions, maps, and trends.</h2>
          </div>
          <div className="statusPill">
            <Sparkles size={15} />
            {compare ? "Comparison mode" : "Dynamic routing"}
          </div>
        </header>

        <div className="chatPane">
          {messages.length === 0 ? (
            <div className="emptyState">
              <MessageSquarePlus size={32} />
              <h3>Start with a precise election question.</h3>
              <p>
                Try a county trend, an Israeli coalition scenario, or ask for a chart.
              </p>
            </div>
          ) : (
            messages.map((message, index) => (
              <MessageBubble key={`${message.role}-${index}`} message={message} />
            ))
          )}

          {isLoading && (
            <div className="message assistant">
              <div className="avatar">
                <Bot size={18} />
              </div>
              <div className="bubble loadingBubble">
                <Loader2 className="spin" size={18} />
                Thinking through the data...
              </div>
            </div>
          )}

          {error && <div className="errorBox">{error}</div>}
          <div ref={bottomRef} className="scrollAnchor" />
        </div>

        {suggestions.length > 0 && (
          <div className="suggestions">
            {suggestions.map((suggestion) => (
              <button key={suggestion} type="button" onClick={() => ask(suggestion)}>
                {suggestion}
              </button>
            ))}
          </div>
        )}

        <form className="composer" onSubmit={submit}>
          <textarea
            ref={inputRef}
            value={input}
            rows={1}
            placeholder="Ask about U.S. or Israeli elections..."
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }}
          />
          <button type="submit" disabled={isLoading || !input.trim()} aria-label="Send">
            <Send size={18} />
          </button>
        </form>
      </section>
    </main>
  );
}

function ExampleGroup({
  title,
  examples,
  onPick
}: {
  title: string;
  examples: string[];
  onPick: (question: string) => Promise<void>;
}) {
  return (
    <section className="exampleGroup">
      <h3>{title}</h3>
      {examples.map((example) => (
        <button key={example} type="button" onClick={() => void onPick(example)}>
          {example}
        </button>
      ))}
    </section>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isAssistant = message.role === "assistant";

  return (
    <div className={`message ${message.role}`}>
      <div className="avatar">
        {isAssistant ? <Bot size={18} /> : <UserRound size={18} />}
      </div>
      <div className={`bubble ${message.comparison ? "comparisonBubble" : ""}`}>
        <RichText text={message.content} />

        {message.chart_urls?.map((url, index) => (
          <Image
            className="chartImage"
            key={`${url}-${index}`}
            src={url}
            alt="Generated chart"
            width={900}
            height={540}
            unoptimized
          />
        ))}

        {message.comparison && <ComparisonGrid comparison={message.comparison} />}

        {isAssistant && (
          <div className="metaRow">
            {message.tools_used?.map((tool, index) => (
              <span className="toolChip" key={`${tool}-${index}`}>
                {tool}
              </span>
            ))}
            {typeof message.time === "number" && (
              <span className="timeChip">
                <Clock3 size={13} />
                {message.time.toFixed(1)}s
              </span>
            )}
          </div>
        )}

        {message.trace && message.trace.length > 0 && (
          <details className="traceBox">
            <summary>Execution trace</summary>
            {message.trace.map((step, index) => (
              <code key={`${step}-${index}`}>{step}</code>
            ))}
          </details>
        )}
      </div>
    </div>
  );
}

function ComparisonGrid({
  comparison
}: {
  comparison: Record<string, ComparisonResult>;
}) {
  return (
    <div className="comparisonGrid">
      {Object.entries(comparison).map(([name, result]) => (
        <article
          key={name}
          className={`comparisonCard ${setupNoticeFor(result.answer) ? "setupCard" : ""}`}
        >
          <h4>{comparisonLabels[name] ?? name}</h4>
          <ResultBody answer={result.answer} />
          {result.chart_urls?.map((url, index) => (
            <Image
              className="chartImage"
              key={`${name}-${url}-${index}`}
              src={url}
              alt="Generated chart"
              width={900}
              height={540}
              unoptimized
            />
          ))}
          {result.tools_used && result.tools_used.length > 0 && (
            <div className="metaRow">
              {result.tools_used.map((tool, index) => (
                <span className="toolChip" key={`${name}-${tool}-${index}`}>
                  {tool}
                </span>
              ))}
            </div>
          )}
          {result.trace && result.trace.length > 0 && (
            <details className="traceBox">
              <summary>Trace</summary>
              {result.trace.map((step, index) => (
                <code key={`${step}-${index}`}>{step}</code>
              ))}
            </details>
          )}
        </article>
      ))}
    </div>
  );
}

function ResultBody({ answer }: { answer: string }) {
  const setupNotice = setupNoticeFor(answer);

  if (setupNotice) {
    return (
      <div className="setupNotice">
        <strong>{setupNotice.title}</strong>
        <span>{setupNotice.detail}</span>
        <code>{setupNotice.command}</code>
      </div>
    );
  }

  return <RichText text={answer} />;
}

function RichText({ text }: { text: string }) {
  const blocks = text.split(/\n{2,}/).map((block) => block.trim()).filter(Boolean);

  return (
    <div className="markdownText">
      {blocks.map((block, index) => (
        <FormattedBlock block={block} key={`${block}-${index}`} />
      ))}
    </div>
  );
}

function FormattedBlock({ block }: { block: string }) {
  const lines = block.split("\n").map((line) => line.trim()).filter(Boolean);
  const isBulleted = lines.length > 0 && lines.every((line) => /^[-*]\s+/.test(line));
  const isNumbered = lines.length > 0 && lines.every((line) => /^\d+\.\s+/.test(line));

  if (isBulleted) {
    return (
      <ul>
        {lines.map((line, index) => (
          <li key={`${line}-${index}`}>
            <InlineText text={line.replace(/^[-*]\s+/, "")} />
          </li>
        ))}
      </ul>
    );
  }

  if (isNumbered) {
    return (
      <ol>
        {lines.map((line, index) => (
          <li key={`${line}-${index}`}>
            <InlineText text={line.replace(/^\d+\.\s+/, "")} />
          </li>
        ))}
      </ol>
    );
  }

  return (
    <p>
      {lines.map((line, index) => (
        <Fragment key={`${line}-${index}`}>
          {index > 0 && <br />}
          <InlineText text={line} />
        </Fragment>
      ))}
    </p>
  );
}

function InlineText({ text }: { text: string }) {
  const parts = text.split(/(\[[^\]]+\]\((?:https?:\/\/|\/)[^)]+\)|https?:\/\/[^\s)]+|\*\*[^*]+\*\*)/g);

  return (
    <>
      {parts.map((part, index) => {
        const markdownLink = part.match(/^\[([^\]]+)\]\(((?:https?:\/\/|\/)[^)]+)\)$/);
        if (markdownLink) {
          return (
            <a
              href={markdownLink[2]}
              key={`${part}-${index}`}
              rel="noreferrer"
              target={markdownLink[2].startsWith("http") ? "_blank" : undefined}
            >
              {markdownLink[1]}
            </a>
          );
        }

        if (/^https?:\/\//.test(part)) {
          const href = part.replace(/[.,;:!?]+$/, "");
          const trailing = part.slice(href.length);
          return (
            <Fragment key={`${part}-${index}`}>
              <a href={href} rel="noreferrer" target="_blank">
                {href}
              </a>
              {trailing}
            </Fragment>
          );
        }

        if (part.startsWith("**") && part.endsWith("**")) {
          return <strong key={`${part}-${index}`}>{part.slice(2, -2)}</strong>;
        }

        return <Fragment key={`${part}-${index}`}>{part}</Fragment>;
      })}
    </>
  );
}
