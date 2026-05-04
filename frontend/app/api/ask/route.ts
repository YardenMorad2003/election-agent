import { NextResponse } from "next/server";

const apiBase = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";
const upstreamUrl = `${apiBase}/api/ask`;

export async function POST(request: Request) {
  let body: unknown;

  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      { error: "Invalid request body." },
      { status: 400 }
    );
  }

  try {
    const response = await fetch(upstreamUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      cache: "no-store",
    });

    const text = await response.text();
    const contentType = response.headers.get("content-type") || "";
    const isJson = contentType.includes("application/json");

    if (!response.ok) {
      if (isJson) {
        return new NextResponse(text, {
          status: response.status,
          headers: { "Content-Type": "application/json" },
        });
      }

      return NextResponse.json(
        {
          error: `Backend request failed with ${response.status}.`,
          detail: text || null,
        },
        { status: response.status }
      );
    }

    return new NextResponse(text, {
      status: response.status,
      headers: {
        "Content-Type": isJson ? "application/json" : "text/plain; charset=utf-8",
      },
    });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Unknown upstream error";

    return NextResponse.json(
      {
        error:
          "The Python API is unavailable. Start it with `./venv/bin/uvicorn api:app --reload --port 8000`.",
        detail: message,
      },
      { status: 503 }
    );
  }
}
