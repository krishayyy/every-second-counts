"""Turns free-text emergency call descriptions into structured, geocoded calls:
LLM (Groq) extracts location + severity, then OpenStreetMap's free geocoder
resolves each location to real coordinates near the target city."""
import json
import os
import re
import time

import requests

from src.graph_problem import PLEASANTON_CA

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "every-second-counts-hackathon-demo/1.0"

SYSTEM_PROMPT = """You are triaging simultaneous 911 call transcripts for an ambulance
dispatch system. For each call described, extract a short searchable location string
(street, intersection, or landmark) and a severity from 1 (minor) to 5 (life-threatening).
Respond with ONLY a JSON array, no prose, like:
[{"location": "4141 Hacienda Dr, Pleasanton, CA", "severity": 5}]"""


def parse_calls_with_llm(raw_text: str, api_key: str) -> list[dict]:
    resp = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": raw_text},
            ],
            "temperature": 0.2,
        },
        timeout=30,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if not match:
        raise ValueError(f"LLM did not return a JSON array: {content!r}")
    return json.loads(match.group(0))


def geocode(location: str, near: tuple[float, float] = PLEASANTON_CA) -> tuple[float, float] | None:
    lat, lon = near
    delta = 0.15
    viewbox = f"{lon - delta},{lat + delta},{lon + delta},{lat - delta}"
    resp = requests.get(
        NOMINATIM_URL,
        params={"q": location, "format": "json", "limit": 1, "viewbox": viewbox, "bounded": 0},
        headers={"User-Agent": USER_AGENT},
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json()
    if not results:
        return None
    return float(results[0]["lat"]), float(results[0]["lon"])


def intake_calls(raw_text: str, api_key: str, near: tuple[float, float] = PLEASANTON_CA) -> list[dict]:
    extracted = parse_calls_with_llm(raw_text, api_key)
    points = []
    for call in extracted:
        coords = geocode(call["location"], near=near)
        time.sleep(1)  # Nominatim's usage policy caps free requests at ~1/sec
        if coords is None:
            continue
        points.append(
            {
                "lat": coords[0],
                "lon": coords[1],
                "label": call["location"],
                "severity": int(call.get("severity", 3)),
            }
        )
    return points
