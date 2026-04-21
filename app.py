from __future__ import annotations

import html
import re
import random
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit.errors import StreamlitAPIException

from src.config import dataset_paths, dataset_source, huggingface_dataset_counts, huggingface_dataset_enabled
from src.io import count_dataset_items, read_document_cached
from src.run import run_matching

try:
    st.set_page_config(page_title="CN6000 Matcher", page_icon="📊", layout="wide")
except StreamlitAPIException:
    pass

MAX_UI_ROWS = 5000
DEFAULT_SUBSET_JOBS = 30
BASE_SUBSET_SEED = 42
USE_SUBSET_MODE = True
USE_EMBEDDING_CACHE = True
GENERATE_EXPLANATIONS = True


# UI theme + accessibility styles for Streamlit components.
def _inject_styles(_dark_mode: bool) -> None:
    theme_vars = """
        :root {
          --bg: radial-gradient(circle at 10% 10%, #f7fbff 0%, #edf7f3 42%, #fff7eb 100%);
          --bg-2: linear-gradient(135deg, rgba(14, 165, 164, 0.10), rgba(217, 119, 6, 0.08));
          --ink: #0f172a;
          --muted: #334155;
          --accent: #0f766e;
          --accent-2: #d97706;
          --card: rgba(255,255,255,0.86);
          --card-strong: rgba(255,255,255,0.92);
          --hero-surface: linear-gradient(120deg, rgba(255,255,255,0.72) 0%, rgba(255,255,255,0.28) 100%);
          --panel-surface: rgba(255,255,255,0.08);
          --input-surface: rgba(255,255,255,0.08);
          --chip-surface: rgba(255,255,255,0.10);
          --border: #dbe7f3;
          --border-strong: #c7d7e8;
          --surface: rgba(255,255,255,0.62);
          --header: rgba(255,255,255,0.88);
          --sidebar: rgba(248,250,252,0.92);
          --dataframe-text: #0f172a;
          --empty-border: #bfd3e6;
          --tab-muted: #475569;
          --tab-active: #0f172a;
          --panel-text: #0f172a;
          --table-head: #ffffff;
          --table-cell: #ffffff;
          --table-border: #dbe7f3;
          --table-bar: #0ea5a4;
          --table-bar-2: #14b8a6;
          --table-bar-bg: rgba(14,165,164,0.10);
        }
        :root[data-theme="dark"] {
          --bg: radial-gradient(circle at 10% 10%, #07111a 0%, #0c1726 34%, #150f1f 100%);
          --bg-2: radial-gradient(circle at 12% 14%, rgba(8, 145, 178, 0.18), transparent 28%), radial-gradient(circle at 86% 16%, rgba(217, 119, 6, 0.14), transparent 24%);
          --ink: #e5eef7;
          --muted: #93a8bb;
          --accent: #22c1b8;
          --accent-2: #ff8a00;
          --card: rgba(11, 20, 31, 0.82);
          --card-strong: rgba(9, 17, 27, 0.92);
          --hero-surface: linear-gradient(135deg, rgba(9, 18, 29, 0.96) 0%, rgba(14, 25, 39, 0.88) 56%, rgba(31, 21, 30, 0.92) 100%);
          --panel-surface: rgba(255,255,255,0.045);
          --input-surface: rgba(255,255,255,0.06);
          --chip-surface: rgba(255,255,255,0.065);
          --border: rgba(148, 163, 184, 0.16);
          --border-strong: rgba(148, 163, 184, 0.24);
          --surface: rgba(255,255,255,0.04);
          --header: rgba(8, 15, 25, 0.78);
          --sidebar: rgba(6, 12, 21, 0.96);
          --dataframe-text: #e5eef7;
          --empty-border: rgba(159, 179, 200, 0.44);
          --tab-muted: rgba(229, 238, 247, 0.78);
          --tab-active: #f8fbff;
          --panel-text: #f3f7fb;
          --table-head: #070b10;
          --table-cell: #05080d;
          --table-border: rgba(110, 231, 183, 0.22);
          --table-bar: #22c55e;
          --table-bar-2: #4ade80;
          --table-bar-bg: rgba(74, 222, 128, 0.10);
        }
        """
    styles = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Source+Sans+3:wght@400;600;700&display=swap');
        __THEME_VARS__
        @keyframes drift {
          0% { transform: translate3d(0, 0, 0) scale(1); }
          50% { transform: translate3d(0, -14px, 0) scale(1.03); }
          100% { transform: translate3d(0, 0, 0) scale(1); }
        }
        @keyframes glowPulse {
          0% { box-shadow: 0 18px 40px rgba(15, 23, 42, 0.12); }
          50% { box-shadow: 0 26px 56px rgba(15, 23, 42, 0.18); }
          100% { box-shadow: 0 18px 40px rgba(15, 23, 42, 0.12); }
        }
        .stApp { background: var(--bg); color: var(--ink); font-family: "Source Sans 3", "Segoe UI", sans-serif; }
        ::selection { background: rgba(34, 193, 184, 0.28); color: var(--ink); }
        [data-testid="stAppViewContainer"] { background: var(--bg) !important; }
        [data-testid="stAppViewContainer"]::before {
          content: "";
          position: fixed;
          inset: 0;
          background: var(--bg-2);
          pointer-events: none;
          z-index: 0;
        }
        .main > div {
          position: relative;
          z-index: 1;
        }
        [data-testid="stHeader"] { background: var(--header) !important; border-bottom: 1px solid var(--border); backdrop-filter: blur(14px); }
        [data-testid="stToolbar"] { right: 0.5rem; }
        :root[data-theme="dark"] [data-testid="stToolbar"] button,
        :root[data-theme="dark"] [data-testid="collapsedControl"],
        :root[data-theme="dark"] [data-testid="collapsedControl"] button,
        :root[data-theme="dark"] [data-testid="baseButton-headerNoPadding"] {
          background: rgba(248, 251, 255, 0.95) !important;
          color: #08111b !important;
          border: 1px solid rgba(255,255,255,0.78) !important;
          border-radius: 12px !important;
          box-shadow: 0 10px 24px rgba(2, 6, 23, 0.30);
        }
        :root[data-theme="dark"] [data-testid="stToolbar"] button:hover,
        :root[data-theme="dark"] [data-testid="collapsedControl"]:hover,
        :root[data-theme="dark"] [data-testid="collapsedControl"] button:hover,
        :root[data-theme="dark"] [data-testid="baseButton-headerNoPadding"]:hover {
          background: #ffffff !important;
          border-color: rgba(255,255,255,0.95) !important;
        }
        :root[data-theme="dark"] [data-testid="stToolbar"] button svg,
        :root[data-theme="dark"] [data-testid="collapsedControl"] svg,
        :root[data-theme="dark"] [data-testid="collapsedControl"] button svg,
        :root[data-theme="dark"] [data-testid="baseButton-headerNoPadding"] svg {
          fill: #08111b !important;
          color: #08111b !important;
        }
        [data-testid="stSidebar"] { background: var(--sidebar) !important; border-right: 1px solid var(--border); backdrop-filter: blur(18px); }
        [data-testid="stSidebar"] > div:first-child { background: var(--sidebar) !important; }
        .stApp, .stMarkdown, .stText, .stCaption, .st-emotion-cache-10trblm, .st-emotion-cache-q8sbsg { color: var(--ink) !important; }
        [data-testid="stSidebar"], [data-testid="stSidebar"] * { color: var(--ink) !important; fill: var(--ink) !important; }
        [data-testid="stSidebar"] [data-testid="stNumberInput"] input,
        [data-testid="stSidebar"] [data-testid="stNumberInput"] button,
        [data-testid="stSidebar"] [data-testid="stNumberInputContainer"] input,
        [data-testid="stSidebar"] [data-testid="stNumberInputContainer"] button {
          background: var(--input-surface) !important;
          color: var(--ink) !important;
          border-color: var(--border-strong) !important;
        }
        [data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div,
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="base-input"] {
          background: var(--input-surface) !important;
          color: var(--ink) !important;
          border-color: var(--border-strong) !important;
        }
        [data-baseweb="popover"] [role="listbox"],
        [data-baseweb="popover"] [role="option"] {
          background: var(--card-strong) !important;
          color: var(--ink) !important;
        }
        [data-testid="stSidebar"] input[type="number"] {
          background: var(--input-surface) !important;
          color: var(--ink) !important;
          caret-color: var(--ink) !important;
        }
        [data-testid="stSidebar"] [data-baseweb="slider"] div { color: var(--ink) !important; }
        [data-testid="stSidebar"] [data-baseweb="select"], [data-testid="stSidebar"] [data-baseweb="input"] {
          background: var(--input-surface) !important;
          color: var(--ink) !important;
        }
        [data-testid="stCheckbox"] label, [data-testid="stRadio"] label { color: var(--ink) !important; }
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: var(--ink) !important; }
        [data-testid="stExpander"] details {
          background: var(--card) !important;
          border: 1px solid var(--border) !important;
          border-radius: 14px;
          transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
          box-shadow: 0 14px 32px rgba(15, 23, 42, 0.08);
        }
        [data-testid="stExpander"] details:hover {
          transform: translateY(-2px) perspective(1200px) rotateX(1deg);
          box-shadow: 0 18px 40px rgba(15, 23, 42, 0.14);
          border-color: rgba(34, 193, 184, 0.35) !important;
        }
        [data-testid="stInfo"], [data-testid="stSuccess"], [data-testid="stWarning"], [data-testid="stError"] { color: var(--ink) !important; }
        :root[data-theme="dark"] [data-testid="stInfo"] {
          background:
            radial-gradient(circle at 12% 18%, rgba(34, 193, 184, 0.34), transparent 34%),
            linear-gradient(90deg, rgba(34, 193, 184, 0.28), rgba(255, 138, 0, 0.26)) !important;
          border: 1px solid rgba(255, 138, 0, 0.52) !important;
          color: #f3f7fb !important;
          box-shadow:
            0 14px 30px rgba(2, 6, 23, 0.18),
            0 0 0 1px rgba(34, 193, 184, 0.10) inset;
        }
        :root[data-theme="dark"] [data-testid="stInfo"] * {
          color: #f3f7fb !important;
          fill: #ffb45c !important;
        }
        [data-testid="stDataFrame"] {
          background: var(--table-cell) !important;
          border: 1px solid var(--table-border) !important;
          border-radius: 18px !important;
          overflow: hidden !important;
          box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
        }
        [data-testid="stDataFrame"] * { color: var(--dataframe-text) !important; }
        [data-testid="stDataFrame"] div[role="columnheader"] {
          background: var(--table-head) !important;
          color: var(--panel-text) !important;
          border-color: var(--table-border) !important;
        }
        [data-testid="stDataFrame"] div[role="gridcell"] {
          background: var(--table-cell) !important;
          color: var(--panel-text) !important;
          border-color: var(--table-border) !important;
        }
        [data-testid="stDataFrame"] canvas,
        [data-testid="stDataFrame"] [data-testid="stElementToolbar"],
        [data-testid="stDataFrame"] [data-testid="stDataFrameResizable"] {
          background: var(--table-cell) !important;
        }
        [data-testid="stDataFrame"] [data-testid="stProgressBar"] > div {
          background: var(--table-bar-bg) !important;
        }
        [data-testid="stDataFrame"] [data-testid="stProgressBar"] > div > div {
          background: linear-gradient(90deg, var(--table-bar), var(--table-bar-2)) !important;
        }
        [data-baseweb="tab-list"] { gap: 8px; }
        button[kind="tab"] {
          border-radius: 999px !important;
          padding: 0.35rem 0.9rem !important;
          color: var(--tab-active) !important;
          background: transparent !important;
          border: 1px solid transparent !important;
          transition: background 180ms ease, transform 180ms ease;
        }
        button[kind="tab"]:hover { transform: translateY(-1px); }
        button[kind="tab"][aria-selected="true"] {
          color: var(--tab-active) !important;
          background: rgba(255,255,255,0.08) !important;
          border: 1px solid var(--border-strong) !important;
        }
        [data-baseweb="tab-list"] button,
        [data-baseweb="tab-list"] button * {
          color: var(--tab-active) !important;
          -webkit-text-fill-color: var(--tab-active) !important;
        }
        h1, h2, h3 { color: var(--ink); letter-spacing: 0.2px; font-family: "Space Grotesk", "Segoe UI", sans-serif; }
        .hero {
          border: 1px solid var(--border);
          border-radius: 22px;
          padding: 22px 24px;
          background:
            radial-gradient(circle at top right, rgba(217, 119, 6, 0.18), transparent 30%),
            radial-gradient(circle at left center, rgba(34, 193, 184, 0.12), transparent 36%),
            var(--hero-surface);
          box-shadow: 0 24px 60px rgba(15, 23, 42, 0.24);
          margin-bottom: 14px;
          overflow: hidden;
          position: relative;
          animation: glowPulse 7s ease-in-out infinite;
          backdrop-filter: blur(20px);
        }
        .hero::after {
          content: "";
          position: absolute;
          width: 240px;
          height: 240px;
          right: -60px;
          top: -80px;
          border-radius: 50%;
          background: radial-gradient(circle, rgba(255,138,0,0.25), rgba(255,138,0,0.0) 68%);
          animation: drift 9s ease-in-out infinite;
          pointer-events: none;
        }
        .chip {
          display:inline-block;
          padding: 6px 12px;
          border-radius: 999px;
          border: 1px solid var(--border);
          background: var(--chip-surface);
          margin-right: 6px;
          font-size: 12px;
          color: var(--ink);
          backdrop-filter: blur(14px);
        }
        .hero-grid {
          display: grid;
          grid-template-columns: minmax(0, 1.7fr) minmax(260px, 1fr);
          gap: 18px;
          align-items: end;
        }
        .hero-kicker {
          color: var(--accent);
          font-size: 12px;
          text-transform: uppercase;
          letter-spacing: 0.8px;
          margin-bottom: 10px;
          font-weight: 700;
        }
        .hero-title {
          margin: 0;
          font-size: 42px;
          line-height: 0.98;
          max-width: 10ch;
        }
        .hero-copy {
          margin: 12px 0 0 0;
          color: var(--muted);
          font-size: 18px;
          max-width: 52ch;
          line-height: 1.45;
        }
        .hero-panel {
          background: var(--panel-surface);
          border: 1px solid var(--border-strong);
          border-radius: 18px;
          padding: 14px 16px;
          backdrop-filter: blur(14px);
          box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
        }
        .hero-panel-label {
          font-size: 11px;
          text-transform: uppercase;
          letter-spacing: 0.6px;
          color: var(--muted);
          margin-bottom: 8px;
        }
        .timing-card {
          border: 1px solid var(--border-strong);
          border-radius: 20px;
          overflow: hidden;
          background: var(--card-strong);
          box-shadow: 0 18px 40px rgba(15, 23, 42, 0.16);
          margin-top: 8px;
        }
        .timing-head, .timing-row {
          display: grid;
          grid-template-columns: 1.2fr 0.8fr;
        }
        .timing-head {
          background: rgba(255,255,255,0.04);
          color: var(--muted);
          font-size: 13px;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          font-weight: 700;
        }
        .timing-head div, .timing-row div {
          padding: 14px 16px;
          border-bottom: 1px solid var(--border);
        }
        .timing-row div:last-child, .timing-head div:last-child {
          border-left: 1px solid var(--border);
          text-align: right;
          font-variant-numeric: tabular-nums;
        }
        .timing-row:last-child div {
          border-bottom: none;
        }
        .timing-row.total {
          background: rgba(34, 193, 184, 0.06);
          font-weight: 700;
        }
        .theme-fab {
          position: fixed;
          right: 18px;
          top: 76px;
          z-index: 9999;
          display: inline-flex;
          align-items: center;
          gap: 8px;
          padding: 10px 14px;
          border-radius: 999px;
          border: 1px solid var(--border-strong);
          background: var(--card-strong);
          color: var(--ink);
          box-shadow: 0 12px 30px rgba(15, 23, 42, 0.16);
          backdrop-filter: blur(16px);
          font: 600 14px "Source Sans 3", "Segoe UI", sans-serif;
          cursor: pointer;
        }
        .theme-fab:hover {
          transform: translateY(-1px);
        }
        .hero-stats {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 12px;
        }
        .hero-stat {
          border-left: 3px solid var(--accent);
          padding-left: 10px;
        }
        .hero-stat-value {
          font-family: "Space Grotesk", "Segoe UI", sans-serif;
          font-weight: 700;
          font-size: 24px;
          line-height: 1.1;
        }
        .hero-stat-label {
          color: var(--muted);
          font-size: 12px;
          margin-top: 2px;
        }
        .snapshot {
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: 14px;
          padding: 12px 14px;
          margin-bottom: 10px;
        }
        .snapshot-label {
          color: var(--muted);
          font-size: 12px;
          text-transform: uppercase;
          letter-spacing: 0.4px;
        }
        .snapshot-value {
          color: var(--ink);
          font-size: 24px;
          font-family: "Space Grotesk", "Segoe UI", sans-serif;
          font-weight: 700;
          line-height: 1.2;
        }
        .stButton > button {
          background: linear-gradient(90deg, var(--accent) 0%, var(--accent-2) 100%);
          border: none;
          color: white;
          font-weight: 700;
          border-radius: 14px;
          box-shadow: 0 8px 18px rgba(15, 118, 110, 0.22);
          transition: transform 180ms ease, box-shadow 180ms ease, filter 180ms ease;
          min-height: 2.9rem;
        }
        .stButton > button:hover {
          transform: translateY(-2px) perspective(1200px) rotateX(2deg);
          box-shadow: 0 16px 36px rgba(15, 118, 110, 0.32);
          filter: saturate(1.08);
        }
        .stDownloadButton > button {
          background:
            radial-gradient(circle at 18% 20%, rgba(255,255,255,0.16), transparent 28%),
            linear-gradient(90deg, var(--accent) 0%, var(--accent-2) 100%) !important;
          color: #f8fbff !important;
          border: 1px solid rgba(255,255,255,0.18) !important;
          font-weight: 700 !important;
          border-radius: 14px !important;
          min-height: 2.9rem !important;
          padding: 0.72rem 1.15rem !important;
          box-shadow:
            0 8px 18px rgba(15, 118, 110, 0.22),
            0 0 0 1px rgba(255,255,255,0.04) inset !important;
          transition:
            transform 180ms ease,
            box-shadow 180ms ease,
            filter 180ms ease,
            border-color 180ms ease !important;
        }
        .stDownloadButton > button:hover {
          transform: translateY(-2px) perspective(1200px) rotateX(2deg);
          filter: saturate(1.08);
          border-color: rgba(255,255,255,0.26) !important;
          box-shadow:
            0 16px 36px rgba(15, 118, 110, 0.32),
            0 0 0 1px rgba(255,255,255,0.08) inset !important;
        }
        .stDownloadButton > button:focus,
        .stDownloadButton > button:focus-visible {
          outline: none !important;
          border-color: rgba(255,255,255,0.28) !important;
          box-shadow:
            0 0 0 3px rgba(34, 193, 184, 0.20),
            0 16px 38px rgba(15, 118, 110, 0.22) !important;
        }
        .stCaption, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: var(--muted) !important; }
        .run-mode-banner {
          background: #e8f3ff;
          color: #0b3a66;
          border: 1px solid #c9dff5;
          border-radius: 10px;
          padding: 10px 12px;
          margin: 8px 0 12px 0;
          font-weight: 600;
        }
        .tips {
          background: #f0f9ff;
          border: 1px solid #cde8fb;
          color: #0b3a66;
          border-radius: 12px;
          padding: 10px 12px;
          margin: 8px 0 10px 0;
        }
        .guide-card {
          background: var(--card);
          border: 1px solid var(--border);
          border-radius: 18px;
          padding: 16px 18px;
          margin: 10px 0 14px 0;
          backdrop-filter: blur(16px);
          box-shadow: 0 18px 42px rgba(15, 23, 42, 0.10);
        }
        .guide-title {
          margin: 0 0 6px 0;
          font-family: "Space Grotesk", "Segoe UI", sans-serif;
          font-size: 22px;
          color: var(--ink);
        }
        .guide-copy {
          margin: 0 0 12px 0;
          color: var(--muted);
          line-height: 1.5;
        }
        .guide-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
          gap: 12px;
        }
        .guide-step {
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: 14px;
          padding: 12px 14px;
        }
        .guide-step-no {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          width: 28px;
          height: 28px;
          border-radius: 999px;
          background: linear-gradient(90deg, var(--accent), var(--accent-2));
          color: #f8fbff;
          font-weight: 700;
          margin-bottom: 8px;
        }
        .guide-step-title {
          font-weight: 700;
          color: var(--ink);
          margin-bottom: 4px;
        }
        .guide-step-copy {
          color: var(--muted);
          line-height: 1.45;
          font-size: 15px;
        }
        .result-shell {
          background: var(--card);
          border: 1px solid var(--border);
          border-radius: 18px;
          padding: 16px 18px;
          margin-bottom: 14px;
          backdrop-filter: blur(16px);
          box-shadow: 0 20px 48px rgba(15, 23, 42, 0.10);
        }
        .result-heading {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 12px;
          margin-bottom: 8px;
        }
        .result-title {
          margin: 0;
          font-family: "Space Grotesk", "Segoe UI", sans-serif;
          font-size: 28px;
        }
        .eyebrow {
          color: var(--muted);
          font-size: 12px;
          text-transform: uppercase;
          letter-spacing: 0.6px;
          margin-bottom: 4px;
        }
        .insight-strip {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
          margin-top: 10px;
        }
        .insight-pill {
          display: inline-block;
          padding: 5px 10px;
          border-radius: 999px;
          background: var(--panel-surface);
          border: 1px solid var(--border-strong);
          font-size: 12px;
          color: var(--ink);
          backdrop-filter: blur(10px);
        }
        .match-summary {
          border: 1px solid var(--border);
          background: linear-gradient(180deg, rgba(255,255,255,0.10), rgba(255,255,255,0.04));
          border-radius: 14px;
          padding: 12px 14px;
          margin-bottom: 10px;
          backdrop-filter: blur(12px);
          color: var(--panel-text) !important;
        }
        .match-summary,
        .match-summary * {
          color: var(--panel-text) !important;
          -webkit-text-fill-color: var(--panel-text) !important;
        }
        .match-meta {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
          margin: 8px 0 10px 0;
        }
        .match-badge {
          display: inline-block;
          padding: 4px 9px;
          border-radius: 999px;
          border: 1px solid var(--border-strong);
          background: var(--panel-surface);
          font-size: 12px;
          color: var(--ink);
        }
        [data-testid="stTextArea"] textarea,
        [data-testid="stTextArea"] textarea:hover,
        [data-testid="stTextArea"] textarea:focus {
          cursor: text !important;
          color: var(--panel-text) !important;
          -webkit-text-fill-color: var(--panel-text) !important;
          background: var(--card-strong) !important;
          border: 1px solid var(--border-strong) !important;
          opacity: 1 !important;
        }
        [data-testid="stTextArea"] textarea:disabled,
        [data-testid="stTextArea"] textarea[disabled],
        [data-testid="stTextArea"] [disabled] {
          color: var(--panel-text) !important;
          -webkit-text-fill-color: var(--panel-text) !important;
          opacity: 1 !important;
          background: var(--card-strong) !important;
        }
        [data-testid="stTextArea"] div[data-baseweb="textarea"] {
          background: var(--card-strong) !important;
          border: 1px solid var(--border-strong) !important;
        }
        [data-testid="stTextArea"] pre,
        [data-testid="stTextArea"] p,
        [data-testid="stTextArea"] span {
          color: var(--panel-text) !important;
          -webkit-text-fill-color: var(--panel-text) !important;
        }
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] *,
        [data-testid="stTabs"] *,
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary * {
          color: var(--panel-text) !important;
          -webkit-text-fill-color: var(--panel-text) !important;
        }
        [data-testid="stTextArea"] label,
        [data-testid="stTextArea"] * {
          cursor: default !important;
        }
        .loading-card {
          border: 1px solid var(--border);
          background: var(--card);
          border-radius: 18px;
          padding: 14px 16px;
          margin: 10px 0 16px 0;
          backdrop-filter: blur(16px);
          box-shadow: 0 18px 40px rgba(15, 23, 42, 0.10);
        }
        .loading-track {
          width: 100%;
          height: 12px;
          border-radius: 999px;
          background: rgba(255,255,255,0.12);
          overflow: hidden;
          border: 1px solid var(--border);
        }
        .loading-bar {
          height: 100%;
          border-radius: 999px;
          background: linear-gradient(90deg, var(--accent) 0%, var(--accent-2) 100%);
          box-shadow: 0 0 24px rgba(34, 193, 184, 0.45);
          transition: width 320ms ease;
        }
        .empty-state {
          border: 1px dashed var(--empty-border);
          border-radius: 18px;
          padding: 24px 22px;
          background: var(--card);
          text-align: center;
          color: var(--muted);
          margin-top: 18px;
          backdrop-filter: blur(16px);
        }
        @media (max-width: 900px) {
          .hero-grid {
            grid-template-columns: 1fr;
          }
          .hero-title {
            font-size: 28px;
            max-width: none;
          }
          .result-heading {
            display: block;
          }
        }
        </style>
        """
    st.markdown(styles.replace("__THEME_VARS__", theme_vars), unsafe_allow_html=True)


def _read_theme_from_query_params() -> bool:
    theme_value = str(st.query_params.get("theme", "light")).strip().lower()
    return theme_value == "dark"


def _sync_theme_query_param(dark_mode: bool) -> None:
    target_theme = "dark" if dark_mode else "light"
    if st.query_params.get("theme") != target_theme:
        st.query_params["theme"] = target_theme


def _candidate_label(candidate_id: str) -> str:
    return re.sub(r"^upload_cand_(\d+)$", lambda m: f"Candidate {int(m.group(1))}", candidate_id)


@st.cache_data(show_spinner=False)
def _current_dataset_limits() -> tuple[int, int]:
    if huggingface_dataset_enabled():
        cv_count, jobs_count = huggingface_dataset_counts()
        return max(1, cv_count), max(1, jobs_count)
    jobs_dir, candidates_dir = dataset_paths()
    jobs_count = count_dataset_items(jobs_dir, r"^jd_\d+$")
    cv_count = count_dataset_items(candidates_dir, r"^(cand_\d+)_cv$", unique_group=1)
    return max(1, cv_count), max(1, jobs_count)


def _show_timings(metrics: dict) -> None:
    timings = metrics.get("timings") or {}
    if not timings:
        return
    st.caption("Run timings")
    timing_rows = [
        ("Load", timings.get("load_seconds", 0), ""),
        ("Filter", timings.get("filter_seconds", 0), ""),
        ("Embed", timings.get("embed_seconds", 0), ""),
        ("Match", timings.get("match_seconds", 0), ""),
        ("Explain", timings.get("explain_seconds", 0), ""),
        ("Total", timings.get("total_seconds", 0), " total"),
    ]
    rows_html = "".join(
        f"<div class='timing-row{extra}'><div>{stage}</div><div>{seconds}</div></div>"
        for stage, seconds, extra in timing_rows
    )
    st.markdown(
        f"""
        <div class='timing-card'>
          <div class='timing-head'><div>Stage</div><div>Seconds</div></div>
          {rows_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _resolve_job_path(job_id: str) -> Path | None:
    jobs_dir_value = st.session_state.get("active_jobs_dir")
    if jobs_dir_value:
        jobs_dir = Path(jobs_dir_value)
    else:
        jobs_dir, _ = dataset_paths()
    for ext in (".txt", ".docx", ".pdf"):
        matches = sorted(jobs_dir.rglob(f"{job_id}{ext}"))
        if matches:
            return matches[0]
    return None


def _resolve_candidate_cv_path(candidate_id: str) -> Path | None:
    candidates_dir_value = st.session_state.get("active_candidates_dir")
    if candidates_dir_value:
        candidates_dir = Path(candidates_dir_value)
    else:
        _, candidates_dir = dataset_paths()
    for ext in (".txt", ".docx", ".pdf"):
        matches = sorted(candidates_dir.rglob(f"{candidate_id}_cv{ext}"))
        if matches:
            return matches[0]
    return None


def _read_doc_text(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return read_document_cached(path).strip()
    except Exception:
        return ""


def _snapshot_card(label: str, value: str) -> None:
    st.markdown(
        f"<div class='snapshot'><div class='snapshot-label'>{label}</div><div class='snapshot-value'>{value}</div></div>",
        unsafe_allow_html=True,
    )


def _render_loading_state(progress_value: int, message: str) -> None:
    st.markdown(
        f"""
        <div class='loading-card'>
          <div class='eyebrow'>Matcher pipeline</div>
          <div style='display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:8px;'>
            <div style='font-weight:700;'>{message}</div>
            <div style='font-family:"Space Grotesk","Segoe UI",sans-serif;font-size:24px;'>{progress_value}%</div>
          </div>
          <div class='loading-track'>
            <div class='loading-bar' style='width:{progress_value}%;'></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _mount_theme_controller() -> None:
    components.html(
        """
        <script>
        (function () {
          const key = "cn6000-theme";
          const parentDoc = window.parent.document;
          const root = parentDoc.documentElement;

          function currentTheme() {
            return root.getAttribute("data-theme") === "dark" ? "dark" : "light";
          }

          function applyTheme(theme) {
            root.setAttribute("data-theme", theme);
            try { window.parent.localStorage.setItem(key, theme); } catch (e) {}
            const btn = parentDoc.getElementById("cn6000-theme-fab");
            if (btn) {
              btn.innerHTML = theme === "dark" ? "☀ Light mode" : "☾ Dark mode";
            }
          }

          function savedTheme() {
            try { return window.parent.localStorage.getItem(key) || "light"; } catch (e) { return "light"; }
          }

          function mount() {
            let btn = parentDoc.getElementById("cn6000-theme-fab");
            if (!btn) {
              btn = parentDoc.createElement("button");
              btn.id = "cn6000-theme-fab";
              btn.className = "theme-fab";
              btn.type = "button";
              btn.onclick = function () {
                applyTheme(currentTheme() === "dark" ? "light" : "dark");
              };
              parentDoc.body.appendChild(btn);
            }
            applyTheme(savedTheme());
          }

          mount();
        })();
        </script>
        """,
        height=0,
    )


def _build_export_df_with_explanations(df: pd.DataFrame) -> pd.DataFrame:
    export_df = df.copy()
    if "explanation" not in export_df.columns:
        export_df["explanation"] = ""
    explanation_text = export_df["explanation"].fillna("").astype(str).str.strip()
    missing_mask = explanation_text.eq("")
    if missing_mask.any():
        sim_series = pd.to_numeric(export_df.get("similarity_score", 0.0), errors="coerce").fillna(0.0)
        rank_series = export_df.get("candidate_rank", pd.Series(["-"] * len(export_df), index=export_df.index)).astype(str)
        job_title_series = export_df.get("job_title", pd.Series(["this role"] * len(export_df), index=export_df.index)).fillna("this role").astype(str)
        strength = pd.Series("partial alignment", index=export_df.index)
        strength = strength.mask(sim_series >= 0.55, "moderate alignment")
        strength = strength.mask(sim_series >= 0.75, "strong alignment")
        generated = (
            "Candidate shows "
            + strength
            + " for "
            + job_title_series
            + " (similarity="
            + sim_series.round(3).astype(str)
            + ", rank="
            + rank_series
            + ")."
        )
        export_df.loc[missing_mask, "explanation"] = generated.loc[missing_mask]
    return export_df


def _render_rankings_dataframe(df: pd.DataFrame) -> None:
    """Render a sortable table that follows the active frontend theme."""
    display_columns = [
        ("__index__", "", "number"),
        ("job_id", "Job ID", "text"),
        ("job_title", "Job Title", "text"),
        ("candidate_id", "Candidate", "text"),
        ("candidate_rank", "Rank", "number"),
        ("similarity_score", "Similarity", "number"),
    ]
    rows_html: list[str] = []
    for index_value, row in df.iterrows():
        cells = [
            f"<td data-sort-value='{html.escape(str(index_value))}'>{html.escape(str(index_value))}</td>"
        ]
        for column_key, _, column_type in display_columns[1:]:
            raw_value = row.get(column_key, "")
            if column_key == "similarity_score":
                numeric_value = pd.to_numeric(pd.Series([raw_value]), errors="coerce").fillna(0.0).iloc[0]
                width_pct = max(0.0, min(100.0, float(numeric_value) * 100.0))
                cells.append(
                    f'''
                    <td class="sim-cell" data-sort-value="{float(numeric_value):.6f}">
                      <div class="sim-wrap">
                        <div class="sim-track"><div class="sim-bar" style="width:{width_pct:.1f}%"></div></div>
                        <span class="sim-value">{float(numeric_value):.3f}</span>
                      </div>
                    </td>
                    '''
                )
            elif column_type == "number":
                numeric_value = pd.to_numeric(pd.Series([raw_value]), errors="coerce").fillna(0).iloc[0]
                cells.append(
                    f"<td data-sort-value='{float(numeric_value):.6f}'>{int(numeric_value)}</td>"
                )
            else:
                safe_value = html.escape(str(raw_value))
                cells.append(f"<td data-sort-value='{safe_value.lower()}'>{safe_value}</td>")
        rows_html.append("<tr>" + "".join(cells) + "</tr>")
    header_html = "".join(
        f'''
        <th data-col-index="{index}" data-sort-type="{sort_type}">
          <button type="button" class="sort-btn">
            <span>{html.escape(label)}</span>
            <span class="sort-indicator">↕</span>
          </button>
        </th>
        '''
        for index, (_, label, sort_type) in enumerate(display_columns)
    )
    components.html(
            f'''
            <style>
              :root {{
                --frame-bg: #ffffff;
                --frame-text: #0f172a;
                --frame-muted: #475569;
                --frame-border: #dbe7f3;
                --frame-row-hover: #f7fafc;
                --frame-track: rgba(14,165,164,0.10);
                --frame-track-border: rgba(14,165,164,0.22);
                --frame-bar: linear-gradient(90deg, #0ea5a4 0%, #14b8a6 100%);
              }}
              body[data-theme="dark"] {{
                --frame-bg: #05080d;
                --frame-text: #f8fbff;
                --frame-muted: #cfead8;
                --frame-border: rgba(74, 222, 128, 0.24);
                --frame-row-hover: #08110f;
                --frame-track: rgba(74, 222, 128, 0.12);
                --frame-track-border: rgba(134, 239, 172, 0.32);
                --frame-bar: linear-gradient(90deg, #22c55e 0%, #4ade80 100%);
              }}
              body {{
                margin: 0;
                background: var(--frame-bg);
                color: var(--frame-text);
                font-family: "Source Sans 3", "Segoe UI", sans-serif;
              }}
              .rankings-shell {{
                border: 1px solid var(--frame-border);
                border-radius: 18px;
                overflow: auto;
                background: var(--frame-bg);
              }}
              table {{
                width: 100%;
                border-collapse: collapse;
                table-layout: fixed;
                background: var(--frame-bg);
              }}
              thead th {{
                position: sticky;
                top: 0;
                z-index: 2;
                background: var(--frame-bg);
                border: 1px solid var(--frame-border);
                padding: 0;
                color: var(--frame-text);
                font-size: 13px;
                font-weight: 700;
              }}
              tbody td {{
                border: 1px solid var(--frame-border);
                padding: 12px 14px;
                color: var(--frame-text);
                background: var(--frame-bg);
                font-size: 15px;
                vertical-align: middle;
                word-wrap: break-word;
              }}
              tbody tr:hover td {{
                background: var(--frame-row-hover);
              }}
              th:nth-child(1), td:nth-child(1) {{
                width: 58px;
                text-align: right;
                color: var(--frame-muted);
              }}
              th:nth-child(2), td:nth-child(2) {{ width: 140px; }}
              th:nth-child(3), td:nth-child(3) {{ width: 360px; }}
              th:nth-child(4), td:nth-child(4) {{ width: 170px; }}
              th:nth-child(5), td:nth-child(5) {{ width: 120px; text-align: right; }}
              th:nth-child(6), td:nth-child(6) {{ width: 170px; }}
              .sort-btn {{
                width: 100%;
                border: 0;
                background: transparent;
                color: inherit;
                padding: 12px 14px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 8px;
                font: inherit;
                cursor: pointer;
              }}
              .sort-btn:hover {{
                background: color-mix(in srgb, var(--frame-text) 6%, transparent);
              }}
              .sort-indicator {{
                color: var(--frame-text);
                opacity: 0.8;
              }}
              th.sorted-asc .sort-indicator::after {{
                content: " ↑";
              }}
              th.sorted-desc .sort-indicator::after {{
                content: " ↓";
              }}
              th.sorted-asc .sort-indicator,
              th.sorted-desc .sort-indicator {{
                opacity: 1;
              }}
              .sim-wrap {{
                display: grid;
                grid-template-columns: 1fr auto;
                gap: 10px;
                align-items: center;
              }}
              .sim-track {{
                height: 12px;
                background: var(--frame-track);
                border: 1px solid var(--frame-track-border);
                border-radius: 999px;
                overflow: hidden;
              }}
              .sim-bar {{
                height: 100%;
                background: var(--frame-bar);
              }}
              .sim-value {{
                min-width: 44px;
                text-align: right;
                font-variant-numeric: tabular-nums;
              }}
            </style>
            <div class="rankings-shell">
              <table id="rankings-table">
                <thead>
                  <tr>{header_html}</tr>
                </thead>
                <tbody>{''.join(rows_html)}</tbody>
              </table>
            </div>
            <script>
              const table = document.getElementById("rankings-table");
              const tbody = table.querySelector("tbody");
              const headers = Array.from(table.querySelectorAll("th"));
              let sortState = {{ index: 5, direction: "desc" }};

              function getCellValue(row, index, type) {{
                const value = row.children[index].dataset.sortValue || "";
                return type === "number" ? Number(value) : value.toLowerCase();
              }}

              function applySort(index, type) {{
                sortState.direction = sortState.index === index && sortState.direction === "asc" ? "desc" : "asc";
                sortState.index = index;
                const rows = Array.from(tbody.querySelectorAll("tr"));
                rows.sort((a, b) => {{
                  const av = getCellValue(a, index, type);
                  const bv = getCellValue(b, index, type);
                  if (av < bv) return sortState.direction === "asc" ? -1 : 1;
                  if (av > bv) return sortState.direction === "asc" ? 1 : -1;
                  return 0;
                }});
                tbody.innerHTML = "";
                rows.forEach((row) => tbody.appendChild(row));
                headers.forEach((header, headerIndex) => {{
                  header.classList.toggle("sorted-asc", headerIndex === index && sortState.direction === "asc");
                  header.classList.toggle("sorted-desc", headerIndex === index && sortState.direction === "desc");
                }});
              }}

              headers.forEach((header, index) => {{
                header.querySelector(".sort-btn").addEventListener("click", () => {{
                  applySort(index, header.dataset.sortType || "text");
                }});
              }});
              function syncTheme() {{
                const parentTheme = window.parent.document.documentElement.getAttribute("data-theme") || "light";
                document.body.setAttribute("data-theme", parentTheme);
              }}
              syncTheme();
              setInterval(syncTheme, 400);
              applySort(5, "number");
            </script>
            ''',
            height=560,
            scrolling=True,
    )


# Render ranking table, metrics, and detailed feedback.
def _show_results(df: pd.DataFrame, metrics: dict) -> None:
    display_df = df.copy()
    display_df = display_df.rename(columns={"rank": "candidate_rank", "similarity": "similarity_score"})
    if "candidate_id" in display_df.columns:
        display_df["candidate_id_raw"] = display_df["candidate_id"]
        display_df["candidate_id"] = display_df["candidate_id"].map(_candidate_label)

    filtered_df = display_df.copy()

    ordered_cols = [
        c
        for c in ["job_id", "job_title", "candidate_id", "candidate_rank", "similarity_score"]
        if c in display_df.columns
    ]
    ui_df = filtered_df[ordered_cols]
    if len(ui_df) > MAX_UI_ROWS:
        st.warning(
            f"Showing first {MAX_UI_ROWS:,} rows out of {len(ui_df):,} to keep the UI responsive."
        )
        ui_df = ui_df.head(MAX_UI_ROWS)
    total_runtime = metrics.get("timings", {}).get("total_seconds", 0)
    top_k_value = str(int(filtered_df["candidate_rank"].max()) if "candidate_rank" in filtered_df.columns else "-")
    st.markdown(
        f"""
        <div class='result-shell'>
          <div class='result-heading'>
            <h2 class='result-title'>Ranked Results</h2>
            <div class='insight-strip'>
              <span class='insight-pill'>{metrics.get('jobs_loaded', 0)} jobs</span>
              <span class='insight-pill'>{metrics.get('candidates_ranked', 0)} candidates</span>
              <span class='insight-pill'>Top-{top_k_value}</span>
              <span class='insight-pill'>{total_runtime}s runtime</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    overview_tab, rankings_tab, feedback_tab = st.tabs(["Overview", "Rankings", "Feedback"])
    with overview_tab:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Jobs", str(metrics.get("jobs_loaded", 0)))
        c2.metric("Candidates", str(metrics.get("candidates_ranked", 0)))
        c3.metric("Rows", str(len(filtered_df)))
        c4.metric("Runtime", f"{total_runtime}s")
        c5.metric("Top-K", top_k_value)
        _show_timings(metrics)

    with rankings_tab:
        if "similarity_score" in ui_df.columns:
            ui_df = ui_df.sort_values(["similarity_score", "candidate_rank"], ascending=[False, True])
        _render_rankings_dataframe(ui_df)
        export_df = _build_export_df_with_explanations(filtered_df)
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download rankings.csv", data=csv_bytes, file_name="rankings.csv", mime="text/csv")

    with feedback_tab:
        feedback_df = filtered_df.copy()
        if "similarity_score" in feedback_df.columns:
            feedback_df = feedback_df.sort_values("similarity_score", ascending=False)
        for row in feedback_df.head(20).to_dict(orient="records"):
            rank_value = row.get("candidate_rank", "-")
            title = f"{row.get('job_title', '')} ({row.get('job_id', '')}) -> {row.get('candidate_id', '')} [rank {rank_value}]"
            with st.expander(title):
                explanation = str(row.get("explanation", "")).strip()
                similarity_value = row.get("similarity_score", "-")
                similarity_label = f"{float(similarity_value):.3f}" if similarity_value not in ("-", None, "") else "-"
                st.markdown(
                    f"""
                    <div class='match-summary'>
                      <div class='eyebrow'>Match snapshot</div>
                      <div><strong>{row.get('job_title', '')}</strong> paired with <strong>{row.get('candidate_id', '')}</strong></div>
                      <div class='match-meta'>
                        <span class='match-badge'>Rank {rank_value}</span>
                        <span class='match-badge'>Similarity {similarity_label}</span>
                        <span class='match-badge'>{row.get('job_id', '')}</span>
                      </div>
                      <div>{explanation if explanation else "No explanation available for this match."}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                raw_candidate_id = str(row.get("candidate_id_raw", "")).strip()
                job_id = str(row.get("job_id", "")).strip()
                cv_path = _resolve_candidate_cv_path(raw_candidate_id)
                job_path = _resolve_job_path(job_id)
                cv_text = _read_doc_text(cv_path)
                job_text = _read_doc_text(job_path)
                cv_tab, job_tab = st.tabs(["CV", "Job Description"])
                with cv_tab:
                    if cv_text:
                        st.text_area(
                            "CV content",
                            value=cv_text,
                            height=260,
                            key=f"cv_{job_id}_{raw_candidate_id}",
                            disabled=True,
                            label_visibility="collapsed",
                        )
                    else:
                        st.write("No readable CV content found.")
                with job_tab:
                    if job_text:
                        st.text_area(
                            "Job description content",
                            value=job_text,
                            height=260,
                            key=f"job_{job_id}_{raw_candidate_id}",
                            disabled=True,
                            label_visibility="collapsed",
                        )
                    else:
                        st.write("No readable job description content found.")


# Dataset-mode runner: loads prepared corpora and executes shared matcher pipeline.
def _run_dataset_mode(
    top_k: int,
    output_explanations: bool,
    explanation_top_n_jobs: int,
    use_cache: bool,
    max_jobs: int | None,
    max_candidates: int | None,
    job_sample_seed: int,
    candidate_sample_seed: int,
    write_outputs: bool,
) -> tuple[pd.DataFrame, dict]:
    loading_shell = st.empty()
    status_placeholder = st.empty()

    def update_progress(percent: int, label: str) -> None:
        with loading_shell.container():
            _render_loading_state(percent, label)
        status_placeholder.info(f"{label} ({percent}%)")

    with st.status("Running matcher", expanded=True) as status:
        update_progress(4, "Booting matcher")
        if max_jobs or max_candidates:
            status.write(
                f"Loading batch: jobs={max_jobs or 'all'}, candidates={max_candidates or 'all'}..."
            )
        else:
            status.write("Loading full dataset...")
        df, metrics = run_matching(
            mode="dataset",
            candidate_doc_mode="cv_only",
            top_k=top_k,
            write_text_report=False,
            use_cache=use_cache,
            clear_cache=False,
            required_skills=None,
            location=None,
            salary_min=None,
            salary_max=None,
            allow_model_download=True,
            refresh_online_data=False,
            refresh_max_candidates=500,
            refresh_max_jobs=100,
            refresh_seed=42,
            refresh_force=False,
            output_explanations=output_explanations,
            explanation_top_n_jobs=explanation_top_n_jobs,
            write_outputs=write_outputs,
            use_faiss=False,
            max_jobs=max_jobs,
            max_candidates=max_candidates,
            sample_seed=job_sample_seed,
            job_sample_seed=job_sample_seed,
            candidate_sample_seed=candidate_sample_seed,
            progress_callback=update_progress,
        )
        timings = metrics.get("timings", {})
        status.write(
            " | ".join(
                [
                    f"Load {timings.get('load_seconds', 0)}s",
                    f"Embed {timings.get('embed_seconds', 0)}s",
                    f"Match {timings.get('match_seconds', 0)}s",
                    f"Explain {timings.get('explain_seconds', 0)}s",
                ]
            )
        )
        status.update(
            label=f"Run completed in {timings.get('total_seconds', 0)}s",
            state="complete",
            expanded=False,
        )
    loading_shell.empty()
    status_placeholder.empty()
    return df, metrics


# Streamlit entrypoint and sidebar controls.
def main() -> None:
    _inject_styles(False)
    _mount_theme_controller()

    dataset_max_candidates, dataset_max_jobs = _current_dataset_limits()
    current_dataset_source = dataset_source()
    dataset_chip = {
        "local": "Local dataset",
        "huggingface": "Hugging Face dataset",
        "demo": "Demo dataset",
    }.get(current_dataset_source, "Dataset")
    st.markdown(
        f"""
        <div class='hero'>
          <div class='hero-grid'>
            <div>
              <div class='hero-kicker'>Semantic matching</div>
              <h1 class='hero-title'>CN6000 Matcher</h1>
              <p class='hero-copy'>Similarity-led CV-to-job matching with ranked candidates, clear feedback, and export-ready results.</p>
              <div style='margin-top:10px'>
                <span class='chip'>Semantic ranking</span>
                <span class='chip'>Explainable feedback</span>
                <span class='chip'>Batch exploration</span>
                <span class='chip'>{dataset_chip}</span>
              </div>
            </div>
            <div class='hero-panel'>
              <div class='hero-panel-label'>Dataset</div>
              <div class='hero-stats'>
                <div class='hero-stat'>
                  <div class='hero-stat-value'>{dataset_max_jobs:,}</div>
                  <div class='hero-stat-label'>Job descriptions</div>
                </div>
                <div class='hero-stat'>
                  <div class='hero-stat-value'>{dataset_max_candidates:,}</div>
                  <div class='hero-stat-label'>Candidate CVs</div>
                </div>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if current_dataset_source == "huggingface":
        st.caption("Deployment dataset is loading from Hugging Face.")
    elif current_dataset_source == "demo":
        st.caption("Fallback demo dataset is active because no larger dataset source was found.")

    st.info("Use the sidebar to customize batch settings, refresh mode, and ranking options before you run the matcher.")
    with st.expander("Quick Start & User Guide", expanded=True):
        st.markdown(
            """
            <div class='guide-card'>
              <h3 class='guide-title'>How To Use The Matcher</h3>
              <p class='guide-copy'>This app compares candidate CVs with job descriptions and ranks the strongest matches. If you are opening it for the first time, follow these steps:</p>
              <div class='guide-grid'>
                <div class='guide-step'>
                  <div class='guide-step-no'>1</div>
                  <div class='guide-step-title'>Choose settings in the sidebar</div>
                  <div class='guide-step-copy'>Use the left sidebar to set Top-K, jobs per run, candidates per run, and batch refresh mode once it unlocks.</div>
                </div>
                <div class='guide-step'>
                  <div class='guide-step-no'>2</div>
                  <div class='guide-step-title'>Click Run matcher</div>
                  <div class='guide-step-copy'>Press the main Run matcher button to generate ranked results for the current batch.</div>
                </div>
                <div class='guide-step'>
                  <div class='guide-step-no'>3</div>
                  <div class='guide-step-title'>Read the three result views</div>
                  <div class='guide-step-copy'>Overview shows totals and timing, Rankings shows the ranked table, and Feedback explains why matches were made.</div>
                </div>
                <div class='guide-step'>
                  <div class='guide-step-no'>4</div>
                  <div class='guide-step-title'>Use Next batch for another sample</div>
                  <div class='guide-step-copy'>After the first run, Batch refresh mode lets you change jobs, candidates, or both when you load the next batch.</div>
                </div>
              </div>
              <p class='guide-copy' style='margin-top:12px;'>If you ever see a sleep page before the app loads, simply wake the app and wait a moment. Streamlit Community Cloud puts inactive apps to sleep automatically.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if "job_batch_seed" not in st.session_state:
        st.session_state["job_batch_seed"] = random.SystemRandom().randint(1, 999999)
    if "candidate_batch_seed" not in st.session_state:
        st.session_state["candidate_batch_seed"] = random.SystemRandom().randint(1, 999999)

    has_previous_run = "last_df" in st.session_state and "last_metrics" in st.session_state
    st.sidebar.header("Run settings")
    use_cache = USE_EMBEDDING_CACHE
    output_explanations = GENERATE_EXPLANATIONS
    subset_mode = USE_SUBSET_MODE
    next_batch_clicked = False
    top_k = st.sidebar.slider("Top-K jobs per candidate", min_value=1, max_value=50, value=3)
    if subset_mode:
        if has_previous_run:
            refresh_mode = st.sidebar.selectbox(
                "Batch refresh mode",
                options=[
                    "Randomize both",
                    "Keep jobs, change candidates",
                    "Keep candidates, change jobs",
                ],
            )
        else:
            refresh_mode = "Randomize both"
            st.sidebar.caption("Batch refresh mode unlocks after the first run.")

        ratio = st.sidebar.selectbox(
            "Candidates per job ratio",
            options=[5, 3, 2],
            index=0,
            format_func=lambda x: f"{x}x",
        )
        subset_jobs = int(
            st.sidebar.number_input(
                "Jobs per run",
                min_value=1,
                max_value=dataset_max_jobs,
                value=min(DEFAULT_SUBSET_JOBS, dataset_max_jobs),
                step=1,
            )
        )
        min_candidates = min(dataset_max_candidates, subset_jobs * ratio)
        max_candidates_allowed = dataset_max_candidates
        default_candidates = min(dataset_max_candidates, subset_jobs * ratio)
        if subset_jobs * ratio > dataset_max_candidates:
            st.sidebar.caption(
                f"Available candidates ({dataset_max_candidates}) are below {ratio}x for {subset_jobs} jobs."
            )
        subset_candidates = int(
            st.sidebar.number_input(
                "Candidates per run",
                min_value=max(1, min_candidates),
                max_value=max_candidates_allowed,
                value=max(1, default_candidates),
                step=1,
            )
        )
        max_jobs = subset_jobs
        max_candidates = subset_candidates
        write_outputs = False
    else:
        refresh_mode = "Randomize both"
        max_jobs = None
        max_candidates = None
        write_outputs = True

    next_batch_clicked = st.sidebar.button("Next batch", disabled=not has_previous_run)
    run_requested = st.button("Run matcher", type="primary", use_container_width=True)

    if subset_mode:
        if next_batch_clicked:
            if refresh_mode == "Randomize both":
                st.session_state["job_batch_seed"] = random.SystemRandom().randint(1, 999999)
                st.session_state["candidate_batch_seed"] = random.SystemRandom().randint(1, 999999)
            elif refresh_mode == "Keep jobs, change candidates":
                st.session_state["candidate_batch_seed"] = random.SystemRandom().randint(1, 999999)
            else:
                st.session_state["job_batch_seed"] = random.SystemRandom().randint(1, 999999)
            st.session_state["batch_notice"] = "Next batch loaded. Running matcher for this batch..."
        active_job_seed = int(st.session_state["job_batch_seed"])
        active_candidate_seed = int(st.session_state["candidate_batch_seed"])
        if "batch_notice" in st.session_state:
            st.sidebar.success(st.session_state.pop("batch_notice"))
    else:
        active_job_seed = BASE_SUBSET_SEED
        active_candidate_seed = BASE_SUBSET_SEED

    explanation_top_n_jobs = 0
    run_requested = run_requested or next_batch_clicked
    if run_requested:
        try:
            df, metrics = _run_dataset_mode(
                top_k=top_k,
                output_explanations=output_explanations,
                explanation_top_n_jobs=explanation_top_n_jobs,
                use_cache=use_cache,
                max_jobs=max_jobs,
                max_candidates=max_candidates,
                job_sample_seed=active_job_seed,
                candidate_sample_seed=active_candidate_seed,
                write_outputs=write_outputs,
            )
            st.session_state["last_df"] = df
            st.session_state["last_metrics"] = metrics
            dataset_paths_info = metrics.get("dataset_paths") or {}
            st.session_state["active_jobs_dir"] = dataset_paths_info.get("jobs_dir")
            st.session_state["active_candidates_dir"] = dataset_paths_info.get("candidates_dir")
            if not has_previous_run:
                st.rerun()
        except Exception as exc:
            st.error(f"Run failed: {exc}")

    if "last_df" in st.session_state and "last_metrics" in st.session_state:
        _show_results(st.session_state["last_df"], st.session_state["last_metrics"])
    else:
        st.markdown(
            """
            <div class='empty-state'>
              <div class='eyebrow'>Ready to run</div>
              <h3 style='margin:0 0 8px 0;'>No rankings loaded yet</h3>
              <div>Run the matcher to generate ranked candidates and feedback.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
