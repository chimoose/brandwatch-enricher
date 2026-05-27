import io
import pandas as pd
import streamlit as st

OUTPUT_COLUMNS = [
    "Date", "Snippet", "Url", "Author", "Expanded URLs", "Mentioned Authors",
    "Thread Author", "Thread Entry Type", "Twitter Author ID", "Twitter Followers",
    "Twitter Reply Count", "Twitter Reply to", "Twitter Retweet of", "Twitter Retweets",
    "Twitter Likes", "Engagement Score", "Name", "Institution", "NPI", "DOL Yes/No",
    "DOL Profile", "Lilly KOL", "Validated US?", "Continent", "Country", "State",
    "City", "Specialty 1", "Specialty 2",
]

BW_KEEP_COLUMNS = [
    "Date", "Snippet", "Url", "Author", "Expanded URLs", "Mentioned Authors",
    "Thread Author", "Thread Entry Type", "Twitter Author ID", "Twitter Followers",
    "Twitter Reply Count", "Twitter Reply to", "Twitter Retweet of", "Twitter Retweets",
    "Twitter Likes", "Engagement Score",
]


def clean(val) -> str:
    """Return stripped string, treating NaN/None/float-nan as empty."""
    if val is None:
        return ""
    s = str(val).strip()
    return "" if s.lower() == "nan" else s


def build_name(row):
    parts = [clean(row.get("First Name")),
             clean(row.get("Middle Name")),
             clean(row.get("Last Name"))]
    name = " ".join(p for p in parts if p)
    if not name:
        name = clean(row.get("Twitter Name"))
    return name


def find_header_row(file) -> int:
    """Return the 0-based row index that contains the Brandwatch column headers."""
    import csv, io as _io
    raw = file.read().decode("utf-8", errors="replace")
    file.seek(0)
    for i, row in enumerate(csv.reader(_io.StringIO(raw))):
        if "Date" in row and "Author" in row and "Snippet" in row:
            return i
    raise ValueError(
        "Could not find the header row in File 1. "
        "Expected a row containing 'Date', 'Author', and 'Snippet'."
    )


def process(file1, file2, file3):
    # --- File 1: Brandwatch ---
    header_row = find_header_row(file1)
    df = pd.read_csv(file1, skiprows=header_row, dtype={"Twitter Author ID": str}, low_memory=False)
    missing = [c for c in BW_KEEP_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"File 1 is missing expected columns: {missing}")
    df = df[BW_KEEP_COLUMNS].copy()
    df["_handle"] = df["Author"].str.lower().str.strip()

    # --- File 2: DOL/KOL lookup ---
    dol = pd.read_csv(file2, low_memory=False)
    dol["_handle"] = dol["Twitter Handle"].str.lower().str.strip()
    dol = dol.rename(columns={"KOL": "Lilly KOL", "DOL": "DOL Yes/No"})
    dol = dol[["_handle", "DOL Yes/No", "DOL Profile", "Lilly KOL"]].drop_duplicates("_handle")

    df = df.merge(dol, on="_handle", how="left")
    df["DOL Yes/No"] = df["DOL Yes/No"].fillna("No")
    df["DOL Profile"] = df["DOL Profile"].fillna("N/A")
    df["Lilly KOL"] = df["Lilly KOL"].fillna("No")

    # --- File 3: Physician metadata ---
    meta = pd.read_csv(file3, low_memory=False)
    meta["_handle"] = meta["Twitter Handle"].str.lower().str.strip()
    meta["Name"] = meta.apply(build_name, axis=1)
    meta["Validated US?"] = meta["Country"].apply(
        lambda c: "Yes" if str(c).strip() == "United States" else "No"
    )
    meta = meta.rename(columns={
        "Medical Specialty 1": "Specialty 1",
        "Medical Specialty 2": "Specialty 2",
    })
    meta_cols = ["_handle", "Name", "Institution", "NPI", "Validated US?",
                 "Continent", "Country", "State", "City", "Specialty 1", "Specialty 2"]
    meta = meta[meta_cols].drop_duplicates("_handle")

    df = df.merge(meta, on="_handle", how="left")

    # Validated US? defaults to No for unmatched rows
    df["Validated US?"] = df["Validated US?"].fillna("No")

    # --- Final output ---
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    return df[OUTPUT_COLUMNS]


def to_excel_bytes(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Enriched")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Brandwatch Enricher", layout="centered")
st.title("Brandwatch Post Enricher")

st.markdown("Upload the three weekly files, then click **Process**.")

col1, col2, col3 = st.columns(3)
with col1:
    file1 = st.file_uploader("File 1 — Brandwatch CSV", type=["csv"], key="bw")
with col2:
    file2 = st.file_uploader("File 2 — DOL/KOL Lookup CSV", type=["csv"], key="dol")
with col3:
    file3 = st.file_uploader("File 3 — Physician Metadata CSV", type=["csv"], key="meta")

if st.button("Process", disabled=not (file1 and file2 and file3)):
    with st.spinner("Processing…"):
        try:
            result = process(file1, file2, file3)
            st.success(f"Done — {len(result):,} rows, {len(result.columns)} columns.")
            st.dataframe(result.head(20), use_container_width=True)
            xlsx = to_excel_bytes(result)
            st.download_button(
                label="Download enriched Excel file",
                data=xlsx,
                file_name="enriched_posts.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as e:
            st.error(f"Error: {e}")
