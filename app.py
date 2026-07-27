import io
import pandas as pd
import streamlit as st

OUTPUT_COLUMNS = [
    "Date", "Url", "Domain", "Author", "Likes", "Comments", "Shares", "Full Text",
    "Mentioned Authors", "Thread Author", "Thread Entry Type", "X Author ID", "X Followers",
    "Engagement Score", "Bluesky Author Id", "Category Details",
    # Category flags from Brandwatch to retain
    "All GLP1 by Category - Drug Class - Dual Agonist",
    "All GLP1 by Category - Drug Class - Oral Tx",
    "All GLP1 by Category - Drug Class - Single Agonist",
    "All GLP1 by Category - Drug Class - Triple Agonist",
    "All GLP1 by Category - Product - Aleniglipron",
    "All GLP1 by Category - Product - Dulaglutide",
    "All GLP1 by Category - Product - Elecoglipron",
    "All GLP1 by Category - Product - Exenatide",
    "All GLP1 by Category - Product - Injectable Semaglutide",
    "All GLP1 by Category - Product - Liraglutide",
    "All GLP1 by Category - Product - Lixisenatide",
    "All GLP1 by Category - Product - Oral Semaglutide",
    "All GLP1 by Category - Product - Orforglipron",
    "All GLP1 by Category - Product - Pemvidutide",
    "All GLP1 by Category - Product - Retatrudide",
    "All GLP1 by Category - Product - Survodutide",
    "All GLP1 by Category - Product - Tirzepatide",
    "All GLP1 by Category - TA - Diabetes",
    "All GLP1 by Category - TA - Obesity",
    # Enrichment / lookup columns
    "Name", "Institution", "NPI", "DOL Yes/No",
    "DOL Profile", "Lilly KOL", "Validated US?", "Continent", "Country", "State",
    "City", "Specialty 1", "Specialty 2",
]

INPUT_NOT_AVAILABLE = "input not available"

# Columns to keep from the incoming Brandwatch CSV
BW_KEEP_COLUMNS = [
    "Date", "Url", "Domain", "Author", "Likes", "Comments", "Shares", "Full Text",
    "Mentioned Authors", "Thread Author", "Thread Entry Type", "X Author ID", "X Followers",
    "Engagement Score", "Bluesky Author Id",
    # Keep Brandwatch category indicator columns so they survive enrichment
    "All GLP1 by Category - Drug Class - Dual Agonist",
    "All GLP1 by Category - Drug Class - Oral Tx",
    "All GLP1 by Category - Drug Class - Single Agonist",
    "All GLP1 by Category - Drug Class - Triple Agonist",
    "All GLP1 by Category - Product - Aleniglipron",
    "All GLP1 by Category - Product - Dulaglutide",
    "All GLP1 by Category - Product - Elecoglipron",
    "All GLP1 by Category - Product - Exenatide",
    "All GLP1 by Category - Product - Injectable Semaglutide",
    "All GLP1 by Category - Product - Liraglutide",
    "All GLP1 by Category - Product - Lixisenatide",
    "All GLP1 by Category - Product - Oral Semaglutide",
    "All GLP1 by Category - Product - Orforglipron",
    "All GLP1 by Category - Product - Pemvidutide",
    "All GLP1 by Category - Product - Retatrudide",
    "All GLP1 by Category - Product - Survodutide",
    "All GLP1 by Category - Product - Tirzepatide",
    "All GLP1 by Category - TA - Diabetes",
    "All GLP1 by Category - TA - Obesity",
    # Also preserve the free-form Category Details column if present
    "Category Details",
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
    cleaned = raw.replace("\x00", "")

    for i, row in enumerate(csv.reader(_io.StringIO(cleaned))):
        lowered = [c.strip().lower().replace("\ufeff", "") for c in row if isinstance(c, str)]
        if not lowered:
            continue

        has_date = any(col in lowered for col in ["date", "created", "created date", "timestamp", "posted"])
        has_content = any(col in lowered for col in ["snippet", "full text", "text", "content", "body", "title", "url", "message", "post"])
        has_author = any(col in lowered for col in ["author", "username", "user name", "screen name", "account", "handle"])

        if has_date and has_content and (has_author or len(row) >= 4):
            return i

    raise ValueError(
        "Could not find the header row in File 1. "
        "Expected a row containing a date field and a content column such as 'Snippet', 'Full Text', 'Text', 'Title', or 'Url'."
    )


def _as_bool_array(mask):
    """Return a NumPy bool array from a pandas Series/array-like mask."""
    if hasattr(mask, "to_numpy"):
        return mask.to_numpy()
    return mask


def read_brandwatch_csv(file, file_label):
    if file is None:
        return None
    header_row = find_header_row(file)
    df = pd.read_csv(
        file,
        skiprows=header_row,
        skip_blank_lines=True,
        dtype={"Date": str, "X Author ID": str, "Bluesky Author Id": str},
        low_memory=False,
    )
    # Support alternate Brandwatch field naming
    if "Full Text" not in df.columns and "Snippet" in df.columns:
        df = df.rename(columns={"Snippet": "Full Text"})

    missing = [c for c in BW_KEEP_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{file_label} is missing expected columns: {missing}")
    return df[BW_KEEP_COLUMNS].copy()


def process(bw_x_file=None, bw_bsky_file=None, lookup_file=None, meta_file=None):
    # --- Brandwatch source files ---
    x_df = read_brandwatch_csv(bw_x_file, "Brandwatch X CSV")
    bsky_df = read_brandwatch_csv(bw_bsky_file, "Brandwatch Bluesky CSV")
    if x_df is None and bsky_df is None:
        raise ValueError("At least one Brandwatch input file is required.")

    if x_df is None:
        df = bsky_df.copy()
    elif bsky_df is None:
        df = x_df.copy()
    else:
        df = pd.concat([x_df, bsky_df], ignore_index=True)

    # normalize author handle for lookups
    df["_handle"] = df["Author"].astype(str).str.lower().str.strip()
    df["_domain"] = df["Domain"].astype(str).str.lower().str.strip()

    # --- DOL/KOL lookup ---
    if lookup_file is not None:
        dol = pd.read_csv(lookup_file, low_memory=False)
        if "Twitter Handle" not in dol.columns:
            raise ValueError("DOL/KOL lookup file must contain a 'Twitter Handle' column.")
        dol["_handle"] = dol["Twitter Handle"].astype(str).str.lower().str.strip()
        dol = dol.rename(columns={"KOL": "Lilly KOL", "DOL": "DOL Yes/No"})
        dol = dol[["_handle", "DOL Yes/No", "DOL Profile", "Lilly KOL"]].drop_duplicates("_handle")

        df = df.merge(dol, on="_handle", how="left")
        df["DOL Yes/No"] = df["DOL Yes/No"].fillna("No")
        df["DOL Profile"] = df["DOL Profile"].fillna("N/A")
        df["Lilly KOL"] = df["Lilly KOL"].fillna("No")
    else:
        df["DOL Yes/No"] = INPUT_NOT_AVAILABLE
        df["DOL Profile"] = INPUT_NOT_AVAILABLE
        df["Lilly KOL"] = INPUT_NOT_AVAILABLE

    # --- Physician metadata ---
    if meta_file is not None:
        meta = pd.read_csv(meta_file, low_memory=False)
        meta_cols = list(meta.columns)
        if "Twitter Handle" in meta_cols:
            meta["_twitter_handle"] = meta["Twitter Handle"].astype(str).str.lower().str.strip()
        else:
            meta["_twitter_handle"] = ""
        if "Bluesky Handle" in meta_cols:
            meta["_bluesky_handle"] = meta["Bluesky Handle"].astype(str).str.lower().str.strip()
        else:
            meta["_bluesky_handle"] = ""
        meta["Name"] = meta.apply(build_name, axis=1)
        meta["Validated US?"] = meta["Country"].apply(
            lambda c: "Yes" if str(c).strip() == "United States" else "No"
        )
        meta = meta.rename(columns={
            "Medical Specialty 1": "Specialty 1",
            "Medical Specialty 2": "Specialty 2",
        })
        meta_select = ["Name", "Institution", "NPI", "Validated US?",
                       "Continent", "Country", "State", "City", "Specialty 1", "Specialty 2",
                       "_twitter_handle", "_bluesky_handle"]
        for c in meta_select:
            if c not in meta.columns:
                meta[c] = ""
        meta = meta[meta_select].drop_duplicates(["_twitter_handle", "_bluesky_handle"])

        df = df.merge(
            meta.drop(columns=["_bluesky_handle"]).rename(columns={"_twitter_handle": "_handle"}),
            on="_handle",
            how="left",
        )

        df = df.reset_index(drop=True)
        is_bluesky = df["_domain"].fillna("").str.lower() == "bsky.app"
        missing_name = df["Name"].isna() | (df["Name"] == "")
        need_bluesky_lookup = is_bluesky & missing_name
        if need_bluesky_lookup.any():
            mask = _as_bool_array(need_bluesky_lookup)
            bsky_lookup = meta.set_index("_bluesky_handle")[ ["Name", "Institution", "NPI", "Validated US?",
                                                               "Continent", "Country", "State", "City",
                                                               "Specialty 1", "Specialty 2"] ]
            df_bsky_handles = df.loc[mask, "_handle"].fillna("")
            looked = df_bsky_handles.astype(str).str.lower().str.strip().map(
                lambda h: bsky_lookup.loc[h].to_dict() if h and h in bsky_lookup.index else None
            )
            for idx, val in looked.items():
                if val:
                    for k, v in val.items():
                        df.at[idx, k] = v

        df["Validated US?"] = df["Validated US?"].fillna("No")
        is_bluesky = df["_domain"].fillna("").str.lower() == "bsky.app"
        has_name = df["Name"].notna() & (df["Name"] != "")
        unmatched_bluesky = is_bluesky & ~has_name
        df = df.loc[~_as_bool_array(unmatched_bluesky)].copy()
    else:
        df["Name"] = INPUT_NOT_AVAILABLE
        df["Institution"] = INPUT_NOT_AVAILABLE
        df["NPI"] = INPUT_NOT_AVAILABLE
        df["Validated US?"] = INPUT_NOT_AVAILABLE
        df["Continent"] = INPUT_NOT_AVAILABLE
        df["Country"] = INPUT_NOT_AVAILABLE
        df["State"] = INPUT_NOT_AVAILABLE
        df["City"] = INPUT_NOT_AVAILABLE
        df["Specialty 1"] = INPUT_NOT_AVAILABLE
        df["Specialty 2"] = INPUT_NOT_AVAILABLE

    # --- Final output ---
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = INPUT_NOT_AVAILABLE if col in ["DOL Yes/No", "DOL Profile", "Lilly KOL",
                                                    "Name", "Institution", "NPI", "Validated US?",
                                                    "Continent", "Country", "State", "City",
                                                    "Specialty 1", "Specialty 2"] else ""

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

st.markdown(
    "Upload at least one Brandwatch file. DOL/KOL Lookup and Physician Metadata files are optional. "
    "Missing lookup or metadata data will be marked as 'input not available'."
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    file_bw_x = st.file_uploader("Brandwatch X CSV", type=["csv"], key="bw_x")
with col2:
    file_bw_bsky = st.file_uploader("Brandwatch Bluesky CSV", type=["csv"], key="bw_bsky")
with col3:
    file_dol = st.file_uploader("DOL/KOL Lookup CSV (optional)", type=["csv"], key="dol")
with col4:
    file_meta = st.file_uploader("Physician Metadata CSV (optional)", type=["csv"], key="meta")

can_process = file_bw_x is not None or file_bw_bsky is not None
if not can_process:
    st.warning("Upload at least one Brandwatch input file to enable processing.")

if st.button("Process", disabled=not can_process):
    with st.spinner("Processing…"):
        try:
            result = process(file_bw_x, file_bw_bsky, file_dol, file_meta)
            st.success(f"Done — {len(result):,} rows, {len(result.columns)} columns.")
            st.dataframe(result.head(20), use_container_width=True)

            csv_bytes = result.to_csv(index=False).encode("utf-8")
            xlsx = to_excel_bytes(result)

            col_download1, col_download2 = st.columns(2)
            with col_download1:
                st.download_button(
                    label="Download enriched CSV",
                    data=csv_bytes,
                    file_name="enriched_posts.csv",
                    mime="text/csv",
                )
            with col_download2:
                st.download_button(
                    label="Download enriched Excel file",
                    data=xlsx,
                    file_name="enriched_posts.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        except Exception as e:
            st.error(f"Error: {e}")
