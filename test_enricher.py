#!/usr/bin/env python3
"""
Test script for enricher pipeline (without streamlit).
"""
import io
import pandas as pd
import sys

# Import processing logic from app (will skip streamlit bits)
import csv
import io as _io

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


INPUT_NOT_AVAILABLE = "input not available"

OUTPUT_COLUMNS = [
    "Date", "Url", "Domain", "Author", "Likes", "Comments", "Shares", "Full Text",
    "Mentioned Authors", "Thread Author", "Thread Entry Type", "X Author ID", "X Followers",
    "Engagement Score", "Bluesky Author Id", "Category Details",
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
    "Name", "Institution", "NPI", "DOL Yes/No",
    "DOL Profile", "Lilly KOL", "Validated US?", "Continent", "Country", "State",
    "City", "Specialty 1", "Specialty 2",
]

BW_KEEP_COLUMNS = [
    "Date", "Url", "Domain", "Author", "Likes", "Comments", "Shares", "Full Text",
    "Mentioned Authors", "Thread Author", "Thread Entry Type", "X Author ID", "X Followers",
    "Engagement Score", "Bluesky Author Id",
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
    "Category Details",
]


def read_brandwatch_csv(file_path, file_label):
    if file_path is None:
        return None
    with open(file_path, "rb") as f:
        header_row = find_header_row(f)
    df = pd.read_csv(
        file_path,
        skiprows=header_row,
        skip_blank_lines=True,
        dtype={"Date": str, "X Author ID": str, "Bluesky Author Id": str},
        low_memory=False,
    )
    if "Full Text" not in df.columns and "Snippet" in df.columns:
        df = df.rename(columns={"Snippet": "Full Text"})
    missing = [c for c in BW_KEEP_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{file_label} is missing expected columns: {missing}")
    return df[BW_KEEP_COLUMNS].copy()


def process(file1_path, file2_path=None, file3_path=None, file4_path=None):
    x_df = read_brandwatch_csv(file1_path, "Brandwatch X CSV")
    bsky_df = read_brandwatch_csv(file2_path, "Brandwatch Bluesky CSV")
    if x_df is None and bsky_df is None:
        raise ValueError("At least one Brandwatch input file is required.")
    df = x_df.copy() if bsky_df is None else (bsky_df.copy() if x_df is None else pd.concat([x_df, bsky_df], ignore_index=True))

    df["_handle"] = df["Author"].astype(str).str.lower().str.strip()
    df["_domain"] = df["Domain"].astype(str).str.lower().str.strip()

    if file3_path is not None:
        dol = pd.read_csv(file3_path, low_memory=False)
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

    if file4_path is not None:
        meta = pd.read_csv(file4_path, low_memory=False)
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

        is_bluesky = df["_domain"].fillna("").str.lower() == "bsky.app"
        missing_name = df["Name"].isna() | (df["Name"] == "")
        need_bluesky_lookup = is_bluesky & missing_name
        if need_bluesky_lookup.any():
            bsky_lookup = meta.set_index("_bluesky_handle")[ ["Name", "Institution", "NPI", "Validated US?",
                                                               "Continent", "Country", "State", "City",
                                                               "Specialty 1", "Specialty 2"] ]
            df_bsky_handles = df.loc[need_bluesky_lookup, "_handle"].fillna("")
            looked = df_bsky_handles.str.lower().str.strip().map(
                lambda h: bsky_lookup.loc[h].to_dict() if h in bsky_lookup.index else None
            )
            for idx, val in looked.items():
                if val:
                    for k, v in val.items():
                        df.at[idx, k] = v
        df["Validated US?"] = df["Validated US?"].fillna("No")
        is_bluesky = df["_domain"].fillna("").str.lower() == "bsky.app"
        has_name = df["Name"].notna() & (df["Name"] != "")
        unmatched_bluesky = is_bluesky & ~has_name
        df = df[~unmatched_bluesky].copy()
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

    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = INPUT_NOT_AVAILABLE if col in ["DOL Yes/No", "DOL Profile", "Lilly KOL",
                                                    "Name", "Institution", "NPI", "Validated US?",
                                                    "Continent", "Country", "State", "City",
                                                    "Specialty 1", "Specialty 2"] else ""
    return df[OUTPUT_COLUMNS]


def test_category_details_column_is_preserved(tmp_path):
    bw_file = tmp_path / "bw.csv"
    # Create a CSV containing all expected Brandwatch input columns
    header = ",".join(BW_KEEP_COLUMNS)
    # Create a simple row: date + 'val' for most fields, and 'foo' for Category Details if present
    values = []
    for c in BW_KEEP_COLUMNS:
        if c == 'Date':
            values.append('2026-07-17')
        elif c == 'Category Details':
            values.append('foo')
        elif c.startswith('All GLP1 by Category'):
            values.append('1')
        else:
            values.append('val')
    bw_file.write_text(header + "\n" + ",".join(values) + "\n", encoding="utf-8")

    result = process(str(bw_file), None, None, None)

    assert "Category Details" in result.columns
    assert result.iloc[0].get("Category Details") == "foo"


def test_find_header_row_skips_metadata_block(tmp_path):
    bw_file = tmp_path / "bw_with_metadata.csv"
    bw_file.write_text(
        "Report:\n"
        "Bulk Mentions Download\n"
        "Brand:\n"
        "Example Brand\n"
        "Label:\n"
        "\n"
        "Date,Author,Text,Url\n"
        "2026-07-17,alice,Hello world,https://example.com\n",
        encoding="utf-8",
    )

    with open(bw_file, "rb") as fh:
        assert find_header_row(fh) == 6


def test_find_header_row_accepts_headers_without_author(tmp_path):
    bw_file = tmp_path / "bw_without_author.csv"
    bw_file.write_text(
        "Date,Title,Snippet,Url\n"
        "2026-07-17,Example title,Hello world,https://example.com\n",
        encoding="utf-8",
    )

    with open(bw_file, "rb") as fh:
        assert find_header_row(fh) == 0


if __name__ == "__main__":
    bw_file = "/Users/ailab/Downloads/X and Bsky Raw BW 0524-053026 copy.csv"
    dol_file = "/Users/ailab/Downloads/Stuff for Claude/DOL-KOL Lookup Sheet 052726.csv"
    meta_file = "/Users/ailab/Downloads/X Bsky Authors 0524-0530.xlsx - Project Contacts.csv"
    
    try:
        result = process(bw_file, dol_file, meta_file)
        print(f"✓ Success! {len(result):,} rows, {len(result.columns)} columns")
        print("\nFirst 5 rows:")
        print(result.head())
        print("\nColumns:")
        print(result.columns.tolist())
        
        # Save to Excel
        output_file = "/Users/ailab/Downloads/enriched_posts_test.xlsx"
        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            result.to_excel(writer, index=False, sheet_name="Enriched")
        print(f"\n✓ Saved to {output_file}")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
