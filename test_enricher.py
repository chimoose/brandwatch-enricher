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
    "City", "Specialty 1", "Specialty 2", "Classification", "Job Role", "Ecosystem Type",
]

CATEGORY_COLUMNS = [
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
]

BASE_BW_KEEP_COLUMNS = [
    "Date", "Url", "Domain", "Author", "Likes", "Comments", "Shares", "Full Text",
    "Mentioned Authors", "Thread Author", "Thread Entry Type", "X Author ID", "X Followers",
    "Engagement Score", "Bluesky Author Id",
]
BW_KEEP_COLUMNS = BASE_BW_KEEP_COLUMNS + CATEGORY_COLUMNS + ["Category Details"]


def _as_bool_array(mask):
    if hasattr(mask, "to_numpy"):
        return mask.to_numpy()
    return mask


def read_brandwatch_csv(file_path, file_label):
    if file_path is None:
        return None
    if isinstance(file_path, str):
        with open(file_path, "rb") as f:
            header_row = find_header_row(f)
        df = pd.read_csv(
            file_path,
            skiprows=header_row,
            skip_blank_lines=True,
            dtype={"Date": str, "X Author ID": str, "Bluesky Author Id": str},
            low_memory=False,
        )
    else:
        if hasattr(file_path, "seek"):
            file_path.seek(0)
        header_row = find_header_row(file_path)
        file_path.seek(0)
        df = pd.read_csv(
            file_path,
            skiprows=header_row,
            skip_blank_lines=True,
            dtype={"Date": str, "X Author ID": str, "Bluesky Author Id": str},
            low_memory=False,
        )
    if "Full Text" not in df.columns and "Snippet" in df.columns:
        df = df.rename(columns={"Snippet": "Full Text"})
    missing = [c for c in BASE_BW_KEEP_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{file_label} is missing expected columns: {missing}")
    selected = [c for c in BW_KEEP_COLUMNS if c in df.columns]
    return df[selected].copy()


def _looks_like_brandwatch_csv(file_obj):
    if file_obj is None:
        return False
    try:
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
        df = read_brandwatch_csv(file_obj, "Brandwatch CSV")
        return df is not None and not df.empty
    except Exception:
        return False


def process(*args, **kwargs):
    bw_file = kwargs.get("bw_file")
    bw_bsky_file = kwargs.get("bw_bsky_file")
    lookup_file = kwargs.get("lookup_file")
    meta_file = kwargs.get("meta_file")

    if args:
        if len(args) == 1:
            bw_file = args[0]
        elif len(args) == 2:
            if _looks_like_brandwatch_csv(args[1]):
                bw_bsky_file = args[1]
            else:
                lookup_file = args[1]
        elif len(args) == 3:
            bw_file = args[0]
            if _looks_like_brandwatch_csv(args[1]):
                bw_bsky_file = args[1]
                lookup_file = args[2]
            else:
                lookup_file = args[1]
                meta_file = args[2]
        elif len(args) >= 4:
            bw_file = args[0]
            bw_bsky_file = args[1]
            lookup_file = args[2]
            meta_file = args[3]

    if bw_file is None and bw_bsky_file is not None and _looks_like_brandwatch_csv(bw_bsky_file):
        bw_file = bw_bsky_file
        bw_bsky_file = None

    brandwatch_files = []
    if bw_file is not None:
        brandwatch_files.append(bw_file)
    if bw_bsky_file is not None and _looks_like_brandwatch_csv(bw_bsky_file):
        brandwatch_files.append(bw_bsky_file)

    if not brandwatch_files:
        raise ValueError("At least one Brandwatch input file is required.")

    df = pd.concat([read_brandwatch_csv(f, "Brandwatch CSV") for f in brandwatch_files], ignore_index=True)

    df["_handle"] = df["Author"].astype(str).str.lower().str.strip()
    df["_domain"] = df["Domain"].astype(str).str.lower().str.strip()

    if lookup_file is not None:
        dol = pd.read_csv(lookup_file, low_memory=False)
        dol.columns = [str(col).strip() for col in dol.columns]
        if "Twitter Handle" not in dol.columns:
            raise ValueError("DOL/KOL lookup file must contain a 'Twitter Handle' column.")
        dol["_handle"] = dol["Twitter Handle"].astype(str).str.lower().str.strip()
        dol = dol.rename(columns={"KOL": "Lilly KOL", "DOL": "DOL Yes/No", "DOL Profile": "DOL Profile"})
        dol = dol[["_handle", "DOL Yes/No", "DOL Profile", "Lilly KOL"]].drop_duplicates("_handle")
        df = df.merge(dol, on="_handle", how="left")
        df["DOL Yes/No"] = df["DOL Yes/No"].fillna("No")
        df["DOL Profile"] = df["DOL Profile"].fillna("N/A")
        df["Lilly KOL"] = df["Lilly KOL"].fillna("No")
    else:
        df["DOL Yes/No"] = INPUT_NOT_AVAILABLE
        df["DOL Profile"] = INPUT_NOT_AVAILABLE
        df["Lilly KOL"] = INPUT_NOT_AVAILABLE

    if meta_file is not None:
        meta = pd.read_csv(meta_file, low_memory=False)
        meta.columns = [str(col).strip() for col in meta.columns]
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
            "Ecosystem": "Ecosystem Type",
        })
        meta_select = ["Name", "Institution", "NPI", "Validated US?",
                       "Continent", "Country", "State", "City", "Specialty 1", "Specialty 2",
                       "Classification", "Job Role", "Ecosystem Type",
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
                                                               "Specialty 1", "Specialty 2", "Classification",
                                                               "Job Role", "Ecosystem Type"] ]
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
        df["Classification"] = INPUT_NOT_AVAILABLE
        df["Job Role"] = INPUT_NOT_AVAILABLE
        df["Ecosystem Type"] = INPUT_NOT_AVAILABLE

    category_present = any(col in df.columns for col in CATEGORY_COLUMNS) or "Category Details" in df.columns
    ordered_columns = [
        col for col in OUTPUT_COLUMNS
        if col in df.columns or (
            col == "Category Details" and category_present
        ) or (
            col in CATEGORY_COLUMNS and category_present
        ) or (
            col not in CATEGORY_COLUMNS and col != "Category Details"
        )
    ]
    for col in ordered_columns:
        if col in df.columns:
            continue
        if col in ["DOL Yes/No", "DOL Profile", "Lilly KOL",
                   "Name", "Institution", "NPI", "Validated US?",
                   "Continent", "Country", "State", "City",
                   "Specialty 1", "Specialty 2", "Classification",
                   "Job Role", "Ecosystem Type"]:
            df[col] = INPUT_NOT_AVAILABLE
        else:
            df[col] = ""
    return df[ordered_columns]


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


def test_as_bool_array_accepts_series_mask():
    mask = pd.Series([True, False], index=[10, 11])
    assert _as_bool_array(mask).tolist() == [True, False]


def test_process_accepts_brandwatch_file_without_category_columns(tmp_path):
    bw_file = tmp_path / "bw_without_categories.csv"
    base_cols = [c for c in BW_KEEP_COLUMNS if c not in CATEGORY_COLUMNS and c != "Category Details"]
    row = {
        "Date": "2026-07-17",
        "Url": "https://example.com/post",
        "Domain": "x.com",
        "Author": "x_handle",
        "Likes": "10",
        "Comments": "2",
        "Shares": "5",
        "Full Text": "hello world",
        "Mentioned Authors": "",
        "Thread Author": "",
        "Thread Entry Type": "post",
        "X Author ID": "abc",
        "X Followers": "123",
        "Engagement Score": "42",
        "Bluesky Author Id": "",
    }
    csv_text = ",".join(base_cols) + "\n" + ",".join(str(row.get(c, "")) for c in base_cols) + "\n"
    bw_file.write_text(csv_text, encoding="utf-8")

    result = process(str(bw_file), None, None)

    assert "Category Details" not in result.columns
    assert all(col not in result.columns for col in CATEGORY_COLUMNS)
    assert len(result) == 1
    assert result.iloc[0]["Author"] == "x_handle"
    assert result.iloc[0]["Classification"] == INPUT_NOT_AVAILABLE
    assert result.iloc[0]["Job Role"] == INPUT_NOT_AVAILABLE
    assert result.iloc[0]["Ecosystem Type"] == INPUT_NOT_AVAILABLE


def test_process_accepts_single_brandwatch_file_with_lookup_and_metadata(tmp_path):
    bw_file = tmp_path / "bw.csv"
    header = ",".join(BW_KEEP_COLUMNS)
    x_values = []
    bsky_values = []
    for c in BW_KEEP_COLUMNS:
        if c == "Date":
            x_values.append("2026-07-17")
            bsky_values.append("2026-07-18")
        elif c == "Domain":
            x_values.append("x.com")
            bsky_values.append("bsky.app")
        elif c == "Author":
            x_values.append("x_handle")
            bsky_values.append("handle1.bsky.social")
        elif c == "Url":
            x_values.append("https://x.com/example")
            bsky_values.append("https://bsky.app/profile/example")
        elif c == "Full Text":
            x_values.append("x tweet")
            bsky_values.append("bsky post")
        elif c.startswith("All GLP1 by Category"):
            x_values.append("1")
            bsky_values.append("0")
        elif c == "Category Details":
            x_values.append("x category")
            bsky_values.append("bsky category")
        else:
            x_values.append("val")
            bsky_values.append("val")
    bw_file.write_text(header + "\n" + ",".join(x_values) + "\n" + ",".join(bsky_values) + "\n", encoding="utf-8")

    dol_file = tmp_path / "dol.csv"
    dol_file.write_text("Twitter Handle,DOL, KOL, DOL Profile\n" + "x_handle,Yes,No,https://example.com/dol\n", encoding="utf-8")

    meta_file = tmp_path / "meta.csv"
    meta_file.write_text(
        "Twitter Handle,Bluesky Handle,First Name,Last Name,Institution,Country,State,City,Medical Specialty 1,Medical Specialty 2,Classification,Job Role,Ecosystem\n"
        "x_handle,,Jane,Smith,Example Health,United States,CA,Los Angeles,Cardiology,Oncology,Clinical,Attending,Care Delivery\n"
        ",handle1.bsky.social,Jane,Johnson,Example Health,United States,NY,New York,Cardiology,Oncology,Academic,Researcher,Health System\n",
        encoding="utf-8",
    )

    result = process(str(bw_file), str(dol_file), str(meta_file))

    assert len(result) == 2
    assert set(result["Domain"]) == {"x.com", "bsky.app"}
    assert result.iloc[0]["DOL Yes/No"] == "Yes"
    x_row = result.loc[result["Domain"] == "x.com"].iloc[0]
    bsky_row = result.loc[result["Domain"] == "bsky.app"].iloc[0]
    assert x_row["Classification"] == "Clinical"
    assert x_row["Job Role"] == "Attending"
    assert x_row["Ecosystem Type"] == "Care Delivery"
    assert bsky_row["Classification"] == "Academic"
    assert bsky_row["Job Role"] == "Researcher"
    assert bsky_row["Ecosystem Type"] == "Health System"


def test_process_handles_duplicate_bluesky_metadata_handles(tmp_path):
    bw_file = tmp_path / "bw.csv"
    header = ",".join(BW_KEEP_COLUMNS)
    values = []
    for c in BW_KEEP_COLUMNS:
        if c == "Date":
            values.append("2026-07-17")
        elif c == "Domain":
            values.append("bsky.app")
        elif c == "Author":
            values.append("handle1.bsky.social")
        elif c == "Full Text":
            values.append("hello world")
        elif c == "Category Details":
            values.append("foo")
        elif c.startswith("All GLP1 by Category"):
            values.append("1")
        else:
            values.append("val")
    bw_file.write_text(header + "\n" + ",".join(values) + "\n", encoding="utf-8")

    meta_file = tmp_path / "meta.csv"
    meta_file.write_text(
        "Twitter Handle,Bluesky Handle,First Name,Last Name,Institution,Country,State,City,Medical Specialty 1,Medical Specialty 2\n"
        "handle1,handle1.bsky.social,Jane,Smith,Example Health,United States,CA,Los Angeles,Cardiology,Oncology\n"
        "handle1,handle1.bsky.social,Jane,Smith,Example Health,United States,CA,Los Angeles,Cardiology,Oncology\n",
        encoding="utf-8",
    )

    result = process(str(bw_file), None, None, str(meta_file))

    assert result.iloc[0]["Name"] == "Jane Smith"
    assert result.iloc[0]["Institution"] == "Example Health"
    assert result.iloc[0]["Country"] == "United States"


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
