# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# import pandas as pd
# from io import BytesIO

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],      # for local dev; restrict in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# def convert_timestamp_ns_to_seconds(ts_str):
#     """
#     Supports:
#     1) 'MM:SS:MS'  -> '00:00:400'
#     2) raw integer nanoseconds -> 400000000 (0.4 sec)
#     """
#     try:
#         s = str(ts_str).strip()

#         # Case 1: "MM:SS:MS"
#         if ":" in s:
#             parts = s.split(":")
#             # "MM:SS:MS"
#             if len(parts) == 3:
#                 minutes = int(parts[0])
#                 seconds = int(parts[1])
#                 millis = int(parts[2])
#                 total_seconds = minutes * 60 + seconds + millis / 1000.0
#                 return total_seconds
#             # "MM:SS"
#             if len(parts) == 2:
#                 minutes = int(parts[0])
#                 seconds = int(parts[1])
#                 return minutes * 60 + seconds
#             # anything else: fail
#             return None

#         # Case 2: plain number -> treat as nanoseconds
#         # e.g. 400000000 -> 0.4 sec
#         val = float(s)
#         # if it's very small (< 1e6) maybe it's already seconds or ms,
#         # but since column is timestamp_ns, we assume nanoseconds:
#         return val / 1e9

#     except Exception as e:
#         print("Timestamp parse error for:", ts_str, "->", e)
#         return None


# @app.post("/upload-csv")
# async def upload_csv(file: UploadFile = File(...)):
#     """
#     CSV must contain: date, timestamp_ns, frame_index, count

#     Returns JSON:
#       - records      -> [{ date, timestamp, count }]
#       - time_series  -> [{ date, time_sec, count }]
#       - per_second   -> [{ date, second, avg_count }]
#       - frame_series -> [{ date, frame_index, count }]
#       - summary      -> stats
#     """
#     try:
#         raw = await file.read()

#         # --- Read CSV into DataFrame ---
#         try:
#             df = pd.read_csv(BytesIO(raw))
#         except Exception as e:
#             print("Pandas read_csv error:", e)
#             return {
#                 "status": "error",
#                 "message": f"Failed to read CSV: {e}",
#                 "records": [],
#             }

#         print("CSV columns:", list(df.columns))

#         required_cols = {"date", "timestamp_ns", "frame_index", "count"}
#         if not required_cols.issubset(df.columns):
#             return {
#                 "status": "error",
#                 "message": (
#                     "CSV must contain columns: "
#                     "date, timestamp_ns, frame_index, count"
#                 ),
#                 "records": [],
#             }

#         # Normalize types
#         df["date"] = df["date"].astype(str)

#         # Convert timestamp_ns -> seconds using robust parser
#         df["time_sec"] = df["timestamp_ns"].apply(convert_timestamp_ns_to_seconds)

#         # Check if *all* conversions failed
#         if df["time_sec"].isna().all():
#             return {
#                 "status": "error",
#                 "message": (
#                     "No valid rows after timestamp conversion. "
#                     "Ensure 'timestamp_ns' is either 'MM:SS:MS' (e.g. 00:00:400) "
#                     "or a numeric nanoseconds value (e.g. 400000000)."
#                 ),
#                 "records": [],
#             }

#         # Drop only rows that failed; keep others
#         df = df.dropna(subset=["time_sec"])

#         # ---------- 1) time series: (date, time_sec, count) ----------
#         time_series_df = df[["date", "time_sec", "count"]].copy()
#         time_series = time_series_df.to_dict(orient="records")

#         # ---------- 2) per-second average, grouped by (date, sec_bucket) ----------
#         df["sec_bucket"] = df["time_sec"].astype(int)
#         per_second_df = (
#             df.groupby(["date", "sec_bucket"])["count"]
#             .mean()
#             .reset_index()
#             .rename(columns={"sec_bucket": "second", "count": "avg_count"})
#         )
#         per_second = per_second_df.to_dict(orient="records")

#         # ---------- 3) frame_index vs count, with date ----------
#         frame_series_df = df[["date", "frame_index", "count"]].copy()
#         frame_series = frame_series_df.to_dict(orient="records")

#         # ---------- Summary stats over all rows ----------
#         summary = {
#             "min_count": float(df["count"].min()),
#             "max_count": float(df["count"].max()),
#             "mean_count": float(df["count"].mean()),
#             "num_points": int(len(df)),
#         }

#         # ---------- records: what Dashboard.js uses for main time-series ----------
#         # Each record: { date, timestamp, count }
#         records = [
#             {
#                 "date": row["date"],
#                 "timestamp": float(row["time_sec"]),
#                 "count": float(row["count"]),
#             }
#             for _, row in time_series_df.iterrows()
#         ]

#         response = {
#             "status": "success",
#             "records": records,
#             "time_series": time_series,
#             "per_second": per_second,
#             "frame_series": frame_series,
#             "summary": summary,
#         }

#         print("Processed rows:", len(records))
#         return response

#     except Exception as e:
#         # This catches any unexpected errors and still returns JSON
#         print("Unexpected server error:", e)
#         return {
#             "status": "error",
#             "message": f"Failed to process CSV: {e}",
#             "records": [],
#         }


from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import BytesIO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local dev; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def convert_timestamp_to_seconds(value, unit_hint="ms"):
    """
    Supports:
    1) 'MM:SS:MS' or 'MM:SS'      -> '00:03:600', '00:03'
    2) plain number:
       - if unit_hint == 'ns'     -> treat as nanoseconds
       - otherwise (ms)           -> treat as milliseconds
    """
    try:
        s = str(value).strip()

        # Case 1: "MM:SS:MS" or "MM:SS"
        if ":" in s:
            parts = s.split(":")
            if len(parts) == 3:  # MM:SS:MS
                minutes = int(parts[0])
                seconds = int(parts[1])
                millis = int(parts[2])
                return minutes * 60 + seconds + millis / 1000.0
            if len(parts) == 2:  # MM:SS
                minutes = int(parts[0])
                seconds = int(parts[1])
                return minutes * 60 + seconds
            return None

        # Case 2: plain number
        val = float(s)
        if unit_hint == "ns":
            return val / 1e9
        else:  # default ms
            return val / 1000.0

    except Exception as e:
        print("Timestamp parse error for:", value, "->", e)
        return None


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """
    Accepts CSV with at least:
      - date
      - one of: timestamp_ms, timestamp_ns, time_stamp
      - one of: frame_index, frame
      - one of: count, cout

    May also contain:
      - alert (e.g. 'lens_covered_or_extremely_dark', 'camera_frozen')

    Returns:
      - records      -> [{ date, timestamp, count, (optional) alert }]
      - time_series  -> [{ date, time_sec, count, (optional) alert }]
      - per_second   -> [{ date, second, avg_count }]
      - frame_series -> [{ date, frame_index, count, (optional) alert }]
      - summary      -> stats
    """
    try:
        raw = await file.read()

        try:
            df = pd.read_csv(BytesIO(raw))
        except Exception as e:
            print("read_csv error:", e)
            return {
                "status": "error",
                "message": f"Failed to read CSV: {e}",
                "records": [],
            }

        cols = set(df.columns)
        print("CSV columns:", list(df.columns))

        # ---- figure out which columns to use (flexible) ----
        if "date" not in cols:
            return {
                "status": "error",
                "message": "CSV must contain a 'date' column.",
                "records": [],
            }

        # timestamp column & unit
        timestamp_col = None
        unit_hint = "ms"
        for cand, hint in [
            ("timestamp_ms", "ms"),
            ("timestamp_ns", "ns"),
            ("time_stamp", "ms"),  # your older name
            ("timestamp", "ms"),
        ]:
            if cand in cols:
                timestamp_col = cand
                unit_hint = hint
                break

        if timestamp_col is None:
            return {
                "status": "error",
                "message": (
                    "CSV must contain one timestamp column: "
                    "timestamp_ms OR timestamp_ns OR time_stamp."
                ),
                "records": [],
            }

        # frame index column
        frame_col = None
        for cand in ["frame_index", "frame"]:
            if cand in cols:
                frame_col = cand
                break

        if frame_col is None:
            return {
                "status": "error",
                "message": (
                    "CSV must contain frame column: frame_index OR frame."
                ),
                "records": [],
            }

        # count column
        count_col = None
        for cand in ["count", "cout"]:
            if cand in cols:
                count_col = cand
                break

        if count_col is None:
            return {
                "status": "error",
                "message": (
                    "CSV must contain count column: count OR cout."
                ),
                "records": [],
            }

        # optional alert
        has_alert = "alert" in cols
        if has_alert:
            df["alert"] = df["alert"].astype(str).fillna("")

        # normalize date
        df["date"] = df["date"].astype(str)

        # ---- convert timestamp -> seconds ----
        df["time_sec"] = df[timestamp_col].apply(
            lambda v: convert_timestamp_to_seconds(v, unit_hint)
        )

        if df["time_sec"].isna().all():
            return {
                "status": "error",
                "message": (
                    "No valid rows after timestamp conversion. "
                    f"Check the format of {timestamp_col}."
                ),
                "records": [],
            }

        df = df.dropna(subset=["time_sec"])

        # ---------- 1) time series ----------
        ts_cols = ["date", "time_sec", count_col]
        if has_alert:
            ts_cols.append("alert")
        time_series_df = df[ts_cols].copy()
        time_series_df.rename(columns={count_col: "count"}, inplace=True)
        time_series = time_series_df.to_dict(orient="records")

        # ---------- 2) per-second average ----------
        df["sec_bucket"] = df["time_sec"].astype(int)
        per_second_df = (
            df.groupby(["date", "sec_bucket"])[count_col]
            .mean()
            .reset_index()
            .rename(columns={"sec_bucket": "second", count_col: "avg_count"})
        )
        per_second = per_second_df.to_dict(orient="records")

        # ---------- 3) frame vs count ----------
        fs_cols = ["date", frame_col, count_col]
        if has_alert:
            fs_cols.append("alert")
        frame_series_df = df[fs_cols].copy()
        frame_series_df.rename(
            columns={frame_col: "frame_index", count_col: "count"},
            inplace=True,
        )
        frame_series = frame_series_df.to_dict(orient="records")

        # ---------- summary ----------
        summary = {
            "min_count": float(df[count_col].min()),
            "max_count": float(df[count_col].max()),
            "mean_count": float(df[count_col].mean()),
            "num_points": int(len(df)),
        }

        # ---------- records for frontend ----------
        records = []
        for _, row in time_series_df.iterrows():
            rec = {
                "date": row["date"],
                "timestamp": float(row["time_sec"]),
                "count": float(row["count"]),
            }
            if has_alert:
                rec["alert"] = row["alert"]
            records.append(rec)

        print("Processed rows:", len(records))
        return {
            "status": "success",
            "records": records,
            "time_series": time_series,
            "per_second": per_second,
            "frame_series": frame_series,
            "summary": summary,
        }

    except Exception as e:
        print("Unexpected server error:", e)
        return {
            "status": "error",
            "message": f"Failed to process CSV: {e}",
            "records": [],
        }
