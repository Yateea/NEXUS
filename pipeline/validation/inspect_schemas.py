from pathlib import Path
import polars as pl

root = Path(r"data\generated")

print("=== DIMENSIONS ===")
for p in sorted((root / "dimensions").glob("*")):
    lf = pl.scan_parquet(p) if p.suffix == ".parquet" else (pl.scan_csv(p) if p.suffix == ".csv" else None)
    if lf is None:
        continue
    df = lf.collect()
    print(f"\n{p.name}  ({df.height:,} lignes)")
    print(dict(df.schema))
    print(df.head(2))

print("\n=== FACTS ===")
for d in sorted((root / "facts").iterdir()):
    lf = pl.scan_parquet(str(d / "*.parquet"))
    n = lf.select(pl.len()).collect().item()
    print(f"\n{d.name}  ({n:,} lignes)")
    print(dict(lf.collect_schema()))
    print(lf.head(2).collect())