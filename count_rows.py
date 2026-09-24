import os
import csv
import time
import sys

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = r"D:\My Project\xai_ids_benchmark\CIC IoT-DIAD 2024"
CATEGORIES = ["Benign", "BruteForce", "DDOS", "DOS", "Mirai", "Recon", "Spoofing", "Web-Based"]

grand_total = 0
sample_per_category = {}

print("=" * 100)
print("CSV ROW COUNT REPORT")
print("=" * 100)

start = time.time()

for cat in CATEGORIES:
    cat_dir = os.path.join(BASE_DIR, cat)
    if not os.path.isdir(cat_dir):
        print(f"\n[WARNING] Category folder not found: {cat_dir}")
        continue

    cat_total = 0
    cat_files = []

    for root, dirs, files in os.walk(cat_dir):
        for fname in sorted(files):
            if not fname.lower().endswith(".csv"):
                continue
            fpath = os.path.join(root, fname)
            count = 0
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                header_line = f.readline()
                for line in f:
                    if line.strip():
                        count += 1

            rel = os.path.relpath(fpath, BASE_DIR)
            cat_files.append((rel, count))
            cat_total += count

            if cat not in sample_per_category:
                cols = [c.strip().strip('"') for c in header_line.strip().split(",")]
                sample_per_category[cat] = (rel, len(cols), cols)

    print(f"\n{'-' * 100}")
    print(f"  CATEGORY: {cat}")
    print(f"{'-' * 100}")
    for rel, count in cat_files:
        print(f"    {rel:<75} {count:>12,} rows")
    print(f"    {'- ' * 44}")
    print(f"    {'CATEGORY TOTAL':<75} {cat_total:>12,} rows  ({len(cat_files)} files)")
    grand_total += cat_total

elapsed = time.time() - start

print(f"\n{'=' * 100}")
print(f"  GRAND TOTAL: {grand_total:>15,} rows")
print(f"  Time elapsed: {elapsed:.1f}s")
print(f"{'=' * 100}")

print(f"\n{'=' * 100}")
print("  COLUMN HEADER ANALYSIS (one sample per category)")
print(f"{'=' * 100}")
for cat in CATEGORIES:
    if cat in sample_per_category:
        rel, ncols, cols = sample_per_category[cat]
        print(f"\n  [{cat}] {rel}")
        print(f"    Columns: {ncols}")
        if ncols <= 15:
            print(f"    Headers: {cols}")
        else:
            print(f"    First 10: {cols[:10]}")
            print(f"    Last  5 : {cols[-5:]}")

print(f"\n{'-' * 100}")
all_ncols = [v[1] for v in sample_per_category.values()]
all_headers = [tuple(v[2]) for v in sample_per_category.values()]
if len(set(all_ncols)) == 1:
    print(f"  All sampled CSVs have the SAME number of columns: {all_ncols[0]}")
else:
    print(f"  WARNING: Different column counts detected:")
    for k, v in sample_per_category.items():
        print(f"    {k}: {v[1]} columns")

if len(set(all_headers)) == 1:
    print(f"  All sampled CSVs have IDENTICAL column headers.")
else:
    print(f"  WARNING: Column headers differ across categories!")
    ref_cat = CATEGORIES[0]
    ref_cols = set(sample_per_category[ref_cat][2])
    for cat in CATEGORIES[1:]:
        if cat in sample_per_category:
            other_cols = set(sample_per_category[cat][2])
            if other_cols != ref_cols:
                diff1 = ref_cols - other_cols
                diff2 = other_cols - ref_cols
                if diff1: print(f"    In {ref_cat} but not {cat}: {diff1}")
                if diff2: print(f"    In {cat} but not {ref_cat}: {diff2}")

print()
