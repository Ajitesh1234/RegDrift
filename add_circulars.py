"""Add many circulars at once: put PDFs/TXTs (or a .zip of them) anywhere, then run

    python add_circulars.py path/to/zip_or_folder [more paths...]

It copies them into circulars/ with clean file names, skips duplicates, and warns about scanned
(image-only) PDFs that have no extractable text.
"""
import os, re, shutil, sys, tempfile, zipfile
import regdrift

DEST = os.path.join(regdrift.HERE, "circulars")
MIN_CHARS = 500


def clean_name(name):
    stem, ext = os.path.splitext(os.path.basename(name))
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()[:60] or "circular"
    return stem + ext.lower()


def collect(paths, tmp):
    files = []
    for p in paths:
        if p.lower().endswith(".zip"):
            with zipfile.ZipFile(p) as z:
                z.extractall(os.path.join(tmp, os.path.basename(p)))
            p = os.path.join(tmp, os.path.basename(p))
        if os.path.isdir(p):
            for root, _, fs in os.walk(p):
                files += [os.path.join(root, f) for f in fs if f.lower().endswith((".pdf", ".txt"))
                          and not f.startswith("._") and "__MACOSX" not in root]
        elif p.lower().endswith((".pdf", ".txt")):
            files.append(p)
    return sorted(files)


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    os.makedirs(DEST, exist_ok=True)
    added = bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for src in collect(argv, tmp):
            target = os.path.join(DEST, clean_name(src))
            if os.path.exists(target):
                print(f"skip (already there): {os.path.basename(target)}")
                continue
            try:
                n = len(regdrift.load_text(src))
            except Exception as e:
                print(f"FAILED to read {os.path.basename(src)}: {e}")
                bad += 1
                continue
            if n < MIN_CHARS:
                print(f"WARNING {os.path.basename(src)}: only {n} characters of text (scanned PDF?). Not added.")
                bad += 1
                continue
            shutil.copy(src, target)
            print(f"added {os.path.basename(target)} ({n} chars)")
            added += 1
    print(f"\n{added} added, {bad} problem file(s). circulars/ now has "
          f"{len([f for f in os.listdir(DEST) if f.lower().endswith(('.pdf', '.txt'))])} files.")
    return 0 if not bad else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
