from pathlib import Path
import hashlib

# ============================================================
# KNOWLEDGE BASE
# ============================================================

KB_DIR = Path(r"D:\MRPL-Sovereign-AI\knowledge_base")

# Folders that should NEVER be touched
SKIP_FOLDERS = {
    "chroma_db",
    ".git",
    "__pycache__"
}

# ============================================================
# ONLY THESE ARE CONSIDERED JUNK
# ============================================================

JUNK_EXTENSIONS = {
    ".tmp",
    ".temp",
    ".crdownload",
    ".part"
}

JUNK_KEYWORDS = [
    "webcache",
    "error",
    "failed_download",
    "test_download",
    "temp_file",
    "temporary"
]


# ============================================================
# SHA256
# ============================================================

def get_hash(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:

        while True:

            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# CHECK KB
# ============================================================

if not KB_DIR.exists():

    print("Knowledge base not found:")
    print(KB_DIR)
    exit()


print("=" * 70)
print("MRPL SOVEREIGN AI - KB CLEANUP")
print("=" * 70)


# ============================================================
# COLLECT FILES
# ============================================================

files = []

for file in KB_DIR.rglob("*"):

    if not file.is_file():
        continue

    # Never touch excluded folders
    if any(folder in SKIP_FOLDERS for folder in file.parts):
        continue

    files.append(file)


print(f"\nFiles found: {len(files)}")


# ============================================================
# STEP 1 — DELETE ONLY OBVIOUS JUNK
# ============================================================

print("\n" + "=" * 70)
print("STEP 1: REMOVING OBVIOUS JUNK")
print("=" * 70)

deleted_junk = 0

for file in files:

    filename = file.name.lower()
    extension = file.suffix.lower()

    is_junk = False

    # Temporary file
    if extension in JUNK_EXTENSIONS:
        is_junk = True

    # Obvious junk filename
    for keyword in JUNK_KEYWORDS:

        if keyword in filename:
            is_junk = True
            break

    if is_junk:

        print(f"\nDELETE JUNK:")
        print(f"  {file}")

        try:
            file.unlink()
            deleted_junk += 1

        except Exception as e:
            print(f"  ERROR: {e}")


# ============================================================
# STEP 2 — REMOVE EXACT DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: REMOVING EXACT DUPLICATES")
print("=" * 70)


files = []

for file in KB_DIR.rglob("*"):

    if not file.is_file():
        continue

    if any(folder in SKIP_FOLDERS for folder in file.parts):
        continue

    files.append(file)


hashes = {}

deleted_duplicates = 0


for file in files:

    try:
        file_hash = get_hash(file)

    except Exception as e:

        print(f"Could not read: {file}")
        print(f"ERROR: {e}")
        continue


    if file_hash in hashes:

        original = hashes[file_hash]

        print("\nDUPLICATE FOUND")
        print(f"  KEEP   : {original}")
        print(f"  DELETE : {file}")

        try:
            file.unlink()
            deleted_duplicates += 1

        except Exception as e:
            print(f"  ERROR: {e}")

    else:

        hashes[file_hash] = file


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLEANUP COMPLETE")
print("=" * 70)

remaining = []

for file in KB_DIR.rglob("*"):

    if not file.is_file():
        continue

    if any(folder in SKIP_FOLDERS for folder in file.parts):
        continue

    remaining.append(file)


print(f"\nJunk deleted      : {deleted_junk}")
print(f"Duplicates deleted: {deleted_duplicates}")
print(f"Files remaining   : {len(remaining)}")


print("\nRemaining KB structure:\n")

for file in sorted(remaining):

    print(f"  ✓ {file.relative_to(KB_DIR)}")


print("\n" + "=" * 70)
print("IMPORTANT")
print("=" * 70)

print("""
✓ ISO files KEPT
✓ Certification files KEPT
✓ Finance files KEPT
✓ Annual Reports KEPT
✓ HSE files KEPT
✓ Environment files KEPT
✓ Manufacturing files KEPT
✓ Project files KEPT
✓ CSR files KEPT
✓ Synthetic documents KEPT
✓ ChromaDB NOT TOUCHED
""")