from pathlib import Path
import zipfile

# paths
TMAX_ZIP_DIR = Path("data/raw/prism/tmax")
PPT_ZIP_DIR = Path("data/raw/prism/ppt")

TMAX_OUT = Path("data/raw/prism_unzipped/tmax")
PPT_OUT = Path("data/raw/prism_unzipped/ppt")


def unzip_all(zip_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_files = list(zip_dir.glob("*.zip"))
    print(f"Found {len(zip_files)} zip files in {zip_dir}")

    for i, zip_path in enumerate(zip_files):
        target_dir = out_dir / zip_path.stem

        if target_dir.exists():
            continue  # skip if already unzipped

        target_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(target_dir)

        if i % 10 == 0:
            print(f"Unzipped {i}/{len(zip_files)}")


def main():
    unzip_all(TMAX_ZIP_DIR, TMAX_OUT)
    unzip_all(PPT_ZIP_DIR, PPT_OUT)

    print("\nDone unzipping PRISM data.")


if __name__ == "__main__":
    main()