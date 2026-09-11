#!/usr/bin/env bash
set -euo pipefail

SOURCE_URL="https://services.healthtech.dtu.dk/suppl/immunology/NAR_NetMHCpan_NetMHCIIpan/NetMHCpan_train.tar.gz"
ARCHIVE_SHA256="06f2c9f20bb959238bf5d601fca0489a0ed3f17648b952f30640205afca8f9b4"
RAW_DIR="$(cd "$(dirname "$0")/../data" && pwd)/pmhc/raw"
ARCHIVE_PATH="$RAW_DIR/NetMHCpan_train.tar.gz"
SOURCE_DIR="$RAW_DIR/NetMHCpan_train"

sha256() {
  shasum -a 256 "$1" | awk '{print $1}'
}

require_hash() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256 "$path")"
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 mismatch for $path: expected $expected, got $actual" >&2
    return 1
  fi
}

verify_extraction() {
  local directory="$1"
  require_hash "$directory/c000_ba" "a5704007e127c8c0e2a3b0043c52ad2f92be5efd01bd2b6a0f7e5424d25d06ef"
  require_hash "$directory/c001_ba" "754c5fecf1c8387b48fc8cf77473fc14c9e89adb2694ac977e51e7028b0304f1"
  require_hash "$directory/c002_ba" "f1ff916e3bd4ed01352b7fa4c6a1852fd9294104f6b344c5dd2eba7961a57aef"
  require_hash "$directory/c003_ba" "9cde2dd51abf1cde03383c5f8120e88243c7b6f486f341a7566bb4ad6f60eb69"
  require_hash "$directory/c004_ba" "a2a28d2565fb8a3e6c76e1f7d2be5b1a3aca28acbabe13c3db4e2063c404f2e5"
  require_hash "$directory/MHC_pseudo.dat" "f46d95dee821db6c139d6cee6f6bf72328468c1d8f6e9741daf21562752ad39a"
}

mkdir -p "$RAW_DIR"

if [[ -e "$ARCHIVE_PATH" ]]; then
  if [[ ! -f "$ARCHIVE_PATH" ]]; then
    echo "Refusing to replace non-file archive path: $ARCHIVE_PATH" >&2
    exit 1
  fi
  require_hash "$ARCHIVE_PATH" "$ARCHIVE_SHA256"
else
  archive_part="$ARCHIVE_PATH.$$.part"
  if [[ -e "$archive_part" ]]; then
    echo "Refusing to overwrite existing partial archive: $archive_part" >&2
    exit 1
  fi
  curl --fail --location --silent --show-error --output "$archive_part" "$SOURCE_URL"
  require_hash "$archive_part" "$ARCHIVE_SHA256"
  mv -n "$archive_part" "$ARCHIVE_PATH"
  nested_archive_path="$ARCHIVE_PATH/$(basename "$archive_part")"
  if [[ -e "$archive_part" ]]; then
    echo "Archive appeared during download; leaving partial archive at $archive_part" >&2
    exit 1
  elif [[ -e "$nested_archive_path" ]]; then
    echo "Archive directory appeared during download; mv nested the verified archive inside it at $nested_archive_path instead of $ARCHIVE_PATH" >&2
    exit 1
  fi
fi

if [[ -e "$SOURCE_DIR" ]]; then
  if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "Refusing to replace non-directory extraction path: $SOURCE_DIR" >&2
    exit 1
  fi
  verify_extraction "$SOURCE_DIR"
else
  extraction_part="$SOURCE_DIR.$$.part"
  if [[ -e "$extraction_part" ]]; then
    echo "Refusing to overwrite existing partial extraction: $extraction_part" >&2
    exit 1
  fi
  mkdir "$extraction_part"
  tar -xzf "$ARCHIVE_PATH" -C "$extraction_part" --strip-components=1
  verify_extraction "$extraction_part"
  mv -n "$extraction_part" "$SOURCE_DIR"
  nested_extraction_path="$SOURCE_DIR/$(basename "$extraction_part")"
  if [[ -e "$extraction_part" ]]; then
    echo "Extraction appeared during unpacking; leaving partial extraction at $extraction_part" >&2
    exit 1
  elif [[ -e "$nested_extraction_path" ]]; then
    echo "Extraction directory appeared during unpacking; mv nested the verified extraction inside it at $nested_extraction_path instead of $SOURCE_DIR" >&2
    exit 1
  fi
fi

printf 'Verified pMHC source data at %s\n' "$SOURCE_DIR"
