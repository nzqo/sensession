#!/bin/bash
set -euo pipefail

# Defaults
DEFAULT_ADDR="seemoo@192.168.10.31"
DEFAULT_HOST="192.168.10.1"
DEFAULT_INPUT_DIR="router/CSING/"
TMP_FILE="/tmp/nmon.zip"
REMOTE_DIR="/jffs/CSING"
REMOTE_TMP="/jffs/tmp.zip"

# Show usage information
show_help() {
  cat << EOF
Usage: $(basename "$0") [OPTIONS]

Transfers files to an ASUS router

Options:
  -a, --address ADDRESS    Target device address (default: $DEFAULT_ADDR)
  -h, --host HOST          Host to connect to (default: $DEFAULT_HOST)
  -i, --input DIR          Input directory to zip and transfer (default: $DEFAULT_INPUT_DIR)
  --help                   Show this help message and exit

Example:
  $(basename "$0") -a seemoo@192.168.1.100 -h 192.168.1.1 -i my_files/
EOF
}

# Detect which “close-on-EOF” flag this netcat supports
detect_nc_flag() {
  local nc_bin
  nc_bin=$(command -v nc || command -v netcat) || {
    echo "Error: neither 'nc' nor 'netcat' found" >&2
    exit 1
  }

  # GNU/openbsd-nc style
  if "$nc_bin" -h 2>&1 | grep -q '\-N'; then
    echo -N
    return
  fi

  # macOS/traditional BSD style
  if "$nc_bin" -h 2>&1 | grep -q '\-c'; then
    echo -c
    return
  fi

  echo "Error: this netcat supports neither -N nor -c" >&2
  exit 1
}

# Parse arguments
addr="$DEFAULT_ADDR"
host="$DEFAULT_HOST"
input_dir="$DEFAULT_INPUT_DIR"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -a|--address)
      addr="$2"; shift 2
      ;;
    -h|--host)
      host="$2"; shift 2
      ;;
    -i|--input)
      input_dir="$2"; shift 2
      ;;
    --help)
      show_help; exit 0
      ;;
    *)
      echo "Error: Unknown option $1" >&2
      show_help
      exit 1
      ;;
  esac
done

# Ensure required tools are installed
for tool in zip ssh; do
  if ! command -v "$tool" >/dev/null; then
    echo "Error: Required tool '$tool' not found" >&2
    exit 1
  fi
done

# Locate netcat and pick the right close-on-EOF flag
NC_BIN=$(command -v nc || command -v netcat)
NC_CLOSE_FLAG=$(detect_nc_flag)


# Validate input directory
if [[ ! -d "$input_dir" ]]; then
  echo "Error: Input directory '$input_dir' does not exist." >&2
  exit 1
fi

# Show summary of what will happen
echo "Sync configuration:"
echo "  Address (target device): $addr"
echo "  Host address...........: $host"
echo "  Input directory........: $input_dir"
echo "  Temporary file.........: $TMP_FILE"
echo "  Remote directory.......: $REMOTE_DIR"
echo

# Clean up old temporary file
rm -f "$TMP_FILE"

# Create a zip archive
echo "Zipping files from '$input_dir'..."
zip --junk-paths "$TMP_FILE" "$input_dir"/*

# Transfer the archive to the router
echo "transmitting zipped file"

cat "$TMP_FILE" | ssh "$addr" "cat > $REMOTE_TMP"

# Your original SSH block, exactly unchanged
echo "Connecting to remote router via SSH..."

# NOTE: This SHOULD expand on the router. So this warning is irrelevant.
# shellcheck disable=SC2029
ssh "$addr" "
  echo 'Extracting files...'
  mkdir -p $REMOTE_DIR
  unzip -d $REMOTE_DIR $REMOTE_TMP
  rm $REMOTE_TMP
  echo 'Fixing permissions...'
  chmod +x $REMOTE_DIR/*.sh $REMOTE_DIR/nexutil $REMOTE_DIR/tcpdump $REMOTE_DIR/makecsiparams
"

# Clean up local temporary file
echo "Cleaning up..."
rm -f "$TMP_FILE"

echo "Transfer and setup complete."
