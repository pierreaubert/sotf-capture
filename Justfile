# Use `mbx` when installed, else plain `cargo` (mirrors ../math-audio).
cargo := `if command -v mbx >/dev/null 2>&1; then echo mbx; else echo cargo; fi`

check:
    {{cargo}} check --workspace --all-targets

clippy:
    {{cargo}} clippy --workspace --all-targets --no-deps

dev:
    {{cargo}} build --workspace

prod:
    {{cargo}} build --workspace --release

fmt:
    {{cargo}} fmt --all -- --check
