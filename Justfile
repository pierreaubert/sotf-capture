# Use `mbx` when installed, else plain `cargo` (mirrors ../math-audio).
cargo := `if command -v mbx >/dev/null 2>&1; then echo mbx; else echo cargo; fi`

_default:
	just --list

check:
    {{cargo}} check --workspace --all-targets

# Full capture path (cpal CoreAudio/RemoteIO) for iPhones/iPads; see docs/ios.md.
check-ios:
    {{cargo}} check -p sotf-capture --target aarch64-apple-ios --all-targets

alias lint := clippy

clippy:
    {{cargo}} clippy --workspace --all-targets --all-features --tests --no-deps

dev:
    {{cargo}} build --workspace

alias build := prod

prod:
    {{cargo}} build --workspace --release

fmt:
    {{cargo}} fmt --all -- --check

test:
    {{cargo}} test --all --workspace --all-features --all-targets -- --check

ntest:
    {{cargo}} nextest run --workspace --all-features --all-targets
    
