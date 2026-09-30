check:
    cargo check --workspace --all-targets

clippy:
    cargo clippy --workspace --all-targets --no-deps

test:
    cargo test --workspace

fmt:
    cargo fmt --all -- --check
