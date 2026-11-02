fn main() {
    let _ctx = keel_kernel::Context::new();
    println!("keel {} (placeholder name) - kernel up, no plugins mounted", env!("CARGO_PKG_VERSION"));
}
