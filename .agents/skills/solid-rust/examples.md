# SOLID (Rust) — Examples

Runnable-style teaching snippets for traits, generics, and module boundaries.
See [SKILL.md](SKILL.md) for narrative and [reference.md](reference.md) for
checklists and tables.

---

## Example 1: SRP — policy vs dependency injection

`OrderService` decides *when* to notify; `Notifier` implementations handle
*how*. One type should not both encode business rules and own SMTP details.

```rust
pub trait Notifier {
    fn notify(&self, message: &str);
}

pub struct OrderService<N: Notifier> {
    notifier: N,
}

impl<N: Notifier> OrderService<N> {
    pub fn new(notifier: N) -> Self {
        Self { notifier }
    }

    pub fn place_order(&self, total_cents: i64) {
        if total_cents > 10_000 {
            self.notifier.notify("Large order placed");
        }
    }
}

pub struct LogNotifier;
impl Notifier for LogNotifier {
    fn notify(&self, message: &str) {
        eprintln!("notify: {message}");
    }
}
```

**Tie-back:** **SRP** — order policy stays in `OrderService`; transport/logging
lives behind `Notifier`. Constructor injection keeps the split explicit.

---

## Example 2: OCP — extend with a new `impl`, not a growing `match`

Add a pricing strategy by implementing a trait. Stable code depends on `Price`
only; new product rules live in new types.

```rust
pub trait Price {
    fn line_total(&self, qty: u32, unit_cents: i64) -> i64;
}

pub struct StandardPrice;
impl Price for StandardPrice {
    fn line_total(&self, qty: u32, unit_cents: i64) -> i64 {
        qty as i64 * unit_cents
    }
}

pub struct TenPercentOff;
impl Price for TenPercentOff {
    fn line_total(&self, qty: u32, unit_cents: i64) -> i64 {
        (qty as i64 * unit_cents * 9) / 10
    }
}

pub fn checkout<P: Price>(pricing: &P, qty: u32, unit_cents: i64) -> i64 {
    pricing.line_total(qty, unit_cents)
}
```

**Tie-back:** **OCP** — new discount types add `impl Price` without editing
`checkout`’s signature or a central product `match`.

---

## Example 3: LSP — trait contracts both implementations honor

Callers of `format_label` expect a `Result`, not a panic. Every `impl` must
preserve that postcondition.

```rust
pub trait LabelFormatter {
    /// Returns `Ok` with a non-empty label, or `Err` — never panics.
    fn format_label(&self, raw: &str) -> Result<String, std::fmt::Error>;
}

pub struct UppercaseFormatter;
impl LabelFormatter for UppercaseFormatter {
    fn format_label(&self, raw: &str) -> Result<String, std::fmt::Error> {
        if raw.trim().is_empty() {
            return Err(std::fmt::Error);
        }
        Ok(raw.trim().to_uppercase())
    }
}

pub struct PassthroughFormatter;
impl LabelFormatter for PassthroughFormatter {
    fn format_label(&self, raw: &str) -> Result<String, std::fmt::Error> {
        if raw.is_empty() {
            return Err(std::fmt::Error);
        }
        Ok(raw.to_string())
    }
}

pub fn render<F: LabelFormatter>(f: &F, input: &str) -> Option<String> {
    f.format_label(input).ok()
}
```

**Tie-back:** **LSP** — either formatter can substitute for `F` in `render`
without breaking callers’ assumptions about errors vs success.

---

## Example 4: ISP — role traits and `+` bounds

Clients that only read should depend on `Readable`, not on a fat “device” trait
that also flashes LEDs.

```rust
pub trait Readable {
    fn read_line(&mut self) -> Option<String>;
}

pub trait Writable {
    fn write_line(&mut self, line: &str);
}

pub struct LogDevice {
    lines: Vec<String>,
    cursor: usize,
}

impl Readable for LogDevice {
    fn read_line(&mut self) -> Option<String> {
        let line = self.lines.get(self.cursor).cloned()?;
        self.cursor += 1;
        Some(line)
    }
}

impl Writable for LogDevice {
    fn write_line(&mut self, line: &str) {
        self.lines.push(line.to_string());
    }
}

/// Only needs reading — does not depend on `Writable`.
pub fn dump<R: Readable>(mut r: R) -> Vec<String> {
    let mut out = Vec::new();
    while let Some(line) = r.read_line() {
        out.push(line);
    }
    out
}
```

**Tie-back:** **ISP** — `dump` binds only `Readable`. Types can implement one or
both traits; clients take the smallest bound they need.

---

## Example 5: DIP — domain depends on `Repository`; wiring at the root

High-level `Greeter` names the abstraction. Concrete databases (or fakes) live
at the composition root.

```rust
pub trait Repository {
    fn display_name(&self, id: u64) -> Option<String>;
}

pub struct PostgresUserRepo;
impl Repository for PostgresUserRepo {
    fn display_name(&self, _id: u64) -> Option<String> {
        // Real driver would go here.
        None
    }
}

pub struct FakeRepo;
impl Repository for FakeRepo {
    fn display_name(&self, id: u64) -> Option<String> {
        (id == 1).then_some("alice".to_string())
    }
}

pub struct Greeter<R: Repository> {
    repo: R,
}

impl<R: Repository> Greeter<R> {
    pub fn new(repo: R) -> Self {
        Self { repo }
    }

    pub fn hello(&self, id: u64) -> String {
        self.repo
            .display_name(id)
            .map(|n| format!("Hello, {n}"))
            .unwrap_or_else(|| "Hello, guest".into())
    }
}

// In tests: Greeter::new(FakeRepo)
// In main:   Greeter::new(PostgresUserRepo::connect(...)?)
```

**Tie-back:** **DIP** — `Greeter` depends on `Repository`, not on a specific
client crate. Inversion keeps policy testable with `FakeRepo`.

---

## Example 6: `ExportRequest` and `ChunkBuffer` (narrow API vs buffer ownership)

Illustrative names matching [SKILL.md](SKILL.md) §7: the export pipeline depends
only on a **small trait** (`ExportRequest`), and chunking policy lives in
**one** type (`ChunkBuffer`).

```rust
/// What the export job needs from any document — not the whole domain surface (ISP).
pub trait ExportRequest {
    fn stable_name(&self) -> &str;
    fn byte_length(&self) -> usize;
}

pub struct Document {
    title: String,
    body: Vec<u8>,
}

impl ExportRequest for Document {
    fn stable_name(&self) -> &str {
        &self.title
    }

    fn byte_length(&self) -> usize {
        self.body.len()
    }
}

pub fn export_header(req: &dyn ExportRequest) -> String {
    format!("{}:{}B", req.stable_name(), req.byte_length())
}

/// Owns the buffer and chunking rules; callers only see slices (SRP).
pub struct ChunkBuffer {
    data: Vec<u8>,
    chunk_size: usize,
}

impl ChunkBuffer {
    pub fn new(data: Vec<u8>, chunk_size: usize) -> Self {
        assert!(chunk_size > 0, "chunk size must be positive");
        Self { data, chunk_size }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn chunk(&self, index: usize) -> Option<&[u8]> {
        let start = index.checked_mul(self.chunk_size)?;
        if start >= self.data.len() {
            return None;
        }
        let end = (start + self.chunk_size).min(self.data.len());
        Some(&self.data[start..end])
    }
}
```

**Tie-back:** **`ExportRequest`** keeps the export boundary small (ISP).
**`ChunkBuffer`** is the only place that decides how bytes are split; export
code does not re-encode that policy (SRP).

---

## Quick navigation

- Narrative guide: [SKILL.md](SKILL.md)
- Checklists and tables: [reference.md](reference.md)