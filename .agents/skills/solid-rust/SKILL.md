---
name: solid-rust-design
description: Apply SOLID design principles in Rust for maintainable modules and traits. Use when writing or reviewing Rust code for modularity, designing traits and trait objects, managing dependencies and visibility, refactoring for testability, or evaluating whether abstractions are justified. Covers traits as interfaces, ownership-aware dependency patterns, static vs dynamic dispatch, and dependency direction.
trigger: >-
  SOLID, SRP, OCP, LSP, ISP, DIP, Rust traits, trait objects, dyn Trait,
  dependency injection, module boundaries, substitutability, refactoring Rust
  for testability, designing abstractions, dependency direction
---

# SOLID Design Principles (Rust)

Use this skill when shaping Rust crates for clarity and safe evolution. Rust
replaces nominal “interfaces” with **traits**; **ownership and borrowing**
determine who holds resources and how dependencies cross function boundaries.
The five principles still apply: they guide where to split modules, how narrow
to make traits, and when generics or `dyn` are appropriate.

---

## 1. SOLID in Rust (brief overview)

| Principle | One-line focus in Rust |
| --------- | ---------------------- |
| **S**RP | One coherent reason to change per type/module; split when responsibilities diverge. |
| **O**CP | Extend behavior via new `impl`s, generics, or enum arms—not by editing stable core logic whenever possible. |
| **L**SP | Every `impl Trait` (or variant arm) must honor the behavioral contract callers rely on. |
| **I**SP | Clients depend only on the methods they need; prefer small traits over “kitchen sink” traits. |
| **D**IP | High-level code depends on abstractions (traits); concrete types implement those traits at the edges. |

**Traits as interfaces:** A trait defines **capabilities**. Types implement
traits explicitly (`impl Foo for Bar`). Object-safe traits can be used as `&dyn
Trait` / `Box<dyn Trait>` for runtime polymorphism.

**Ownership affects dependencies:** Passing `&T` vs owned `T` vs `Arc<T>`
encodes **lifetime and responsibility**. Constructor injection with `impl Trait`
or generic bounds keeps allocation and sharing choices explicit compared to many
OO languages.

---

## 2. Single Responsibility Principle (SRP)

**Idea:** A module, struct, or function should have **one reason to change**—one
axis of change driven by business or technical requirements.

**Cohesion:** Elements that change together belong together. High cohesion means
edits are localized; low cohesion means unrelated concerns churn in the same
place.

**“One reason to change”:** Ask: *If product requirements shift on axis X, how
many unrelated places must I edit?* If the answer is “many,” responsibilities
may be entangled.

**Misconceptions and balance:**

- SRP is **not** “one function per file” or “never put two fields in a struct.”
  Over-fragmentation creates noise, harder navigation, and artificial
  boundaries.
- Aim for **practical cohesion**: split when a responsibility is clearly
  separable and has its own change pressure; keep together when splitting would
  only duplicate context.

**Rust-specific mechanisms:**

- **Ownership/borrowing:** A type that owns a socket or file should be the
  **only** place that closes it; callers borrow for shorter operations.
- **Modules (`mod`, `pub`, `pub(crate)`):** Isolate auth, parsing, IO, and
  policy into separate modules with explicit APIs.
- **Traits:** Separate **behavior interfaces** (e.g. `Notifier`) from concrete
  transports (`EmailNotifier`, `SmsNotifier`).
- **Enums + pattern matching:** Push variant-specific logic into `match` arms or
  methods on the enum; avoid boolean soup that mixes unrelated states.
- **Dependency injection:** Pass dependencies via **constructors** with trait
  bounds (`fn new(n: impl Notifier)`) so the type’s responsibility stays clear.

---

## 3. Open/Closed Principle (OCP)

**Idea:** Designs should be **open for extension** (add new behavior) and
**closed for modification** (avoid churn in stable, well-tested core code).

**Framing:** Prefer **adding** new types (`impl Trait` for a new variant, new
enum arm with delegated logic, new strategy struct) over repeatedly editing a
central `match` that knows every detail—*when that core is truly stable*. The
goal is to **minimize edits** to the part everyone depends on, not to freeze the
codebase.

**Misconceptions:**

- OCP does **not** mean “never change any file.” Early code changes freely;
  mature boundaries deserve stability.
- OCP is **not** an excuse for layers of indirection on every feature. **Simple
  code** that changes in one place is often correct.

**Over-engineering warnings:** Introduce extension points (traits, enums) when
**multiple variants** or **plugin-style** growth is real. A single
implementation does not need a trait hierarchy “for future use.”

**Rust-specific mechanisms:**

- **Traits as extension points:** Add behavior by implementing the trait for a
  new type without editing existing implementations (when the trait is stable).
- **Generics:** `fn process<T: Format>(t: T)` — static dispatch,
  monomorphization, typically fastest.
- **Trait objects:** `&dyn Trait`, `Box<dyn Trait>` — runtime polymorphism, one
  binary, dynamic dispatch cost.
- **Enums + delegated arms:** New variant extends behavior; each arm can call
  shared helpers or inner types.

**GoF patterns (common in Rust):**

- **Strategy:** Swappable algorithms via a trait (e.g. `PaymentStrategy`).
- **Decorator:** Wrapper type also implements the same trait, delegating with
  added behavior.
- **Abstract Factory:** Factory trait returning `Box<dyn Product>` (or generic
  associated types) to construct families of objects.

**Static vs dynamic dispatch:** Prefer **generics** when the implementing type
is known at compile time and performance matters. Use **trait objects** when you
need heterogeneous collections or runtime selection.

---

## 4. Liskov Substitution Principle (LSP)

**Idea:** **Behavioral subtyping:** Any implementation usable where a trait or
abstraction is expected must **preserve correctness** for callers—substitution
must not break assumed behavior.

**Behavioral contracts:**

- **Preconditions:** What must hold before a method runs (e.g. “buffer
  non-empty”).
- **Postconditions:** What the method guarantees after (e.g. “returns parsed
  value or error”).
- **Invariants:** What stays true for the object across operations.

**Contract rules (informal):** Subtypes should **not strengthen** preconditions
(don’t require *more* from callers) or **weaken** postconditions / invariants
(don’t deliver *less* than promised).

**Misconceptions:** LSP is not “all structs must look the same.” It is about
**preserving assumptions**—including error behavior and side effects—that
generic code relies on.

**Rust-specific mechanisms:**

- **Traits** document the contract; document semantics in rustdoc, not only
  signatures.
- **Enums:** Each variant’s payload and handling should respect the same
  **logical** expectations where code treats the enum uniformly.
- **Generics + associated types:** Associated types are part of the contract;
  changing their meaning can break substitutability.
- **Trait objects:** If `dyn Trait` is used, every implementation must be a
  valid substitute—LSP is the **pressure test** for whether `dyn` is safe.

**Over-engineering / pitfalls:** Avoid **needlessly complex** trait bounds;
avoid **`downcast`-style** escape hatches that defeat abstraction. Prefer
correct contracts over clever inheritance-style hierarchies.

---

## 5. Interface Segregation Principle (ISP)

**Idea:** Clients should not depend on methods they do not use. Prefer **small,
focused** traits over a single fat trait that forces every implementor to care
about unrelated operations.

**Misconception:** ISP is **not** “exactly one method per trait.” It is
**client-driven** decomposition: group methods by **who needs them** (role
interfaces).

**Role interfaces:** Name traits by **capability or role**—e.g. `Switchable`,
`Adjustable`, `Lockable`—rather than `EverythingDeviceDoes`.

**Rust-specific mechanisms:**

- **Fine-grained traits** and **multiple `impl` blocks** on one type for
  different traits.
- **`+` bounds:** `fn f<T: Read + Write>(t: T)` — require only the combination
  this API needs.
- **Supertraits:** `trait Serial: Read + Write` when the combined role is
  stable.
- **Trait objects:** Smaller traits mean smaller vtables and clearer `dyn`
  boundaries.
- **Precise generic bounds:** Avoid `T: MegaTrait` when only `T: Serialize` is
  needed.

**Balance:** Like SRP, avoid **over-fragmentation**—ten one-method traits can be
as hard to follow as one god trait. Segregate where **different clients** have
different needs.

---

## 6. Dependency Inversion Principle (DIP)

**Idea:** High-level policy should not depend on low-level details; **both**
should depend on **abstractions**. Abstractions (traits) should not depend on
concrete types; details implement abstractions.

**Misconception:** DIP is **not** the same as **dependency injection (DI)**. DIP
is about **dependency direction** and **ownership of abstractions**. DI
(constructor injection, factories) is a **technique** to achieve inverted
dependencies.

**Inversion of Control (IoC):** High-level modules define the **trait** they
need; low-level modules **implement** it. Wiring happens at the root (`main`,
composition root, tests).

**Rust-specific mechanisms:**

- **Traits** as the stable abstraction layer between layers.
- **Generics with trait bounds** — static dispatch, compile-time checking, no
  heap indirection for the trait itself.
- **`&dyn Trait` / `Box<dyn Trait>`** — dynamic dispatch when details vary at
  runtime.
- **Module visibility** — `pub` / `pub(crate)` to expose **interfaces** while
  hiding struct fields and submodules.
- **Constructor injection** — `new(repo: impl Repository)` or `new(logger:
  Box<dyn Logger>)`; builders for complex graphs.

**GoF patterns (DIP-aligned):**

- **Abstract Factory** — factory trait producing trait objects or generic
  products.
- **Strategy** — algorithm trait injected into a high-level type.
- **Template Method** — trait with **default methods** calling hooks implemented
  by subtypes.

**Over-engineering:** Not every dependency deserves a trait. Direct use of a
concrete type is fine when there is **one** implementation and **no** test or
extension pressure.

**Testing:** Traits and injection enable **mocks and stubs** so high-level logic
can be unit-tested without databases or the network.

---

## 7. Cross-references

- **Runnable patterns and teaching snippets:** [examples.md](examples.md) —
  numbered examples for SRP through DIP, including patterns aligned with this
  repository’s `ChunkBuffer` and `ExportRequest` traits.
- **Checklists, tables, and quick lookup:** [reference.md](reference.md) —
  principle summary, expanded checklists per principle, “when to apply vs.
  overkill,” trait composition, and Rust-to-SOLID mapping.

---

## 8. Quick checklist (rapid validation)

**SRP**

- [ ] Can I name **one primary responsibility** for this type/module?
- [ ] Would unrelated requirement changes force edits in the **same** place
  repeatedly?
- [ ] Did I split **enough** for cohesion without useless fragmentation?

**OCP**

- [ ] Can I **add** a new behavior (new `impl`, variant, strategy) with
  **minimal** edits to stable core code?
- [ ] Is my extension point (trait, enum) **justified** by real variation, not
  speculation?

**LSP**

- [ ] Would **any** implementation of this trait (or enum handling) keep callers
  **correct**?
- [ ] Are preconditions/postconditions **documented** and honored consistently?

**ISP**

- [ ] Does each client need **all** methods on this trait, or should it be
  **split** by role?
- [ ] Are `+` bounds and traits **as small** as the call site requires?

**DIP**

- [ ] Does high-level code depend on a **trait** (or stable API), not a concrete
  DB/HTTP/SDK type?
- [ ] Is wiring (**IoC**) at the boundary (`main`, tests) rather than hard-coded
  `new Concrete()` everywhere?
- [ ] Am I avoiding **needless** abstraction layers on top of a single stable
  implementation?

---

## 9. References

- **Project reference sheets:** [reference.md](reference.md)
- **Code examples:** [examples.md](examples.md)