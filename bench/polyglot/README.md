# Polyglot benchmark: Keel vs other harnesses, same local model

Answers the question in the main README: **does Keel beat other harnesses
with the same small local model?** It uses the Python exercises of Aider's
public [polyglot benchmark](https://github.com/Aider-AI/polyglot-benchmark)
(34 Exercism tasks). No Docker needed.

## Fairness rules (enforced by `run.py`)

- Same model (`bonsai-9b`, llama-server at `127.0.0.1:8080`, 16k context),
  same prompt, same time limit (default 300 s) for every harness.
- Task copies contain the stub, test file and `.docs/`, excluding `.meta/`
  reference solutions. This does not enforce read isolation; see below.
- A task passes only if the **original** test file passes (restored before
  scoring); `tampered_test` records if an agent edited it.
- Every harness and the scoring run inside the same **bubblewrap** sandbox:
  the whole system is read-only except the task folder and a throwaway HOME,
  with a **clean environment** (no API keys or base URLs from your shell).
  Stock settings: no personal configs, rules files or plugins.
- Heat guard: a task starts only when `Tctl < 60 °C`; the run stops if it
  passes 95 °C. Each task records its peak.
- Harnesses are interleaved per exercise, and the run resumes from
  `results.jsonl`, so stopping early still leaves comparable data.

## Run

```bash
git clone --depth 1 https://github.com/Aider-AI/polyglot-benchmark ~/Projects/ref/polyglot-benchmark
uv tool run --python 3.12 --from aider-chat aider --version   # cache aider once
python3 bench/polyglot/run.py \
  --exercises ~/Projects/ref/polyglot-benchmark/python/exercises/practice \
  --out /tmp/pg-out --only affine-cipher,bowling,food-chain,grep,phone-number,proverb,scale-generator,tree-building,zebra-puzzle
```

**Heat:** sustained inference holds the Beelink around 90–92 °C for the whole
of each task. The 9-exercise sample above is about 3.5 hours of loud fans.
Run it when nobody is near the machine, not in the evening.

## Findings so far (2026-11-05)

Size of one model request on the same tiny task (proxy-measured):

| Harness | Request | Fits 16k? |
|---|---:|---|
| Keel | ~11 KB | yes |
| Aider | ~3.2k tokens | yes |
| OpenCode 2.2.5 | ~47 KB (~12k tokens) | barely |
| Codex CLI 0.166 | 131 KB | **no**, and its mid-conversation system messages break Bonsai's chat template; it cannot run here at all |

`list-ops`, one run each, 300 s limit:

| Harness | Result |
|---|---|
| Aider | pass (24/24) |
| Keel | 22/24 at timeout; a second run passed in 30 s |
| OpenCode | 22/24 at timeout |

Single runs are noisy with a 2-bit 9B model, so compare pass rates over
several runs per task, not one.

Keel bugs the pilot found: a context overflow killed a whole task with a bare
`status code 400` (fixed in #58: show the server's message, compact and retry).

## Inspect a running benchmark

```bash
python3 bench/polyglot/analyze.py /path/to/run \
  --sessions /path/to/keel/sessions/*.jsonl
python3 bench/polyglot/analyze.py /path/to/run --json
python3 -m unittest discover -s bench/polyglot -p 'test_*.py'
```

Analysis reads files only. Redirect stdout to a separate file if you want a
saved report. It compares complete exercise/repetition blocks, keeps missing
test counts unknown, rejects duplicate attempts, and tolerates a live writer's
incomplete final JSON record with a warning. Session metrics are separate from
attempts because the legacy runner does not record a session identifier.

The current sandbox isolates **writes**, not reads: `.meta` is omitted from
the task copy but remains reachable in the original dataset by absolute path.
Treat the current comparison as exploratory. See [DECISIONS.md](DECISIONS.md)
for the next controlled experiment and the evidence required before changing
the agent architecture.
