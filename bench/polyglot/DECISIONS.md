# What this benchmark should decide

The first decision is whether Keel makes a small local model do useful coding
more reliably than stock alternatives. A passing toy suite, lower prompt size,
or more partial credit cannot establish that on its own.

Keep the ongoing 2026-11-05 Bonsai 9B Q2_0 run as an exploratory pilot.
Do not change its binary, model, task list, time budget, or sampling mid-run.
Report complete exercise/repetition blocks and preserve setup failures separately.
Three repeats of one exercise are still one exercise, not three independent tasks.

## Evidence before changes

Use `analyze.py` to inspect completed attempts and optional Keel session logs.
Its Markdown and JSON reports do not modify the run or call a model. They count
persisted tool calls, actual receipts, first explicit edit, tests, compactions,
and raw output size. They do not estimate tokens from character counts, infer
reasoning speed, attribute unlinked sessions, or classify a timeout's cause.

The first inspected sessions have many rewrites and few test calls. Aider's
affine-cipher log also contains a context error even though its process exited
zero. These observations suggest experiments; they do not yet justify changes
to Keel's production agent loop.

Before a confirmatory run, the runner needs:

- Read isolation that makes the source dataset's `.meta` solutions inaccessible.
  The current read-only `/` bind only prevents writes outside the workspace.
- Per-attempt archived solutions, full original-test output, timestamps, binary
  identity, dataset revision, request settings, and an explicit session link.
- Separate setup/scoring errors from valid attempts that failed to solve a task.
- A small end-to-end smoke task for each harness before a long run.
- A stop-time check after cooling and a fail-closed temperature sensor check.

## First controlled experiment: shorter test feedback

Hypothesis: long pytest tracebacks consume the small model's budget and reduce
useful corrections. First inspect full session outputs and existing pruning:
Keel already prunes tool results; raw logged output is not what the next model
request necessarily contains. A change is warranted only if this remains a
plausible bottleneck after inspecting the request, not merely the disk log.

Compare the existing prompt's `python3 -m pytest -q TEST` with
`python3 -m pytest -q --tb=short TEST` in a **new run directory**. Keep the same
binary, model, task set, context, generation settings, and wall-clock budget.
Original-test scoring is unchanged. Apply the prompt change to every harness
in the comparison; this tests feedback format, not a Keel architecture change.
Do not overlap inference with the existing single-slot server workload.

Use three repetitions per selected development task, rotate harness/variant
order, and include both a pilot task that has passed and tasks that have failed.
Reserve `list-ops`, `affine-cipher`, and `bowling` for development; `list-ops`
is the positive control from the earlier pilot. Reserve `pig-latin`,
`robot-name`, and `wordy` for confirmation before running either variant on
them. These are held out from this experiment, not claimed unseen by the model.
Report fully solved tasks as the primary outcome, per-task partial credit as
secondary, then time to first edit, verified corrections, and context failures.
Archive incomplete attempts and state actual coverage; do not substitute easier
tasks after seeing results.

Promote nothing from a single win. If the candidate improves development solve
rate without more setup/context failures, repeat it on tasks not used to tune
the prompt. Reserve those tasks before running either variant. With small
samples, report the paired task outcomes and uncertainty rather than a broad
claim about superiority. If no variant solves anything, first establish a
useful task/model combination; zero solves cannot select an architecture.

## Architecture decisions that must wait

- Add verification scheduling only if a controlled comparison shows better
  solutions than the same model and budget with the existing loop.
- Change context management only after request-level evidence identifies lost
  information or context failures; inspect existing compaction first.
- Keep swarm, tree search, speculation, cascade, and daemon work frozen. This
  pilot supplies no evidence that their cost helps the basic coding loop.
- Validate any chosen change with real repository repair tasks after Exercism.
  Passing isolated Python functions does not establish product readiness.
