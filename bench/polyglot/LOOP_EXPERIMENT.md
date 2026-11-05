# Next change: test after editing, fix the reported failure

The 2026-11-05 pilot is exploratory, not a harness ranking. At 41 saved
attempts, the 13 completed three-harness blocks had 2 Keel solves, 1 OpenCode
solve and 1 Aider solve. Keel timed out on 13 of its 14 saved attempts.
The useful question is whether Keel can use those five minutes better.

## Three actual failures

Inspected the first-repeat persisted Keel sessions, not just final scores:

| Task | What happened | Concrete target |
|---|---|---|
| affine-cipher | Edited at 82s, tested at 86s: 10 failed / 6 passed. Edited at 199s, tested at 203s: the same failures. The second write changed comments but left executable logic unchanged. | Correct a reported failure instead of restating the implementation. |
| bowling | Four full-file writes at 73s, 134s, 198s, 239s. First pytest at 245s: 20 failed / 11 passed. | Get feedback after the first implementation, while time remains to fix it. |
| food-chain | Tested at 57s and 92s; second result was 9 failed / 1 passed. Three more writes at 188s, 231s, 280s without another persisted pytest call. Final original-test score: 10 failed. | Test subsequent edits and catch regressions before stopping. |

Source sessions were archived in the pilot's
`analysis/20261105-150844/sessions/` directory, respectively
`session-1793876001278.jsonl`, `session-1793877294136.jsonl`, and
`session-1793878636067.jsonl`. Times are measured from their user event.
The affine result matters: simply increasing test frequency will not fix all
three failures. The model also needs to act on what the tests say.

## Small controlled comparison

`compare_loop.py` runs only Keel. It changes the user prompt, not Rust code:
the candidate asks for a test after every substantive edit, a correction
directed at a reported failure, and a check of the latest solution.
The default prompt is the unchanged pilot prompt. Full prompt text, binary
hash, input hashes, server defaults and the ordered schedule are recorded.
Both variants use the isolated runner merged in PR #61, the same pinned
binary, model, task inputs and 300-second budget. Original-test scoring and
pytest formatting remain unchanged. Actual request sampling is not captured;
this compares the same Keel defaults rather than claiming matched random seeds.

Run three task pairs as an initial screen, then repeat to three repetitions
if the schedule permits. Variant order alternates across tasks and repeats.
The script preserves all attempts and can resume without rerunning completed
attempts. It refuses changed settings or previous infrastructure failures.
A small real-model smoke must pass before the comparison. The pilot must
finish before this script makes any model request. It stops before quiet hours.

```sh
python3 bench/polyglot/compare_loop.py \
  --exercises ~/Projects/ref/polyglot-benchmark/python/exercises/practice \
  --out ~/.cache/keel-bench/runs/2026-11-05/test-after-edit \
  --keel-bin ~/.cache/keel-bench/keel-53eee69 \
  --wait-for-pilot ~/.cache/keel-bench/runs/2026-11-05/results.jsonl \
  --repeat 3 --timeout 300 --stop-at 19:30
```

Read `report.md` for final original-test results, time of first persisted
pytest receipt, edits and tests. Inspect `VARIANT/attempts/keel/TASK.REP/`
for each candidate, command, original scoring output and linked session.
Primary outcome is fully solved tasks in completed pairs; partial test counts
are secondary. Repetitions are not independent tasks. A faster first test or
more test calls alone is not an improvement.

If the candidate repeatedly improves correctness without infrastructure/context
failures, confirm on the reserved `pig-latin`, `robot-name`, `wordy` tasks,
then real repository bug fixes. If it merely adds tests or produces mixed
results, do not change the production loop. If the model ignores the guidance,
report that explicitly; it does not establish whether enforced scheduling works.

This experiment takes priority over the earlier traceback-format proposal in
DECISIONS.md. No swarm, tree, cascade or other architecture expansion follows
from this pilot.
