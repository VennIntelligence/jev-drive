# <topic>: <title, <= 8 words>

status: live | concluded | superseded-by <topic>
decisions: <entry numbers in research/decisions.md>
index: <one clause <= 70 chars with the key number; INDEX.md shows it>
key: <the 8-12 files an agent opens first, repo-root paths; the file list below is generated from it>

**Question.** <one sentence>

**Conclusion.** <at most two sentences, numbers as in the decision entry, "(decisions N)">

**Next.** <live topics only>

**Read more.** <narrative docs, plans/ (live lanes), `git show <sha>:todos/<plan>.md` for removed plans>

<!-- files:begin -->
<!-- files:end -->

Layout: `scripts/` entry points (live), `lib/` code other topics import, `archive/` one-off code of a concluded
experiment, `results/` small result files, `figs/` figures, `plans/` live plan notes. Refresh the file list and
INDEX.md with `python tools/topic_index.py`.
