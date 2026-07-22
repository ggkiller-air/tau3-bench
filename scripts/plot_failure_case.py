"""Figure: Telecom strat90 failure analysis (two panels).

Usage: python scripts/plot_failure_case.py
Writes figs/failure_case.pdf + figs/failure_case.png
       (+ /root/Project/SchemaFlex/FailureCase.png preview copy)

Values are integer percentages produced by scripts/failure_taxonomy.py from
reports/{react,ours}_strat90_failure.csv (stacked rows balanced to 100 with
the largest-remainder method). They are plotted verbatim — no recomputation.

Palette validated with the dataviz six-checks validator (light mode):
  panel a: #2a78d6 (SchemaFlex) / #008300 (SLM-ReAct) — categorical, all pass
  panel b: success #2a78d6 + ordinal red ramp (Final -> Identify,
           light->dark = failed at an earlier stage) — ordinal all pass; the
           2.23:1 light end is relieved by direct labels and 2px surface gaps.
"""

from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

BLUE = "#2a78d6"
GREEN = "#008300"
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK2 = "#52514e"

STAGES = ["G1\nIdentify", "G2\nRepair", "G3\nVerify", "G4\nSuccess"]

# recorded percentages — plotted exactly as-is
FUNNEL = {
    "SLM-ReAct": [57, 36, 32, 32],
    "SchemaFlex": [87, 82, 80, 78],
}
# stack order: success, then failures by how far they got (furthest first)
OUTCOME_NAMES = ["Success", "Final", "Verify", "Repair", "Identify"]
OUTCOME_COLORS = [BLUE, "#ee9480", "#e05a44", "#b93227", "#7e1d16"]
BREAKDOWN = {
    "SLM-ReAct": [32, 0, 3, 21, 44],
    "SchemaFlex": [78, 2, 2, 5, 13],
}
ROW_ORDER = ["SchemaFlex", "SLM-ReAct"]  # bottom-to-top

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK, "ytick.color": INK,
    "text.color": INK, "axes.labelcolor": INK,
    "pdf.fonttype": 42,
})

fig, (ax_a, ax_b) = plt.subplots(
    1, 2, figsize=(7.0, 1.85), dpi=200,
    gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.17},
)
fig.patch.set_facecolor(SURFACE)

# ---------------- panel (a): cumulative funnel ---------------- #
x = range(len(STAGES))
w = 0.36
for off, (name, color) in [(-w / 2, ("SLM-ReAct", GREEN)), (w / 2, ("SchemaFlex", BLUE))]:
    vals = FUNNEL[name]
    bars = ax_a.bar([i + off for i in x], vals, width=w - 0.03, color=color,
                    zorder=3, label=name)
    for b, v in zip(bars, vals):
        ax_a.text(b.get_x() + b.get_width() / 2, v + 2, f"{v}", ha="center",
                  va="bottom", fontsize=6.8, color=color, fontweight="bold")
ax_a.set_ylim(0, 112)
ax_a.set_yticks([0, 25, 50, 75, 100])
ax_a.tick_params(axis="y", labelsize=6.5, length=2, pad=1)
ax_a.set_ylabel("Cumulative reach (% of episodes)", fontsize=6.5, labelpad=1)
ax_a.set_xticks(list(x))
ax_a.set_xticklabels(STAGES, fontsize=6.5)
ax_a.tick_params(axis="x", length=0, pad=2)
ax_a.grid(axis="y", color="#e6e5e1", linewidth=0.6, zorder=0)
ax_a.spines[["top", "right"]].set_visible(False)
ax_a.legend(frameon=False, fontsize=6.5, loc="lower left", ncol=2,
            bbox_to_anchor=(-0.02, 0.97), handlelength=1.0, handleheight=1.0,
            borderaxespad=0.0, columnspacing=1.0, handletextpad=0.4)
ax_a.set_title("(a) Cumulative reach across workflow stages", fontsize=7.5, pad=15)

# ---------------- panel (b): outcome breakdown ---------------- #
BAR_H = 0.52
LABEL_GAP = 13  # min horizontal spacing (in %-units) between narrow labels
for yi, name in enumerate(ROW_ORDER):
    vals = BREAKDOWN[name]
    left = 0.0
    narrow = []  # (segment-center x, text) for outside labels
    for v, cat, color in zip(vals, OUTCOME_NAMES, OUTCOME_COLORS):
        if v == 0:
            continue
        ax_b.barh(yi, v, left=left, height=BAR_H, color=color,
                  edgecolor=SURFACE, linewidth=1.4, zorder=3)
        cx = left + v / 2
        if v >= 10:  # category + % inside, two lines
            lum = int(color[1:3], 16) * 0.3 + int(color[3:5], 16) * 0.6
            ax_b.text(cx, yi, f"{cat}\n{v}%", ha="center", va="center",
                      fontsize=6, linespacing=1.25,
                      color=("#ffffff" if lum < 120 else INK))
        else:
            narrow.append((cx, f"{cat} {v}%"))
        left += v
    # narrow labels: evenly spread around their mean, angled leader lines
    if narrow:
        mean = sum(cx for cx, _ in narrow) / len(narrow)
        lx = [mean + (i - (len(narrow) - 1) / 2) * LABEL_GAP for i in range(len(narrow))]
        shift = max(0, lx[-1] - 96) or min(0, lx[0] - 4)
        for (cx, txt), tx in zip(narrow, [v - shift for v in lx]):
            ax_b.annotate(
                txt, xy=(cx, yi + BAR_H / 2), xytext=(tx, yi + 0.42),
                ha="center", va="bottom", fontsize=5.8, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.5,
                                shrinkA=0.5, shrinkB=1))
ax_b.set_xlim(0, 100)
ax_b.set_ylim(-0.45, len(ROW_ORDER) - 0.15)
ax_b.set_yticks(range(len(ROW_ORDER)))
ax_b.set_yticklabels(ROW_ORDER, fontsize=6.5)
ax_b.tick_params(axis="y", length=0, pad=2)
ax_b.set_xticks([0, 20, 40, 60, 80, 100])
ax_b.tick_params(axis="x", labelsize=6.5, length=2, pad=1)
ax_b.set_xlabel("Fraction of episodes (%)", fontsize=6.5, labelpad=1)
ax_b.spines[["top", "right", "left"]].set_visible(False)
ax_b.set_title("(b) Outcome breakdown by terminal stage", fontsize=7.5, pad=15)

fig.subplots_adjust(left=0.065, right=0.985, top=0.80, bottom=0.17)
out = ROOT / "figs"
out.mkdir(exist_ok=True)
fig.savefig(out / "failure_case.pdf")
fig.savefig(out / "failure_case.png", dpi=300)
fig.savefig("/root/Project/SchemaFlex/FailureCase.png", dpi=300)
print("funnel:", FUNNEL)
print("breakdown:", BREAKDOWN)
print("wrote", out / "failure_case.pdf", "and .png + preview copy")
