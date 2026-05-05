from pathlib import Path
import textwrap
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def wrap_line(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def add_box(
    ax,
    center,
    width,
    height,
    title,
    body,
    fc,
    ec="#1f2937",
    lw=2.5,
    title_size=14,
    body_size=12,
):
    x = center[0] - width / 2
    y = center[1] - height / 2
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(patch)

    ax.text(
        center[0],
        center[1] + height * 0.17,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        color="#0f172a",
        fontweight="bold",
        zorder=3,
    )

    ax.text(
        center[0],
        center[1] - height * 0.06,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        color="#0f172a",
        fontweight="semibold",
        linespacing=1.32,
        zorder=3,
    )


def add_arrow(ax, p1, p2, color="#334155", lw=2.4):
    arrow = FancyArrowPatch(
        p1,
        p2,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=lw,
        color=color,
        connectionstyle="arc3,rad=0.0",
        zorder=1,
    )
    ax.add_patch(arrow)


def main():
    fig, ax = plt.subplots(figsize=(15, 19), dpi=260)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Soft poster-friendly background tint
    fig.patch.set_facecolor("#f8fbff")
    ax.set_facecolor("#f8fbff")

    # Title
    ax.text(
        0.5,
        0.965,
        "CIRRUS System Block Diagram (Poster Version)",
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
        color="#0b3b5a",
    )
    ax.text(
        0.5,
        0.935,
        "Top-Down Architecture: Edge Compute -> Subsystems -> Sensor Stacks -> Outputs",
        ha="center",
        va="center",
        fontsize=15,
        color="#1e3a5f",
    )

    # Core blocks
    add_box(
        ax,
        center=(0.5, 0.84),
        width=0.52,
        height=0.11,
        title="EDGE COMPUTE",
        body=wrap_line(
            "Data Ingestion | Feature Processing | Model Inference | Rule Engine",
            62,
        ),
        fc="#dbeafe",
        ec="#1d4ed8",
        title_size=17,
        body_size=14,
    )

    add_box(
        ax,
        center=(0.29, 0.65),
        width=0.42,
        height=0.12,
        title="SUBSYSTEM A",
        body="Wet Bath Lifecycle Model\nGradient Boosting Regressor",
        fc="#cffafe",
        ec="#0891b2",
        title_size=16,
        body_size=13.5,
    )
    add_box(
        ax,
        center=(0.71, 0.65),
        width=0.42,
        height=0.12,
        title="SUBSYSTEM B",
        body="Heat Cascade Routing Model\nRandom Forest Classifier",
        fc="#ffedd5",
        ec="#ea580c",
        title_size=16,
        body_size=13.5,
    )

    add_box(
        ax,
        center=(0.29, 0.43),
        width=0.42,
        height=0.16,
        title="A SENSOR STACK",
        body=(
            "pH | ORP | Conductivity | Ion-selective\n"
            "Turbidity | Bath Temp | Recirc Flow | Level"
        ),
        fc="#ecfeff",
        ec="#0e7490",
        title_size=15,
        body_size=12.8,
    )
    add_box(
        ax,
        center=(0.71, 0.43),
        width=0.42,
        height=0.16,
        title="B SENSOR STACK",
        body=(
            "Exhaust Temp | Flow | Pressure\n"
            "HX In-Out Temp & Flow\n"
            "DI/HVAC Sensors | ORC Temp/Pressure"
        ),
        fc="#fff7ed",
        ec="#c2410c",
        title_size=15,
        body_size=12.4,
    )

    add_box(
        ax,
        center=(0.29, 0.24),
        width=0.35,
        height=0.10,
        title="A OUTPUTS",
        body="CONTINUE | TOP-OFF | DUMP",
        fc="#dcfce7",
        ec="#15803d",
        title_size=14,
        body_size=12.8,
    )
    add_box(
        ax,
        center=(0.71, 0.24),
        width=0.35,
        height=0.10,
        title="B OUTPUTS",
        body="ORC | DI_PREHEAT | HVAC | BLEND",
        fc="#fef3c7",
        ec="#b45309",
        title_size=14,
        body_size=12.8,
    )

    add_box(
        ax,
        center=(0.5, 0.09),
        width=0.56,
        height=0.11,
        title="UNIFIED CIRRUS DASHBOARD + KPI LAYER",
        body=wrap_line(
            "Chemical Reduction | kWh Recovery | CO2e Reduction | Safety Events",
            58,
        ),
        fc="#ede9fe",
        ec="#6d28d9",
        title_size=15,
        body_size=13,
    )

    # Arrows (top-down)
    add_arrow(ax, (0.5, 0.785), (0.29, 0.71), color="#1d4ed8")
    add_arrow(ax, (0.5, 0.785), (0.71, 0.71), color="#1d4ed8")

    add_arrow(ax, (0.29, 0.59), (0.29, 0.505), color="#0891b2")
    add_arrow(ax, (0.71, 0.59), (0.71, 0.505), color="#ea580c")

    add_arrow(ax, (0.29, 0.355), (0.29, 0.29), color="#0e7490")
    add_arrow(ax, (0.71, 0.355), (0.71, 0.29), color="#c2410c")

    add_arrow(ax, (0.29, 0.19), (0.46, 0.13), color="#334155")
    add_arrow(ax, (0.71, 0.19), (0.54, 0.13), color="#334155")

    out_path = OUTPUTS_DIR / "cirrus_block_diagram_poster.png"
    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
