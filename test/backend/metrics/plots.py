from pathlib import Path
import matplotlib.pyplot as plt

def plot_latency(df, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    ax = df.plot(
        x="query",
        y="avg_latency_ms",
        kind="bar",
        legend=False,
        title="Average Query Latency"
    )
    ax.set_ylabel("Latency (ms)")
    fig = ax.get_figure()
    fig.tight_layout()

    fig.savefig(out_dir / "avg_latency.png")
    plt.close(fig)


def plot_precision(df, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    ax = df.plot(
        x="query",
        y="precision@5",
        kind="bar",
        legend=False,
        title="Precision@5 per Query"
    )
    ax.set_ylabel("Precision@5")
    ax.set_ylim(0, 1)

    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(out_dir / "precision_at_5.png")
    plt.close(fig)
