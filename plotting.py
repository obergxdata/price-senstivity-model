import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import shutil
from model import Product


def clear_charts_folder():
    """Clear all existing charts in the charts folder"""
    charts_path = Path("charts")
    if charts_path.exists():
        shutil.rmtree(charts_path)
    charts_path.mkdir(exist_ok=True)
    print("Charts folder cleared")


def plot_customer_position(customer, product, current_score, ref_price, wtp, value, rel_uncert):
    """
    Generate a chart showing customer position on pricing curve.
    Called from the score function when chart=True.
    """
    # Create charts directory
    Path("charts").mkdir(exist_ok=True)

    # Generate price range around reference price
    prices = np.linspace(ref_price * 0.5, ref_price * 2.0, 100)
    scores = []

    # Calculate scores across price range
    for price in prices:
        test_product = Product(
            price=float(price),
            sku=product.sku,
            category=product.category,
            features=product.features,
        )
        mem_1 = customer.access_memory(test_product)
        value = customer.l1_distance(
            item_1=test_product.features, item_2=customer.preferences
        )
        s = customer.score(mem_1=mem_1, product=test_product, value=value, chart=False)
        scores.append(s)

    plt.figure(figsize=(12, 8))

    # Plot the curve
    plt.plot(
        prices,
        scores,
        linewidth=2,
        color="#3498db",
        label=f"Customer (Sensitivity={customer.price_sensitivty})",
    )

    # Highlight current position
    plt.scatter(
        [product.price],
        [current_score],
        s=200,
        color="red",
        zorder=5,
        label=f"Current Price: ${product.price}",
    )
    plt.annotate(
        f"Score: {current_score:.1f}",
        xy=(product.price, current_score),
        xytext=(product.price + ref_price * 0.1, current_score + 10),
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
    )

    # Add reference price line
    plt.axvline(
        x=ref_price,
        color="green",
        linestyle="--",
        label=f"Reference Price: ${ref_price:.0f}",
        alpha=0.7,
        linewidth=2,
    )

    # Add WTP line
    plt.axvline(
        x=wtp,
        color="orange",
        linestyle="--",
        label=f"WTP: ${wtp:.0f}",
        alpha=0.7,
        linewidth=2,
    )

    # Add tolerance band around WTP (shaded region)
    tolerance_abs = wtp * rel_uncert
    tolerance_band_lower = wtp - tolerance_abs
    tolerance_band_upper = wtp + tolerance_abs
    plt.axvspan(
        tolerance_band_lower,
        tolerance_band_upper,
        alpha=0.15,
        color="purple",
        label=f"Tolerance Band (±{rel_uncert*100:.1f}%)",
    )

    # Add neutral score line
    plt.axhline(y=50, color="gray", linestyle=":", alpha=0.5, label="Neutral (50)")

    plt.xlabel("Price ($)", fontsize=12)
    plt.ylabel("Customer Score (0-100)", fontsize=12)
    plt.title(
        f"Customer Position: {product.sku}\n(Value={value:.2f}, Sensitivity={customer.price_sensitivty}, Price=${product.price}, Score={current_score:.1f})",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)

    plt.tight_layout()

    # Generate filename with counter
    charts_path = Path("charts")
    existing_charts = list(charts_path.glob("*.png"))
    chart_num = len(existing_charts) + 1
    filename = f"charts/{chart_num:02d}_{product.sku}.png"

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Chart saved to {filename}")
    plt.close()
