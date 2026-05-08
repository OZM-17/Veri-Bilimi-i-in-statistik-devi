"""
=============================================================
  A/B Testing ve Conversion Rate Analizi
  Proje 1 — Bernoulli/Binom Dağılımı, Z-test, Güven Aralıkları
=============================================================
Veri setleri:
  1A: Website Button A/B Test   (~300K kullanıcı)
  1B: Mobile App A/B Test       (~90K kullanıcı)
  1C: Marketing Campaign A/B    (~588 kullanıcı)
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
#  RENK PALETİ
# ──────────────────────────────────────────────
COLORS = {
    "A":       "#185FA5",
    "B":       "#0F6E56",
    "accent":  "#D85A30",
    "bg":      "#F8F7F4",
    "card":    "#FFFFFF",
    "muted":   "#888780",
    "success": "#3B6D11",
    "warn":    "#BA7517",
    "fail":    "#A32D2D",
}

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor":  COLORS["bg"],
    "figure.facecolor": COLORS["bg"],
    "axes.labelsize":  11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


# ══════════════════════════════════════════════
#  1. SENTETİK VERİ ÜRETİMİ
#     (Kaggle'dan indirilmiş gerçek veri ile
#      aynı parametreler kullanılmıştır)
# ══════════════════════════════════════════════

np.random.seed(42)

def create_dataset_1a(n=300000):
    """Website Button A/B Test — Kaggle parametreleri."""
    nA, nB = 147239, 147234
    pA, pB = 0.1188, 0.1173          # gerçek CR değerleri
    groupA = pd.DataFrame({
        "user_id":      range(1, nA + 1),
        "group":        "control",
        "landing_page": "old_page",
        "converted":    np.random.binomial(1, pA, nA),
    })
    groupB = pd.DataFrame({
        "user_id":      range(nA + 1, nA + nB + 1),
        "group":        "treatment",
        "landing_page": "new_page",
        "converted":    np.random.binomial(1, pB, nB),
    })
    df = pd.concat([groupA, groupB], ignore_index=True)
    df["timestamp"] = pd.date_range("2023-01-01", periods=len(df), freq="min")
    return df

def create_dataset_1b():
    """Mobile App A/B Test — Cookie Cats parametreleri."""
    nA, nB = 44700, 45489
    ret7A, ret7B = 0.4482, 0.4423    # 7-gün retention
    groupA = pd.DataFrame({
        "user_id":       range(1, nA + 1),
        "version":       "gate_30",
        "sum_gamerounds": np.random.poisson(51, nA),
        "retention_1":   np.random.binomial(1, 0.55, nA),
        "retention_7":   np.random.binomial(1, ret7A, nA),
    })
    groupB = pd.DataFrame({
        "user_id":       range(nA + 1, nA + nB + 1),
        "version":       "gate_40",
        "sum_gamerounds": np.random.poisson(49, nB),
        "retention_1":   np.random.binomial(1, 0.53, nB),
        "retention_7":   np.random.binomial(1, ret7B, nB),
    })
    return pd.concat([groupA, groupB], ignore_index=True)

def create_dataset_1c():
    """Marketing Campaign A/B Test — Kaggle parametreleri."""
    nA, nB = 235, 353
    pA, pB = 0.0979, 0.1445
    groupA = pd.DataFrame({
        "user_id":       range(1, nA + 1),
        "test_group":    "psa",
        "made_purchase": np.random.binomial(1, pA, nA),
        "most_ads_day":  np.random.choice(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], nA),
        "most_ads_hour": np.random.randint(0, 24, nA),
    })
    groupB = pd.DataFrame({
        "user_id":       range(nA + 1, nA + nB + 1),
        "test_group":    "ad",
        "made_purchase": np.random.binomial(1, pB, nB),
        "most_ads_day":  np.random.choice(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], nB),
        "most_ads_hour": np.random.randint(0, 24, nB),
    })
    return pd.concat([groupA, groupB], ignore_index=True)


# ══════════════════════════════════════════════
#  2. İSTATİSTİKSEL ARAÇLAR
# ══════════════════════════════════════════════

def conversion_rate(df, group_col, metric_col, groupA_val, groupB_val):
    """İki grup için conversion rate ve temel istatistikler."""
    grA = df[df[group_col] == groupA_val]
    grB = df[df[group_col] == groupB_val]
    nA, nB = len(grA), len(grB)
    kA, kB = grA[metric_col].sum(), grB[metric_col].sum()
    pA, pB = kA / nA, kB / nB
    return dict(
        nA=nA, nB=nB, kA=kA, kB=kB,
        pA=pA, pB=pB,
        liftPct=round((pB - pA) / pA * 100, 4),
    )

def binom_stats(p, n):
    """Bernoulli → Binom istatistikleri."""
    mean  = n * p                   # E[X] = n·p
    var   = n * p * (1 - p)        # Var(X) = n·p·(1-p)
    std   = np.sqrt(var)
    return mean, var, std

def confidence_interval(p, n, alpha=0.05):
    """Wald yöntemi ile güven aralığı."""
    z = stats.norm.ppf(1 - alpha / 2)   # z_α/2
    margin = z * np.sqrt(p * (1 - p) / n)
    return p - margin, p + margin

def z_test_two_proportions(kA, nA, kB, nB):
    """
    H₀: p_A = p_B
    H₁: p_A ≠ p_B  (iki yönlü)
    Pooled Z-test.
    """
    pA, pB = kA / nA, kB / nB
    p_pool = (kA + kB) / (nA + nB)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/nA + 1/nB))
    z = (pA - pB) / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_val

def power_analysis(pA, pB, alpha=0.05, power=0.80):
    """Minimum gerekli örneklem büyüklüğü."""
    effect = abs(pB - pA)
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta  = stats.norm.ppf(power)
    p_bar   = (pA + pB) / 2
    n = ((z_alpha + z_beta)**2 * 2 * p_bar * (1 - p_bar)) / effect**2
    return int(np.ceil(n))


# ══════════════════════════════════════════════
#  3. VERİ KEŞFİ VE YAZDIRMA
# ══════════════════════════════════════════════

SEPARATOR = "=" * 60

def print_analysis(title, stats_d, z, p_val, ciA, ciB, meanA, varA, stdA, meanB, varB, stdB):
    """Konsol çıktısı."""
    sig = p_val < 0.05
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)

    print(f"\n{'GRUP':>10} {'n':>8} {'Converted':>10} {'Rate':>8}")
    print("-" * 42)
    print(f"{'A':>10} {stats_d['nA']:>8,} {stats_d['kA']:>10,} {stats_d['pA']:>8.4%}")
    print(f"{'B':>10} {stats_d['nB']:>8,} {stats_d['kB']:>10,} {stats_d['pB']:>8.4%}")

    print(f"\n📊  Binom Dağılımı — Grup A (n={stats_d['nA']:,}, p={stats_d['pA']:.4f})")
    print(f"    E[X] = {meanA:,.2f}  |  Var(X) = {varA:,.2f}  |  σ = {stdA:,.2f}")
    print(f"\n📊  Binom Dağılımı — Grup B (n={stats_d['nB']:,}, p={stats_d['pB']:.4f})")
    print(f"    E[X] = {meanB:,.2f}  |  Var(X) = {varB:,.2f}  |  σ = {stdB:,.2f}")

    print(f"\n🔬  Hipotez Testi  (H₀: p_A = p_B)")
    print(f"    Z-istatistiği : {z:+.4f}")
    print(f"    p-değeri      : {p_val:.6f}")
    print(f"    Kritik değer  : ±1.96  (α = 0.05)")
    print(f"    Sonuç         : {'H₀ REDDEDİLDİ ✓' if sig else 'H₀ REDDEDİLEMEDİ ✗'}")

    print(f"\n📏  Güven Aralıkları (95%)")
    print(f"    Grup A: [{ciA[0]:.4%} — {ciA[1]:.4%}]")
    print(f"    Grup B: [{ciB[0]:.4%} — {ciB[1]:.4%}]")

    print(f"\n💡  Lift      : {stats_d['liftPct']:+.2f}%")
    if sig:
        winner = "B" if stats_d["pB"] > stats_d["pA"] else "A"
        print(f"    Öneri    : Grup {winner} istatistiksel olarak daha başarılı → Değişiklik önerilir.")
    else:
        print(f"    Öneri    : Anlamlı fark yok → Mevcut versiyonu koruyun veya daha fazla veri toplayın.")


# ══════════════════════════════════════════════
#  4. GÖRSELLEŞTİRME
# ══════════════════════════════════════════════

def binom_pmf_array(n, p, radius=60):
    """Binom PMF dizisi."""
    mu = int(round(n * p))
    k_vals = np.arange(max(0, mu - radius), min(n, mu + radius) + 1)
    pmf = stats.binom.pmf(k_vals, n, p)
    return k_vals, pmf

def plot_dataset(ax_bar, ax_ci, ax_binom, title, stats_d, z, p_val, ciA, ciB):
    sig = p_val < 0.05
    label_color = COLORS["success"] if sig else COLORS["fail"]
    groups = ["Grup A", "Grup B"]
    rates  = [stats_d["pA"] * 100, stats_d["pB"] * 100]

    # — Bar chart —
    bars = ax_bar.bar(groups, rates, color=[COLORS["A"], COLORS["B"]],
                      width=0.45, zorder=3, edgecolor="white", linewidth=1.5)
    for bar, rate in zip(bars, rates):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f"{rate:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="500")
    ax_bar.set_title(f"{title}\nConversion Rate", fontsize=10, fontweight="500", pad=8)
    ax_bar.set_ylabel("Rate (%)", fontsize=9)
    ax_bar.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax_bar.set_ylim(0, max(rates) * 1.25)
    ax_bar.grid(axis="y", alpha=0.3, zorder=0)

    # sonuç etiketi
    verdict = f"p = {p_val:.4f}  →  {'Anlamlı ✓' if sig else 'Anlamsız ✗'}"
    ax_bar.text(0.5, 0.97, verdict, transform=ax_bar.transAxes,
                ha="center", va="top", fontsize=9, color=label_color,
                bbox=dict(boxstyle="round,pad=0.3", fc=COLORS["card"], ec=label_color, lw=1.2))

    # — CI chart —
    y = [1, 0]
    for i, (p_val_ci, ci, col) in enumerate([(stats_d["pA"], ciA, COLORS["A"]),
                                               (stats_d["pB"], ciB, COLORS["B"])]):
        ax_ci.errorbar(p_val_ci * 100, y[i],
                       xerr=[[( p_val_ci - ci[0]) * 100],
                              [(ci[1] - p_val_ci) * 100]],
                       fmt="o", color=col, capsize=8, capthick=2,
                       elinewidth=2.5, markersize=9, zorder=4)
        ax_ci.text(p_val_ci * 100, y[i] + 0.15, f"{p_val_ci:.3%}",
                   ha="center", fontsize=9, color=col, fontweight="500")
    ax_ci.set_yticks([0, 1])
    ax_ci.set_yticklabels(["Grup B", "Grup A"])
    ax_ci.set_title("95% Güven Aralıkları", fontsize=10, fontweight="500", pad=8)
    ax_ci.set_xlabel("Rate (%)", fontsize=9)
    ax_ci.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax_ci.grid(axis="x", alpha=0.3)
    ax_ci.set_ylim(-0.5, 1.5)

    # — Binom dağılımı —
    nSample = min(stats_d["nA"], 5000)
    kA, pmfA = binom_pmf_array(nSample, stats_d["pA"], radius=50)
    kB, pmfB = binom_pmf_array(nSample, stats_d["pB"], radius=50)
    ax_binom.fill_between(kA, pmfA * 100, alpha=0.4, color=COLORS["A"], label="Grup A")
    ax_binom.fill_between(kB, pmfB * 100, alpha=0.4, color=COLORS["B"], label="Grup B")
    ax_binom.plot(kA, pmfA * 100, color=COLORS["A"], linewidth=2)
    ax_binom.plot(kB, pmfB * 100, color=COLORS["B"], linewidth=2)
    ax_binom.axvline(stats_d["pA"] * nSample, color=COLORS["A"], linestyle="--", alpha=0.7)
    ax_binom.axvline(stats_d["pB"] * nSample, color=COLORS["B"], linestyle="--", alpha=0.7)
    ax_binom.set_title(f"Binom Dağılımı (n={nSample:,})", fontsize=10, fontweight="500", pad=8)
    ax_binom.set_xlabel("Başarı Sayısı (k)", fontsize=9)
    ax_binom.set_ylabel("P(X=k) ×100", fontsize=9)
    ax_binom.legend(fontsize=9, framealpha=0.6)
    ax_binom.grid(axis="y", alpha=0.3)


def plot_all(results):
    """3 veri seti × 3 grafik = 3×3 ızgara."""
    fig = plt.figure(figsize=(18, 15), facecolor=COLORS["bg"])
    fig.suptitle("A/B Testing ve Conversion Rate Analizi", fontsize=16,
                 fontweight="500", y=0.98)

    titles = [
        "Dataset 1A — Website Button",
        "Dataset 1B — Mobile App (7-gün Retention)",
        "Dataset 1C — Marketing Campaign",
    ]

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35,
                           left=0.06, right=0.97, top=0.94, bottom=0.04)

    for row, (r, title) in enumerate(zip(results, titles)):
        ax_bar   = fig.add_subplot(gs[row, 0])
        ax_ci    = fig.add_subplot(gs[row, 1])
        ax_binom = fig.add_subplot(gs[row, 2])
        plot_dataset(ax_bar, ax_ci, ax_binom, title, **r)

    plt.savefig("ab_testing_sonuclar.png", dpi=150, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.show()
    print("\n✅  Grafik 'ab_testing_sonuclar.png' olarak kaydedildi.")


# ══════════════════════════════════════════════
#  5. ANA AKIŞ
# ══════════════════════════════════════════════

def analyze(df, group_col, metric_col, groupA_val, groupB_val, title):
    st = conversion_rate(df, group_col, metric_col, groupA_val, groupB_val)
    meanA, varA, stdA = binom_stats(st["pA"], st["nA"])
    meanB, varB, stdB = binom_stats(st["pB"], st["nB"])
    z, p_val = z_test_two_proportions(st["kA"], st["nA"], st["kB"], st["nB"])
    ciA = confidence_interval(st["pA"], st["nA"])
    ciB = confidence_interval(st["pB"], st["nB"])
    n_req = power_analysis(st["pA"], st["pB"])

    print_analysis(title, st, z, p_val, ciA, ciB, meanA, varA, stdA, meanB, varB, stdB)
    print(f"    Güç (80%) için gereken min örneklem: {n_req:,} / grup")

    return dict(stats_d=st, z=z, p_val=p_val, ciA=ciA, ciB=ciB)


def main():
    print("\n" + SEPARATOR)
    print("  VERİ SETLERİ OLUŞTURULUYOR…")
    print(SEPARATOR)
    df1a = create_dataset_1a()
    df1b = create_dataset_1b()
    df1c = create_dataset_1c()
    print(f"  1A: {len(df1a):,} satır  |  1B: {len(df1b):,} satır  |  1C: {len(df1c):,} satır")

    results = [
        analyze(df1a, "group",    "converted",    "control",  "treatment", "Dataset 1A — Website Button"),
        analyze(df1b, "version",  "retention_7",  "gate_30",  "gate_40",   "Dataset 1B — Mobile App (7-gün Retention)"),
        analyze(df1c, "test_group","made_purchase","psa",      "ad",        "Dataset 1C — Marketing Campaign"),
    ]

    plot_all(results)

    # ── Özet tablo ──────────────────────────────
    print(f"\n{SEPARATOR}")
    print("  GENEL ÖZET")
    print(SEPARATOR)
    print(f"\n{'Veri Seti':<28} {'p_A':>8} {'p_B':>8} {'Lift':>8} {'p-değeri':>10} {'Karar':>20}")
    print("-" * 86)
    names = ["1A Website", "1B Mobile App", "1C Marketing"]
    for name, r in zip(names, results):
        st = r["stats_d"]
        sig = r["p_val"] < 0.05
        decision = ("B daha iyi → Değiştir" if sig and st["pB"] > st["pA"] else
                    "A daha iyi → Koru"     if sig and st["pA"] > st["pB"] else
                    "Anlamlı fark yok")
        print(f"{name:<28} {st['pA']:>8.4%} {st['pB']:>8.4%} {st['liftPct']:>+7.2f}% "
              f"{r['p_val']:>10.4f} {decision:>20}")
    print()


if __name__ == "__main__":
    main()
