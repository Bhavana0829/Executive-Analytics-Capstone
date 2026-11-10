"""Executive analytics capstone: customer churn.
Reads data/cleaned_customer_churn.csv and writes insights, charts, executive_brief.pdf and dashboard.html.
Run: python3 build.py
"""
import os, json
from math import comb
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle

ROOT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f"{ROOT}/charts", exist_ok=True)
df = pd.read_csv(f"{ROOT}/data/cleaned_customer_churn.csv")
df["c"] = (df.Churn == "Yes").astype(int)

N, K = len(df), int(df.c.sum())
MRR = df.MonthlyCharges.sum()
MRR_LOST = df.loc[df.c == 1, "MonthlyCharges"].sum()

# ---------- data quality ----------
quality = {
    "rows": N,
    "missing_values": int(df.isna().sum().sum()),
    "duplicate_ids": int(df.CustomerID.duplicated().sum()),
    "totalcharges_equals_monthly_x_tenure": bool(
        ((df.TotalCharges - df.MonthlyCharges * df.TenureMonths).abs() < 0.01).all()),
}

def p_tail(n, k):
    """Exact one-sided probability that a random group of n customers contains >= k churners."""
    return sum(comb(K, i) * comb(N - K, n - i) for i in range(k, min(n, K) + 1)) / comb(N, n)

def seg(series):
    g = df.groupby(series, observed=True)
    out = pd.DataFrame({"n": g.size(), "churned": g.c.sum(), "rate": g.c.mean(),
                        "mrr_lost": g.apply(lambda x: x.loc[x.c == 1, "MonthlyCharges"].sum(), include_groups=False)})
    return out

df["TicketBand"] = pd.cut(df.SupportTickets, [-1, 2, 4, 99], labels=["0-2", "3-4", "5+"])
df["TenureBand"] = pd.cut(df.TenureMonths, [0, 6, 12, 24, 999], labels=["1-6", "7-12", "13-24", "25+"])
df["PayGroup"] = df.PaymentMethod

S = {
    "contract": seg(df.ContractType).loc[["Month-to-Month", "One Year", "Two Year"]],
    "tenure": seg(df.TenureBand),
    "tickets": seg(df.TicketBand),
    "plan": seg(df.SubscriptionType).loc[["Basic", "Pro", "Enterprise"]],
    "payment": seg(df.PaymentMethod).loc[["UPI", "Debit Card", "Credit Card", "Bank Transfer"]],
}
for k, v in S.items():
    v["rate"] = v.churned / v.n

# statistics used in the brief
m2m = S["contract"].loc["Month-to-Month"]
stats = {
    "churn_rate": K / N,
    "mrr": MRR, "mrr_lost": MRR_LOST, "mrr_lost_pct": MRR_LOST / MRR,
    "avg_mrr_churned": MRR_LOST / K,
    "avg_mrr_retained": df.loc[df.c == 0, "MonthlyCharges"].mean(),
    "tenure_churned": df.loc[df.c == 1, "TenureMonths"].mean(),
    "tenure_retained": df.loc[df.c == 0, "TenureMonths"].mean(),
    "tickets_churned": df.loc[df.c == 1, "SupportTickets"].mean(),
    "tickets_retained": df.loc[df.c == 0, "SupportTickets"].mean(),
    "p_contract": p_tail(7, 7),
    "p_tickets3plus": p_tail(7, 7),
    "p_tenure_le12": p_tail(6, 6),
    "p_basic": p_tail(7, 5),
    "p_upi_debit": p_tail(6, 5),
    # exact 95% (two-sided) bounds: 7/7 lower bound, 0/8 upper bound
    "m2m_lower95": 0.025 ** (1 / 7),
    "contract_upper95": 1 - 0.025 ** (1 / 8),
}
# overlap check: do month-to-month and 3+ tickets identify the same customers?
stats["m2m_equals_3plus_tickets"] = bool(
    set(df.loc[df.ContractType == "Month-to-Month", "CustomerID"]) ==
    set(df.loc[df.SupportTickets >= 3, "CustomerID"]))

# ---------- impact scenarios (what-if, not a forecast) ----------
scen = []
for label, save in [("Low", 0.10), ("Mid", 0.20), ("High", 0.30)]:
    saved = K * save
    rec = MRR_LOST * save
    scen.append({"scenario": label, "save_rate": save, "customers_saved": saved,
                 "churn_after": (K - saved) / N, "mrr_recovered": rec,
                 "annualised": rec * 12, "pct_of_mrr": rec / MRR})
scen_df = pd.DataFrame(scen)

# ---------- insights table ----------
def row(i, title, seg_df, note, p, conf):
    return {"#": i, "insight": title, "evidence": note, "exact_p_one_sided": round(p, 4), "confidence": conf}

insights = [
    row(1, "Contract type separates churners from stayers",
        None, "Month-to-month 7/7 churned; one-year 0/4; two-year 0/4", stats["p_contract"], "High (sample), but confounded"),
    row(2, "Customers churn early",
        None, "Tenure up to 12 months: 6/6 churned; 25+ months: 0/4", stats["p_tenure_le12"], "Medium"),
    row(3, "Heavy support contact precedes churn",
        None, "0-2 tickets: 0/8 churned; 3+ tickets: 7/7 churned", stats["p_tickets3plus"], "High (sample), but identical to contract split"),
    row(4, "Churn hits low-priced plans; revenue loss is smaller than customer loss",
        None, f"Basic 5/7, Pro 2/4, Enterprise 0/4; churners pay ${stats['avg_mrr_churned']:.2f}/mo vs ${stats['avg_mrr_retained']:.2f}", stats["p_basic"], "Medium"),
    row(5, "UPI and debit-card payers churn more (weak signal)",
        None, "UPI 3/4, debit 2/2, credit card 2/6, bank transfer 0/3", stats["p_upi_debit"], "Low; likely a proxy for contract/plan"),
]
pd.DataFrame(insights).to_csv(f"{ROOT}/insights.csv", index=False)
scen_df.to_csv(f"{ROOT}/impact_scenarios.csv", index=False)
with open(f"{ROOT}/summary_stats.json", "w") as f:
    json.dump({"quality": quality, "stats": stats}, f, indent=2, default=float)

# ---------- charts ----------
INK, MUTED, ACC, NEU = "#1F2933", "#6B7280", "#C2410C", "#94A3B8"

def chart(name, sdf, title):
    fig, ax = plt.subplots(figsize=(4.4, 3.0), dpi=200)
    labels = [str(i) for i in sdf.index]
    cols = [ACC if r >= 0.5 else NEU for r in sdf.rate]
    bars = ax.bar(labels, sdf.rate * 100, color=cols, width=0.62)
    for b, (_, r) in zip(bars, sdf.iterrows()):
        ax.text(b.get_x() + b.get_width() / 2, r.rate * 100 + 3,
                f"{r.rate*100:.0f}%\n{int(r.churned)}/{int(r.n)}", ha="center", va="bottom",
                fontsize=8, color=INK)
    ax.set_ylim(0, 128); ax.set_yticks([0, 50, 100])
    ax.set_yticklabels(["0%", "50%", "100%"], fontsize=8, color=MUTED)
    ax.tick_params(axis="x", labelsize=8, colors=INK, length=0)
    ax.tick_params(axis="y", length=0)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.yaxis.grid(True, color="#E5E7EB", lw=0.6); ax.set_axisbelow(True)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{ROOT}/charts/{name}.png"); plt.close(fig)

chart("contract", S["contract"], "Churn rate by contract type")
chart("tenure", S["tenure"], "Churn rate by tenure (months)")
chart("tickets", S["tickets"], "Churn rate by support tickets")
chart("plan", S["plan"], "Churn rate by plan")
chart("payment", S["payment"], "Churn rate by payment method")

# ---------- executive PDF (16:9 pages) ----------
W, H = 960, 540
c = canvas.Canvas(f"{ROOT}/executive_brief.pdf", pagesize=(W, H))
c.setTitle("Customer Churn: Executive Brief"); c.setAuthor("Executive Analytics Capstone")
pi = lambda x: HexColor(x)

def st(size, color=INK, bold=False, lead=None, align=0):
    return ParagraphStyle("s", fontName="Helvetica-Bold" if bold else "Helvetica", fontSize=size,
                          textColor=pi(color), leading=lead or size * 1.3, alignment=align)

def para(text, x, ytop, w, style):
    p = Paragraph(text, style)
    _, h = p.wrap(w, 1000)
    p.drawOn(c, x, ytop - h)
    return h

page = [0]
def new_page(headline, kicker):
    if page[0]:
        c.showPage()
    page[0] += 1
    c.setFillColor(pi(ACC)); c.rect(0, H - 6, W, 6, stroke=0, fill=1)
    para(kicker.upper(), 48, H - 30, 600, st(9, MUTED, True))
    para(headline, 48, H - 46, W - 96, st(24, INK, True, 29))
    c.setFillColor(pi(MUTED)); c.setFont("Helvetica", 8)
    c.drawString(48, 22, f"Source: cleaned_customer_churn.csv, {N} customers. Small sample: directional evidence only.")
    c.drawRightString(W - 48, 22, str(page[0]))

pct = lambda x, d=1: f"{x*100:.{d}f}%"

# P1 – bottom line
new_page("Churn is a month-to-month problem: 7 of 7 left, 0 of 8 on contracts", "Customer churn executive brief")
para("<b>Decision question.</b> Where should we focus retention effort and spend over the next quarter?",
     48, H - 100, 820, st(15, INK, False, 21))
c.setFillColor(pi("#FFF7ED")); c.roundRect(48, 190, W - 96, 190, 8, stroke=0, fill=1)
c.setFillColor(pi(ACC)); c.rect(48, 190, 5, 190, stroke=0, fill=1)
lo, hi = scen_df.iloc[0], scen_df.iloc[2]
para("<b>Recommendation</b>", 72, 366, 800, st(11, ACC, True))
para(f"Run a 90-day retention pilot, with a holdout group, aimed at month-to-month customers in their first year "
     f"and anyone raising 3 or more support tickets. If the pilot saves 10-30% of the customers who would otherwise churn, "
     f"churn falls from {pct(stats['churn_rate'])} to {pct(hi.churn_after)}-{pct(lo.churn_after)} "
     f"and about ${lo.mrr_recovered:.0f}-${hi.mrr_recovered:.0f} of monthly revenue is kept in this {N}-customer sample.",
     72, 346, 810, st(16, INK, False, 23))
para("These are what-if ranges built on stated assumptions, not forecasts. The sample is small (15 customers), "
     "so the finding is a strong lead to test, not a proven cause.", 72, 238, 810, st(11.5, MUTED, False, 16))
para("Contents: the situation, five insights, impact ranges, limitations, recommended action.",
     48, 150 - 20, 820, st(10, MUTED))

# P2 – situation
new_page("Nearly half of customers churned, but only a third of revenue", "The situation")
tiles = [
    (pct(stats["churn_rate"]), "customers churned", f"{K} of {N}"),
    (f"${MRR_LOST:,.0f}", "monthly revenue lost", f"{pct(stats['mrr_lost_pct'])} of ${MRR:,.0f}"),
    (f"{stats['tenure_churned']:.0f} vs {stats['tenure_retained']:.0f}", "avg tenure, months", "churned vs retained"),
    (f"{stats['tickets_churned']:.1f} vs {stats['tickets_retained']:.1f}", "avg support tickets", "churned vs retained"),
]
tw = (W - 96 - 3 * 16) / 4
for i, (big, lab, sub) in enumerate(tiles):
    x = 48 + i * (tw + 16)
    c.setFillColor(pi("#F8FAFC")); c.roundRect(x, 300, tw, 130, 8, stroke=0, fill=1)
    para(big, x + 16, 412, tw - 24, st(30, ACC if i < 2 else INK, True, 34))
    para(lab, x + 16, 366, tw - 24, st(12, INK, True))
    para(sub, x + 16, 346, tw - 24, st(10.5, MUTED))
para(f"<b>What this means.</b> Customers who leave are newer, contact support far more, and pay less "
     f"(${stats['avg_mrr_churned']:.2f} a month on average versus ${stats['avg_mrr_retained']:.2f} for those who stay). "
     f"That is why customer churn ({pct(stats['churn_rate'],0)}) is higher than revenue churn ({pct(stats['mrr_lost_pct'],0)}). "
     f"The business is losing many small accounts, while Enterprise accounts, all 4 retained, carry nearly half of monthly revenue.",
     48, 270, 840, st(14, INK, False, 20))
para("Data check: no missing values or duplicate IDs; TotalCharges equals monthly charge times tenure for every row, "
     "so it adds no information beyond those two fields.", 48, 140, 840, st(10.5, MUTED, False, 15))

# P3 – insights 1-2
new_page("Contract type and early tenure mark who leaves", "Insights 1 and 2")
c.drawImage(f"{ROOT}/charts/contract.png", 48, 200, width=360, height=360 * 3.0 / 4.4, preserveAspectRatio=True)
c.drawImage(f"{ROOT}/charts/tenure.png", 492, 200, width=360, height=360 * 3.0 / 4.4, preserveAspectRatio=True)
para(f"<b>1. Contract type.</b> All 7 month-to-month customers churned; none of the 8 on one- or two-year terms did "
     f"(exact one-sided p = {stats['p_contract']:.4f}). With so few customers, the true month-to-month churn rate could "
     f"plausibly be as low as {pct(stats['m2m_lower95'],0)}, still very high.", 48, 190, 420, st(11.5, INK, False, 16))
para("<b>2. Early tenure.</b> Every customer with 12 months or less (6 of 6) churned, and nobody beyond 24 months did. "
     "The first year is when we win or lose the account.", 492, 190, 420, st(11.5, INK, False, 16))

# P4 – insights 3-5
new_page("Support load, low-priced plans and payment method add detail", "Insights 3, 4 and 5")
cw = 288
for i, nm in enumerate(["tickets", "plan", "payment"]):
    c.drawImage(f"{ROOT}/charts/{nm}.png", 48 + i * (cw + 12), 250, width=cw, height=cw * 3.0 / 4.4, preserveAspectRatio=True)
txt = [
    "<b>3. Support load.</b> Nobody with 0-2 tickets churned (0 of 8); everyone with 3 or more did (7 of 7). "
    "Tickets are a usable early-warning signal.",
    f"<b>4. Plan and value.</b> Basic lost 5 of 7, Enterprise none. Churners pay about half as much, so fixing churn "
    f"protects customer count more than revenue (${MRR_LOST:.0f} a month at stake).",
    "<b>5. Payment method (weak).</b> UPI and debit-card payers churn most, but this probably mirrors contract and plan. "
    "Treat it as a hypothesis, not a lever.",
]
for i, t in enumerate(txt):
    para(t, 48 + i * (cw + 12), 242, cw - 6, st(10.5, INK, False, 14.5))
if stats["m2m_equals_3plus_tickets"]:
    para("Note: the month-to-month customers and the customers with 3+ tickets are the same 7 people, so this sample cannot "
         "tell contract effect and support experience apart.", 48, 118, 860, st(10.5, ACC, True, 14))

# P5 – impact
new_page("A 10-30% save rate would cut churn to about 33-42%", "Expected impact (what-if ranges)")
data = [["Scenario", "Churners saved", "Churn rate after", "Monthly revenue kept", "Per year", "Share of monthly revenue"]]
for _, r in scen_df.iterrows():
    data.append([f"{r.scenario}: save {r.save_rate*100:.0f}% of churners", f"{r.customers_saved:.1f} of {K}",
                 pct(r.churn_after), f"${r.mrr_recovered:,.0f}", f"${r.annualised:,.0f}", pct(r.pct_of_mrr)])
t = Table(data, colWidths=[230, 110, 120, 140, 100, 150])
t.setStyle(TableStyle([
    ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 10), ("FONT", (0, 1), (-1, -1), "Helvetica", 11),
    ("TEXTCOLOR", (0, 0), (-1, 0), white), ("BACKGROUND", (0, 0), (-1, 0), pi(INK)),
    ("TEXTCOLOR", (0, 1), (-1, -1), pi(INK)), ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [pi("#F8FAFC"), white]),
    ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ("LINEBELOW", (0, -1), (-1, -1), 0.6, pi("#CBD5E1")),
]))
tw_, th_ = t.wrap(0, 0); t.drawOn(c, 48, H - 130 - th_)
y = H - 150 - th_
para("<b>Assumptions</b>", 48, y, 400, st(12, INK, True))
bul = [
    f"Baseline: {K} of {N} customers churned ({pct(stats['churn_rate'])}), losing ${MRR_LOST:.2f} of ${MRR:,.2f} monthly revenue.",
    f"A saved customer keeps paying the average churner's ${stats['avg_mrr_churned']:.2f} a month, for 12 months.",
    "The 10-30% save rate is a planning assumption to be replaced by pilot results. It is not measured in this data.",
    "Program cost (discounts, staff time) is not included. Net benefit = revenue kept minus that cost.",
]
para("<br/>".join("&bull; " + b for b in bul), 48, y - 22, 860, st(11.5, INK, False, 17))
para("Read these as the size of the prize in this sample. Apply the same percentages to the full customer base for a real estimate.",
     48, 120, 860, st(11, MUTED, False, 15))

# P6 – limitations
new_page("What this analysis can and cannot tell us", "Limitations")
lim = [
    ("Very small sample", f"{N} customers. A 100% or 0% rate here is a sample fact; the true rate is uncertain "
     f"(month-to-month could be as low as about {pct(stats['m2m_lower95'],0)}; contracts could be as high as about {pct(stats['contract_upper95'],0)})."),
    ("Drivers overlap", "Contract type, ticket volume and early tenure describe largely the same customers, so we cannot say which one causes churn."),
    ("Association, not cause", "Nothing here shows that moving a customer to an annual contract will keep them. That is what the pilot tests."),
    ("No timing or reasons", "We have a snapshot with no cancellation dates, exit reasons, pricing history or acquisition channel."),
    ("No cost data", "Impact figures are revenue kept, not profit. Program costs must be netted off."),
]
yy = H - 130
for h_, b_ in lim:
    c.setFillColor(pi(ACC)); c.rect(48, yy - 40, 4, 44, stroke=0, fill=1)
    para(f"<b>{h_}.</b> {b_}", 64, yy, 830, st(12.5, INK, False, 17.5))
    yy -= 66

# P7 – recommendations
new_page("Recommended action: a 90-day pilot with a holdout group", "Recommendations and next steps")
recs = [
    ("Offer a contract upgrade", "Give month-to-month customers a discounted annual plan at the 60-90 day mark, focusing on Basic and Pro."),
    ("Protect the first year", "Add an onboarding check-in programme for the first 12 months, the period where 6 of 6 customers churned."),
    ("Act on the third ticket", "Trigger proactive outreach when a customer raises a third support ticket, before they raise a fourth."),
    ("Keep Enterprise as is", "No Enterprise customer churned; spend retention effort elsewhere."),
    ("Test and measure", "Hold out a control group. Success: churn among treated month-to-month customers at least 5 points below the holdout."),
]
yy = H - 130
for i, (h_, b_) in enumerate(recs, 1):
    c.setFillColor(pi(ACC)); c.circle(62, yy - 8, 11, stroke=0, fill=1)
    c.setFillColor(white); c.setFont("Helvetica-Bold", 11); c.drawCentredString(62, yy - 12, str(i))
    para(f"<b>{h_}.</b> {b_}", 86, yy, 810, st(12.5, INK, False, 17.5))
    yy -= 58
para("<b>Decision requested:</b> approve the pilot and a larger data pull (cancellation dates and reasons) so the next review uses real save rates.",
     48, 110, 860, st(12.5, ACC, False, 17))
c.save()

# ---------- dashboard ----------
def bars(sdf):
    out = ""
    for idx, r in sdf.iterrows():
        cls = "hi" if r.rate >= 0.5 else "lo"
        out += (f'<div class="row"><span class="lab">{idx}</span><div class="track">'
                f'<div class="bar {cls}" style="width:{max(r.rate*100,0.8):.1f}%"></div></div>'
                f'<span class="val">{r.rate*100:.0f}% <small>({int(r.churned)}/{int(r.n)})</small></span></div>')
    return out

cards = [
    ("1", "Contract type separates churners from stayers", "contract",
     "All 7 month-to-month customers churned; no customer on a one- or two-year contract did."),
    ("2", "Customers churn early", "tenure", "Everyone with up to 12 months of tenure left; nobody past 24 months did."),
    ("3", "Heavy support contact precedes churn", "tickets",
     "0-2 tickets: nobody churned. 3+ tickets: everyone did. Same 7 people as month-to-month."),
    ("4", "Low-priced plans lose the most customers", "plan",
     f"Churners pay ${stats['avg_mrr_churned']:.2f} a month versus ${stats['avg_mrr_retained']:.2f} for stayers, so revenue loss is {pct(stats['mrr_lost_pct'],0)}, below the {pct(stats['churn_rate'],0)} customer loss."),
    ("5", "Payment method: weak signal", "payment",
     "UPI and debit payers churn more, but this likely mirrors contract and plan. Treat as a hypothesis."),
]
cards_html = "".join(
    f'<section class="card"><h3><span class="n">{n}</span>{t}</h3>{bars(S[k])}<p>{d}</p></section>'
    for n, t, k, d in cards)

html = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Churn Dashboard</title>
<style>
:root{--bg:#fff;--ink:#1F2933;--mut:#6B7280;--card:#F8FAFC;--line:#E5E7EB;--acc:#C2410C;--neu:#94A3B8}
@media (prefers-color-scheme:dark){:root{--bg:#111418;--ink:#E5E7EB;--mut:#9CA3AF;--card:#1A1F26;--line:#2B323B;--acc:#F97316;--neu:#64748B}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1000px;margin:0 auto;padding:28px 16px 48px}
h1{font-size:26px;line-height:1.25;margin:0 0 6px}h2{font-size:18px;margin:32px 0 12px}
.sub{color:var(--mut);margin:0 0 20px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.tile b{display:block;font-size:28px;color:var(--acc)}.tile span{color:var(--mut);font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}
.card h3{font-size:15px;margin:0 0 12px;display:flex;gap:8px;align-items:flex-start}
.n{background:var(--acc);color:#fff;border-radius:50%;min-width:22px;height:22px;display:inline-flex;align-items:center;justify-content:center;font-size:12px}
.card p{color:var(--mut);font-size:13.5px;margin:12px 0 0}
.row{display:grid;grid-template-columns:96px 1fr 84px;align-items:center;gap:8px;margin:6px 0;font-size:13px}
.track{background:var(--line);border-radius:4px;height:14px}.bar{height:14px;border-radius:4px}
.bar.hi{background:var(--acc)}.bar.lo{background:var(--neu)}.val{text-align:right}.val small{color:var(--mut)}
.box{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}
input[type=range]{width:100%;accent-color:var(--acc)}
.out{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-top:12px}
.out div b{display:block;font-size:22px}.out div span{color:var(--mut);font-size:12.5px}
ul{padding-left:20px;margin:8px 0}li{margin:4px 0}.note{color:var(--mut);font-size:13px}
.call{border-left:4px solid var(--acc);padding:6px 12px;margin:16px 0;background:var(--card)}
</style></head><body><main>
<h1>Churn is a month-to-month problem: 7 of 7 left, 0 of 8 on contracts</h1>
<p class="sub">Executive dashboard, __N__ customers. Directional evidence only: the sample is small.</p>
<div class="tiles">
<div class="tile"><b>__CHURN__</b><span>customers churned (__K__ of __N__)</span></div>
<div class="tile"><b>$__LOST__</b><span>monthly revenue lost (__LOSTPCT__ of $__MRR__)</span></div>
<div class="tile"><b>__TC__ vs __TR__</b><span>avg tenure in months, churned vs retained</span></div>
<div class="tile"><b>__KC__ vs __KR__</b><span>avg support tickets, churned vs retained</span></div>
</div>
<h2>Five insights</h2><div class="grid">__CARDS__</div>
<div class="call">__OVERLAP__</div>
<h2>Impact explorer (what-if, not a forecast)</h2>
<div class="box"><label for="s"><b>Share of would-be churners saved: <span id="sv">20</span>%</b></label>
<input id="s" type="range" min="0" max="50" value="20">
<div class="out">
<div><b id="o1"></b><span>churn rate after (now __CHURN__)</span></div>
<div><b id="o2"></b><span>customers saved of __K__</span></div>
<div><b id="o3"></b><span>monthly revenue kept</span></div>
<div><b id="o4"></b><span>revenue kept per year</span></div></div>
<p class="note">Assumes a saved customer pays the average churner's $__AVG__ a month for 12 months. Program cost not included. The 10-30% range in the brief is a planning assumption to be replaced by pilot results.</p></div>
<h2>Limitations</h2><div class="box"><ul>
<li><b>Very small sample.</b> 100% and 0% rates are sample facts; month-to-month churn could plausibly be as low as about __M2M__.</li>
<li><b>Drivers overlap.</b> Contract, tickets and early tenure describe largely the same customers.</li>
<li><b>Association, not cause.</b> The pilot is what tests whether contracts retain customers.</li>
<li><b>No timing, reasons or cost data.</b> Impact is revenue kept, not profit.</li></ul></div>
<h2>Recommended action</h2><div class="box"><ul>
<li>Offer month-to-month customers a discounted annual plan at 60-90 days.</li>
<li>Add a first-year onboarding check-in programme.</li>
<li>Trigger proactive outreach on the third support ticket.</li>
<li>Run it as a 90-day pilot with a holdout group; success is at least 5 points lower churn than the holdout.</li></ul></div>
</main>
<script>
var K=__K__,N=__N__,LOST=__LOSTNUM__;
function upd(){var p=+document.getElementById('s').value/100;
document.getElementById('sv').textContent=Math.round(p*100);
document.getElementById('o1').textContent=((K-K*p)/N*100).toFixed(1)+'%';
document.getElementById('o2').textContent=(K*p).toFixed(1);
document.getElementById('o3').textContent='$'+(LOST*p).toFixed(0);
document.getElementById('o4').textContent='$'+(LOST*p*12).toFixed(0).replace(/\\B(?=(\\d{3})+(?!\\d))/g,',');}
document.getElementById('s').addEventListener('input',upd);upd();
</script></body></html>"""
rep = {
    "__N__": str(N), "__K__": str(K), "__CHURN__": pct(stats["churn_rate"]),
    "__LOST__": f"{MRR_LOST:,.0f}", "__LOSTPCT__": pct(stats["mrr_lost_pct"]), "__MRR__": f"{MRR:,.0f}",
    "__TC__": f"{stats['tenure_churned']:.0f}", "__TR__": f"{stats['tenure_retained']:.0f}",
    "__KC__": f"{stats['tickets_churned']:.1f}", "__KR__": f"{stats['tickets_retained']:.1f}",
    "__CARDS__": cards_html, "__AVG__": f"{stats['avg_mrr_churned']:.2f}",
    "__M2M__": pct(stats["m2m_lower95"], 0), "__LOSTNUM__": f"{MRR_LOST:.2f}",
    "__OVERLAP__": ("<b>Note:</b> the month-to-month customers and the customers with 3+ tickets are the same 7 people, "
                    "so this sample cannot separate contract effect from support experience."
                    if stats["m2m_equals_3plus_tickets"] else ""),
}
for k, v in rep.items():
    html = html.replace(k, v)
open(f"{ROOT}/dashboard.html", "w").write(html)
print(json.dumps({"quality": quality, "stats": stats}, indent=1, default=float))
print(scen_df.round(3).to_string())
for k, v in S.items():
    print(k); print(v.round(3))
