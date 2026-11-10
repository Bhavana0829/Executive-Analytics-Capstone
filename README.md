# Executive Analytics Capstone: Customer Churn

**Question:** Where should we focus retention effort and spend next quarter?
**Answer:** On month-to-month customers in their first year and anyone raising 3 or more support tickets. All 7 month-to-month customers churned; none of the 8 contract customers did. The sample is small (15 customers), so treat this as a lead to test with a pilot.

## Files
| File | What it is |
|---|---|
| `executive_brief.pdf` | 7-page executive brief: situation, five insights, impact ranges, limitations, recommendation |
| `dashboard.html` | Self-contained dashboard (open in any browser, no internet needed), with an impact slider |
| `presentation_script.md` | Script for the optional 3-minute recording |
| `insights.csv` | The five insights with evidence, exact p-values and confidence |
| `impact_scenarios.csv` | Low / mid / high what-if scenarios |
| `summary_stats.json` | Data-quality checks and every headline figure |
| `charts/` | The five chart images used in the PDF |
| `data/cleaned_customer_churn.csv` | Source data |
| `build.py` | Reproduces everything: `python3 build.py` (needs pandas, matplotlib, reportlab) |

## Five insights
1. Contract type: month-to-month 7/7 churned; one-year 0/4; two-year 0/4.
2. Early tenure: 12 months or less, 6/6 churned; 25+ months, 0/4.
3. Support load: 0-2 tickets 0/8 churned; 3+ tickets 7/7.
4. Plan and value: Basic 5/7, Pro 2/4, Enterprise 0/4; churners pay $58.56/mo vs $107.49.
5. Payment method (weak): UPI 3/4, debit 2/2, credit card 2/6, bank transfer 0/3.

## Impact (what-if, not a forecast)
Saving 10-30% of would-be churners lowers churn from 46.7% to 42.0%-32.7% and keeps about $41-$123 of monthly revenue ($492-$1,476 a year). Assumes saved customers pay the average churner's $58.56/mo for 12 months; programme cost excluded.

## Limitations
- 15 customers: rates of 0% and 100% are sample facts, not true rates.
- Month-to-month and 3+ tickets are the same 7 customers, so their effects cannot be separated.
- Association, not causation; no timing, exit reasons or cost data.

## To submit
Upload this folder (or the zip) to a public/view-only location such as Google Drive or a GitHub repository, then submit that one link. Optionally add your 3-minute recording to the same folder.
