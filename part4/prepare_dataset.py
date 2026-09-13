"""
Part 4, Step 1 — Prepare financial Q&A dataset for fine-tuning.
Uploads train/val/test splits to S3 and writes dataset_manifest.json.

Usage:
    export TRAINING_BUCKET=my-sagemaker-bucket
    python prepare_dataset.py
"""

import csv
import json
import math
import os
import random
from collections import Counter
from pathlib import Path

import boto3

# ── Dataset ──────────────────────────────────────────────────────────────────

QA_PAIRS = [
    # Savings accounts
    {"q": "What is APY and how does it differ from APR?",
     "a": "APY (Annual Percentage Yield) reflects the total amount of interest earned on an account in one year, including compounding. APR (Annual Percentage Rate) is the simple interest rate without compounding. APY is always equal to or greater than APR; the difference widens with more frequent compounding periods."},
    {"q": "What does FDIC insurance cover?",
     "a": "The FDIC insures deposits up to $250,000 per depositor, per insured bank, per ownership category. Coverage applies to checking accounts, savings accounts, money market deposit accounts, and CDs. It does not cover investment products such as stocks, bonds, or mutual funds, even if purchased through a bank."},
    {"q": "How does compound interest work on a savings account?",
     "a": "Compound interest is calculated on both the principal and the accumulated interest from prior periods. For example, $10,000 at 5% APY compounded monthly grows to approximately $10,511.62 after one year—compared to $10,500 under simple interest. The formula is A = P(1 + r/n)^(nt), where n is compounding periods per year."},
    {"q": "What is a high-yield savings account?",
     "a": "A high-yield savings account offers an APY significantly above the national average (currently around 0.45%). Online banks typically offer rates of 4–5% APY because they have lower overhead than traditional branches. FDIC insurance still applies, and funds remain fully liquid."},
    {"q": "What is the difference between a savings account and a money market account?",
     "a": "Both are FDIC-insured interest-bearing accounts, but money market accounts typically offer higher rates in exchange for higher minimum balance requirements. Money market accounts may also include check-writing and debit card privileges, while traditional savings accounts generally do not. Both are subject to federal transaction limits."},
    {"q": "What is a certificate of deposit (CD)?",
     "a": "A CD is a time-deposit savings product that pays a fixed interest rate for a specified term—typically 3 months to 5 years. In exchange for locking funds for the term, you receive a higher rate than a standard savings account. Early withdrawal incurs a penalty, usually 60–180 days of interest depending on the term."},
    {"q": "What is a CD ladder strategy?",
     "a": "A CD ladder splits deposits across multiple CDs with staggered maturities—for example, 1-year, 2-year, 3-year, 4-year, and 5-year CDs. As each CD matures, you reinvest at the longest term. This balances liquidity (a CD matures every year) with higher long-term rates while reducing reinvestment risk."},

    # Mortgage products
    {"q": "What is the difference between a fixed-rate and adjustable-rate mortgage?",
     "a": "A fixed-rate mortgage maintains the same interest rate and monthly payment for the entire loan term (typically 15 or 30 years). An adjustable-rate mortgage (ARM) starts with a fixed rate for an initial period (e.g., 5/1 ARM = fixed for 5 years, then adjusts annually). ARMs carry rate risk after the fixed period but often offer lower initial rates."},
    {"q": "What is loan-to-value (LTV) ratio and why does it matter?",
     "a": "LTV is calculated as (loan amount / property value) × 100. An 80% LTV means you borrowed 80% and put down 20%. Lenders use LTV to assess risk—lower LTV means less risk. LTV above 80% typically requires private mortgage insurance (PMI). Better LTV ratios generally qualify for lower interest rates."},
    {"q": "What is private mortgage insurance (PMI) and when can it be removed?",
     "a": "PMI is required when a conventional mortgage LTV exceeds 80%. It protects the lender, not the borrower, and typically costs 0.5%–1.5% of the loan balance annually. Under the Homeowners Protection Act, you can request PMI cancellation when LTV reaches 80% based on original value. It automatically terminates at 78% LTV."},
    {"q": "What are mortgage points and when are they worth buying?",
     "a": "One mortgage point equals 1% of the loan amount paid upfront to reduce the interest rate—typically by 0.25% per point. The break-even analysis: divide point cost by monthly payment reduction. If break-even is 36 months and you plan to stay 10 years, buying points makes sense. Points are generally not worthwhile if you expect to refinance or sell within a few years."},
    {"q": "What is a debt-to-income (DTI) ratio for mortgage qualification?",
     "a": "DTI is total monthly debt payments divided by gross monthly income. Most conventional loans require a back-end DTI below 43%; some allow up to 50% with compensating factors. Front-end DTI (housing costs only) should typically be below 28%. Lower DTI improves approval odds and may qualify you for better rates."},
    {"q": "What is a pre-approval for a mortgage?",
     "a": "Pre-approval is a lender's conditional commitment to lend a specific amount based on verified income, credit, assets, and employment. It requires a hard credit pull and documentation review. Pre-approval differs from pre-qualification (which is an estimate based on self-reported data). Sellers prefer offers from pre-approved buyers."},

    # Credit products
    {"q": "How does APR differ from interest rate on a credit card?",
     "a": "The interest rate is the annual cost of borrowing expressed as a percentage. APR includes the interest rate plus fees (e.g., annual fee, origination fee), giving a broader measure of cost. For credit cards, APR and interest rate are often the same since most fees are charged separately. APR matters most when comparing loan products with fees."},
    {"q": "How does credit utilization affect your credit score?",
     "a": "Credit utilization—the ratio of credit card balances to credit limits—accounts for approximately 30% of a FICO score. Utilization below 30% is generally recommended; below 10% is ideal for top scores. Both per-card and overall utilization matter. Paying balances in full monthly or multiple times per month keeps utilization low."},
    {"q": "What is the impact of a hard credit inquiry on your score?",
     "a": "A hard inquiry occurs when a lender checks your credit for a lending decision. It typically reduces your FICO score by 5 points or fewer and remains on your report for 2 years. Multiple mortgage or auto loan inquiries within a 45-day window are treated as a single inquiry by FICO. Soft inquiries (e.g., checking your own credit) do not affect scores."},
    {"q": "What is a balance transfer and how does it work?",
     "a": "A balance transfer moves debt from a high-APR card to a new card with a promotional 0% APR period (typically 12–21 months). A transfer fee of 3%–5% usually applies. The strategy is effective when you can pay off the balance before the promotional period ends. Failing to do so results in the standard APR applying to any remaining balance."},
    {"q": "How is the minimum payment on a credit card calculated?",
     "a": "Most issuers calculate the minimum as the greater of a flat amount ($25–$35) or a percentage of the statement balance (typically 1%–3%) plus interest and fees. Paying only minimums on a $5,000 balance at 20% APR can take over 15 years and cost more than $4,000 in interest. Always pay more than the minimum when possible."},

    # Investment accounts
    {"q": "What is the difference between a Traditional IRA and a Roth IRA?",
     "a": "Traditional IRA contributions may be tax-deductible; withdrawals in retirement are taxed as ordinary income. Roth IRA contributions are made with after-tax dollars; qualified withdrawals in retirement are tax-free. The 2024 contribution limit is $7,000 ($8,000 if age 50+) for both combined. Roth IRAs have income limits for contributions; Traditional IRAs do not."},
    {"q": "What is the 2024 401(k) contribution limit?",
     "a": "The 2024 IRS contribution limit for employees is $23,000. Workers age 50 and older can make an additional catch-up contribution of $7,500, for a total of $30,500. The combined employer + employee contribution limit is $69,000 ($76,500 with catch-up). These limits are indexed to inflation and typically increase annually."},
    {"q": "What is a Roth conversion and when does it make sense?",
     "a": "A Roth conversion moves pre-tax retirement funds (Traditional IRA or 401k) to a Roth IRA, triggering income tax in the conversion year. It makes sense when: current tax rate is lower than expected future rate, you have funds to pay taxes outside the IRA, you want to avoid RMDs, or you plan to pass wealth to heirs (Roth has no RMDs in the owner's lifetime)."},
    {"q": "What is an index ETF and how does it compare to an actively managed fund?",
     "a": "An index ETF tracks a market index (e.g., S&P 500) and passively holds all or representative securities in that index. Expense ratios are typically 0.03%–0.20%. Actively managed funds employ portfolio managers selecting securities, with expense ratios of 0.5%–1.5% or higher. Over 10-year periods, the majority of actively managed funds underperform their benchmark index after fees."},
    {"q": "What is dollar-cost averaging (DCA)?",
     "a": "Dollar-cost averaging is the practice of investing a fixed dollar amount at regular intervals (e.g., $500 monthly) regardless of market conditions. When prices fall, you buy more shares; when prices rise, you buy fewer. Over time this reduces average cost per share and eliminates the risk of investing a lump sum at a market peak. It is especially effective in volatile markets."},
    {"q": "What is an investor's risk profile and how is it assessed?",
     "a": "A risk profile captures an investor's capacity and willingness to accept investment risk. Assessment considers: time horizon (longer = higher risk capacity), financial goals, income stability, existing assets, and psychological comfort with volatility. Risk profiles range from conservative (capital preservation) to aggressive (maximum growth). Asset allocation should match the risk profile."},
    {"q": "What is required minimum distribution (RMD) from a retirement account?",
     "a": "RMDs are mandatory annual withdrawals from Traditional IRAs and most employer-sponsored retirement plans starting at age 73 (per SECURE 2.0 Act, effective 2023). The amount is calculated by dividing the prior year-end account balance by an IRS life expectancy factor. Failing to take the full RMD results in a 25% excise tax on the shortfall (reduced to 10% if corrected timely)."},

    # Compliance
    {"q": "What is Know Your Customer (KYC) and what information is collected?",
     "a": "KYC is a regulatory requirement for financial institutions to verify the identity of clients and assess risk. Required information includes: full legal name, date of birth, address, and government-issued ID number (SSN for US persons). Enhanced due diligence applies to high-risk customers. KYC is the foundation of Anti-Money Laundering (AML) programs and is mandated by FinCEN's Customer Identification Program rules."},
    {"q": "What are common red flags for money laundering (AML)?",
     "a": "Common AML red flags include: structuring transactions just below $10,000 reporting thresholds (smurfing), frequent large cash deposits inconsistent with business type, rapid movement of funds through multiple accounts, transactions involving high-risk jurisdictions, customers reluctant to provide identification, and unexplained wire transfers to foreign accounts. All suspicious activity must be reported via a Suspicious Activity Report (SAR)."},
    {"q": "What is a Suspicious Activity Report (SAR) and when must it be filed?",
     "a": "A SAR is a report filed with FinCEN when a financial institution suspects or knows that a transaction involves funds from illegal activity, is designed to evade reporting requirements, lacks a lawful purpose, or involves an insider crime. SARs must be filed within 30 days of detecting suspicious activity (60 days if no suspect is identified). Filing is mandatory for transactions of $5,000 or more meeting the criteria. SARs are confidential."},
    {"q": "What is Currency Transaction Report (CTR) and when is it required?",
     "a": "A CTR must be filed with FinCEN for any cash transaction exceeding $10,000 in a single business day, regardless of whether it appears suspicious. Banks must file within 15 days of the transaction. CTRs aggregate multiple transactions by the same person in the same day. Structuring transactions to avoid the $10,000 threshold is itself a federal crime under 31 U.S.C. § 5324."},
    {"q": "How does GDPR affect financial institutions handling EU customer data?",
     "a": "GDPR requires lawful basis for processing EU personal data, explicit consent for marketing, right to access and erasure ('right to be forgotten'), data portability, and breach notification within 72 hours. Financial institutions must appoint a Data Protection Officer if processing at scale. Penalties reach €20 million or 4% of global annual turnover. AML record-retention requirements (typically 5–7 years) create tension with the right to erasure—regulated retention needs trump erasure requests."},
    {"q": "What is Regulation E and what consumer protections does it provide?",
     "a": "Regulation E implements the Electronic Fund Transfer Act and protects consumers in electronic transactions including debit cards, ATMs, ACH transfers, and online bill pay. Key protections: limited liability for unauthorized transactions if reported within 2 days ($50 cap), within 60 days ($500 cap), and after 60 days (unlimited). Banks must investigate disputes within 10 business days (45 days if provisional credit is given)."},

    # Customer service scenarios
    {"q": "How do I dispute an unauthorized charge on my account?",
     "a": "Contact your bank immediately by phone or through online banking. For debit cards/EFTs, disputes are governed by Regulation E—report within 2 business days for maximum protection ($50 liability). For credit cards, the Fair Credit Billing Act gives 60 days from the statement date to dispute. Provide transaction details, submit in writing if requested, and the bank must acknowledge within 30 days and resolve within 60 days for credit card disputes."},
    {"q": "How do I report suspected fraud on my account?",
     "a": "Call the number on the back of your card immediately. The bank will freeze the compromised account, issue new cards, and open a fraud investigation. File a police report if identity theft is involved—your bank may require it. Place a fraud alert with the three credit bureaus (Equifax, Experian, TransUnion). Review 90 days of transactions for other unauthorized activity. Document everything in writing."},
    {"q": "How do I request a copy of my account statement?",
     "a": "Statements are available through online banking (typically 12–24 months history) or the mobile app. For older statements or paper copies, contact customer service or visit a branch. Banks are required to retain statements for 5 years. Some institutions charge a fee ($5–$10) for paper statement reprints beyond 90 days. For mortgage or loan statements, year-end tax statements are available in January for interest paid during the prior year."},
    {"q": "What should I do if I think my identity has been stolen?",
     "a": "Immediately: (1) Place a fraud alert with one credit bureau (it notifies the other two). (2) Request free credit reports from AnnualCreditReport.com. (3) File an identity theft report at IdentityTheft.gov (FTC). (4) Contact your financial institutions to flag accounts. (5) File a police report. Consider a credit freeze (free, must be placed with each bureau separately) to prevent new account openings. Monitor all accounts for 12 months."},
    {"q": "How does my bank's overdraft protection work?",
     "a": "Standard overdraft coverage allows the bank to pay transactions that exceed your balance, subject to an overdraft fee (typically $25–$35 per item). Opt-in is required for ATM and one-time debit card transactions under Regulation E. Alternatives include: overdraft protection linked to a savings account or credit line (lower fees), overdraft grace periods, or no-fee overdraft programs. You can opt out of standard overdraft coverage at any time."},

    # Digital banking
    {"q": "What are the limits for mobile check deposits?",
     "a": "Mobile deposit limits vary by institution and customer relationship. New customers may have limits of $1,000–$2,500 per day and $5,000 per month. Established customers often have limits of $5,000–$10,000 per day. Funds availability follows Regulation CC: the first $225 is available next business day; the rest may be held 2–5 business days for new accounts or large checks. Checks over the limit must be deposited in person."},
    {"q": "What is the difference between ACH and wire transfer?",
     "a": "ACH (Automated Clearing House) transfers are processed in batches, typically taking 1–3 business days; fees are low ($0–$3). Wire transfers are processed individually in real-time through the Federal Reserve Wire Network (Fedwire) or SWIFT for international; same-day settlement; fees of $15–$50 domestic, $25–$50+ international. Use ACH for routine transfers; wire for time-sensitive or large transactions requiring same-day finality."},
    {"q": "What are typical Zelle daily and weekly transfer limits?",
     "a": "Zelle limits vary by participating bank. Typical limits are $500–$2,500 per day and $5,000–$10,000 per week for personal accounts; business accounts may have higher limits. Zelle transactions are instant and irrevocable once sent—you cannot reverse a payment to the wrong recipient unless the recipient voluntarily returns funds. Zelle is intended for payments to people you know and trust."},
    {"q": "What is a SWIFT code and when do I need it?",
     "a": "A SWIFT (Society for Worldwide Interbank Financial Telecommunication) code identifies a bank for international wire transfers. It is 8 or 11 characters: 4-letter bank code + 2-letter country code + 2-character location code + optional 3-character branch code. You need it when sending or receiving international wires. For US domestic wires, use the 9-digit ABA routing number instead."},
    {"q": "What is two-factor authentication (2FA) and why should I use it for online banking?",
     "a": "2FA requires two forms of verification: something you know (password) plus something you have (phone for OTP) or something you are (biometric). It prevents unauthorized access even if your password is compromised. Authenticator apps (Google Authenticator, Authy) are more secure than SMS-based 2FA, which is vulnerable to SIM-swapping attacks. Always enable 2FA on financial accounts—it is your single most effective account security measure."},

    # Fees and charges
    {"q": "What is an NSF fee and how can I avoid it?",
     "a": "An NSF (Non-Sufficient Funds) fee is charged when a transaction is declined because your account balance is insufficient—typically $25–$35 per item. It differs from an overdraft fee (where the bank pays the item). Avoid NSF fees by: maintaining a buffer balance, setting low-balance alerts, enrolling in overdraft protection, or choosing a bank with no NSF fees. Many banks have eliminated NSF fees due to regulatory pressure."},
    {"q": "What are typical wire transfer fees?",
     "a": "Domestic outgoing wires: $15–$35. Domestic incoming wires: $0–$15. International outgoing wires: $25–$50+. International incoming wires: $10–$20. Some premium accounts (Private Banking, Premier) waive wire fees entirely. SWIFT fees are charged by intermediary banks in the correspondent banking chain, reducing the received amount—consider sending a slightly larger amount to ensure exact delivery."},
    {"q": "What ATM fees should I expect when using an out-of-network ATM?",
     "a": "Out-of-network ATM fees have two components: (1) your bank's surcharge (typically $2.50–$3.50) and (2) the ATM operator's surcharge (typically $2.50–$5.00), for a combined cost of $5–$8 per transaction. Some accounts (online banks, Charles Schwab brokerage) reimburse ATM fees globally. To avoid fees: use your bank's network, withdraw larger amounts less frequently, or get cash back at grocery stores."},
    {"q": "What is a monthly maintenance fee and how can it be waived?",
     "a": "Monthly maintenance fees range from $5–$25 for checking accounts. Waiver conditions typically include: maintaining a minimum daily balance (e.g., $1,500), a minimum monthly direct deposit (e.g., $500), holding multiple accounts, or qualifying for a student/senior account. Online banks and credit unions generally offer fee-free checking. Review your account's fee schedule in your account agreement or online banking portal."},
    {"q": "What are early termination fees for CDs?",
     "a": "Early withdrawal penalties (EWPs) for CDs vary by term. Common structures: terms under 1 year = 60–90 days interest; 1–2 year terms = 150–180 days interest; 3–5 year terms = 180–365 days interest. Some 'no-penalty' CDs allow withdrawal after 7 days with no fee. Always compare the EWP cost against the benefit of reinvesting at a higher rate before breaking a CD early."},

    # Additional mixed coverage
    {"q": "What is a beneficiary designation and why is it important?",
     "a": "A beneficiary designation specifies who receives account assets upon death, bypassing probate. It overrides your will—outdated beneficiary designations (e.g., ex-spouse) are a common estate planning error. Review and update beneficiaries after major life events (marriage, divorce, birth, death). Payable-on-Death (POD) for bank accounts and Transfer-on-Death (TOD) for brokerage accounts function the same way. Name both primary and contingent beneficiaries."},
    {"q": "What is the difference between a checking and savings account for cash management?",
     "a": "Checking accounts are designed for frequent transactions—unlimited debits, check writing, debit card access, bill pay—with low or no interest. Savings accounts earn interest but historically had a 6-transaction-per-month federal limit (suspended in 2020, though many banks retain it). Optimal cash management: maintain a checking account for daily expenses and a high-yield savings account for emergency funds and short-term savings goals."},
    {"q": "What is a home equity line of credit (HELOC)?",
     "a": "A HELOC is a revolving line of credit secured by home equity, with a draw period (typically 10 years) and repayment period (typically 20 years). Interest rates are variable (usually Prime + margin). The credit limit is typically 80%–85% of home value minus the mortgage balance. Interest paid may be tax-deductible if used for home improvement. HELOCs carry risk—failure to repay could result in foreclosure."},
    {"q": "How do I calculate my home equity?",
     "a": "Home equity = current market value − outstanding mortgage balance. For example: $450,000 home value − $280,000 mortgage = $170,000 equity. Equity grows through: principal paydown, home appreciation, and improvements. Most lenders allow borrowing up to 80%–85% combined LTV (mortgage + HELOC). You can access equity through a cash-out refinance, HELOC, or home equity loan."},
    {"q": "What is the difference between a secured and unsecured loan?",
     "a": "A secured loan is backed by collateral (e.g., car loan, mortgage, secured credit card). If you default, the lender can seize the collateral. Interest rates are lower because risk to the lender is reduced. An unsecured loan has no collateral (e.g., personal loan, credit card). Lenders rely solely on creditworthiness; rates are higher. In bankruptcy, secured creditors have priority over unsecured creditors."},
]


def build_instruction_format(pair: dict) -> dict:
    return {
        "prompt": f"Human: {pair['q']}\n\nAssistant:",
        "completion": f" {pair['a']}<|endoftext|>",
    }


def split_dataset(data: list, train=0.80, val=0.10, seed=42):
    random.seed(seed)
    shuffled = data[:]
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = math.floor(n * train)
    n_val = math.floor(n * val)
    return (
        shuffled[:n_train],
        shuffled[n_train : n_train + n_val],
        shuffled[n_train + n_val :],
    )


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer"])
        writer.writeheader()
        for r in rows:
            writer.writerow({"question": r["q"], "answer": r["a"]})


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(build_instruction_format(r)) + "\n")


def upload_to_s3(local_path: Path, bucket: str, key: str) -> str:
    # COMPLIANCE NOTICE: Before uploading real training data to S3, operators MUST:
    # 1. Scrub all PII/PHI from training examples (names, SSNs, account numbers, DOBs).
    # 2. Verify that using customer data for model training is permitted under GDPR
    #    (Article 5 purpose limitation), GLBA (if US financial data), and any applicable
    #    local data protection laws.
    # 3. Apply appropriate S3 server-side encryption (SSE-KMS) and bucket policies.
    # The synthetic examples in this script contain no real customer data and are safe
    # to use as-is for demonstration. Operators are responsible for production data.
    s3 = boto3.client("s3")
    s3.upload_file(str(local_path), bucket, key)
    return f"s3://{bucket}/{key}"


def vocabulary_size(rows: list[dict]) -> int:
    tokens: set[str] = set()
    for r in rows:
        tokens.update(r["q"].lower().split())
        tokens.update(r["a"].lower().split())
    return len(tokens)


def stats(label: str, rows: list[dict]):
    prompts = [build_instruction_format(r)["prompt"] for r in rows]
    completions = [build_instruction_format(r)["completion"] for r in rows]
    avg_p = sum(len(p) for p in prompts) / len(prompts)
    avg_c = sum(len(c) for c in completions) / len(completions)
    print(f"  {label}: {len(rows)} examples | avg prompt {avg_p:.0f} chars | avg completion {avg_c:.0f} chars")


def main():
    bucket = os.environ.get("TRAINING_BUCKET")
    prefix = "financial-qa"
    out_dir = Path("data")

    train_rows, val_rows, test_rows = split_dataset(QA_PAIRS)

    # Write local files
    for label, rows, fmt in [
        ("train", train_rows, "train"),
        ("val", val_rows, "val"),
        ("test", test_rows, "test"),
    ]:
        write_csv(out_dir / f"{fmt}.csv", rows)
        write_jsonl(out_dir / f"{fmt}.jsonl", rows)

    print("\nDataset statistics:")
    stats("Train", train_rows)
    stats("Val", val_rows)
    stats("Test", test_rows)
    print(f"  Vocabulary size (full dataset): {vocabulary_size(QA_PAIRS)}")

    manifest = {
        "total_examples": len(QA_PAIRS),
        "splits": {
            "train": {"count": len(train_rows)},
            "val": {"count": len(val_rows)},
            "test": {"count": len(test_rows)},
        },
        "s3_paths": {},
    }

    if bucket:
        print(f"\nUploading to s3://{bucket}/{prefix}/")
        for fmt in ("train", "val", "test"):
            for ext in ("csv", "jsonl"):
                local = out_dir / f"{fmt}.{ext}"
                key = f"{prefix}/{fmt}.{ext}"
                uri = upload_to_s3(local, bucket, key)
                manifest["s3_paths"][f"{fmt}_{ext}"] = uri
                print(f"  Uploaded {uri}")
    else:
        print("\nTRAINING_BUCKET not set — skipping S3 upload. Files written to ./data/")
        for fmt in ("train", "val", "test"):
            for ext in ("csv", "jsonl"):
                manifest["s3_paths"][f"{fmt}_{ext}"] = f"./data/{fmt}.{ext}"

    with open("dataset_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("\nManifest written to dataset_manifest.json")


if __name__ == "__main__":
    main()
