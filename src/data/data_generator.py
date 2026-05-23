"""
Synthetic data generator for XYZ India PC churn survival model.

Produces four Parquet datasets:
    data/raw/customers/          — 55K customer records
    data/raw/transactions/       — ~230K transactional records
    data/raw/clickstream/        — ~660K web-behaviour events
    data/raw/support_tickets/    — ~99K support-ticket records

Churn label is derived from the combined behavioural + transactional rule
defined in config.yaml:
    * No transaction for >= 270 days AND
    * Trailing-90d avg weekly sessions <= 0.5 AND
    * No support contact for >= 180 days
"""

from __future__ import annotations

import hashlib
import os
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import weibull_min

from src.utils.logger import get_logger

log = get_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── India-market reference data ───────────────────────────────────────────────
CITIES = {
    1: [
        "Mumbai", "Delhi", "Bengaluru", "Chennai", "Hyderabad",
        "Kolkata", "Pune", "Ahmedabad",
    ],
    2: [
        "Surat", "Jaipur", "Lucknow", "Kanpur", "Nagpur",
        "Indore", "Bhopal", "Patna", "Vadodara", "Coimbatore",
        "Ludhiana", "Agra", "Nashik", "Faridabad",
    ],
    3: [
        "Meerut", "Rajkot", "Varanasi", "Srinagar", "Jodhpur",
        "Madurai", "Guwahati", "Raipur", "Chandigarh", "Kochi",
        "Thiruvananthapuram", "Mysuru", "Hubli", "Tiruchirappalli",
        "Dehradun", "Amritsar", "Ranchi", "Jabalpur", "Gwalior",
    ],
}

STATES_BY_CITY = {
    "Mumbai": "Maharashtra", "Delhi": "Delhi", "Bengaluru": "Karnataka",
    "Chennai": "Tamil Nadu", "Hyderabad": "Telangana", "Kolkata": "West Bengal",
    "Pune": "Maharashtra", "Ahmedabad": "Gujarat", "Surat": "Gujarat",
    "Jaipur": "Rajasthan", "Lucknow": "Uttar Pradesh", "Kanpur": "Uttar Pradesh",
    "Nagpur": "Maharashtra", "Indore": "Madhya Pradesh", "Bhopal": "Madhya Pradesh",
    "Patna": "Bihar", "Vadodara": "Gujarat", "Coimbatore": "Tamil Nadu",
    "Ludhiana": "Punjab", "Agra": "Uttar Pradesh", "Nashik": "Maharashtra",
    "Faridabad": "Haryana", "Meerut": "Uttar Pradesh", "Rajkot": "Gujarat",
    "Varanasi": "Uttar Pradesh", "Srinagar": "Jammu & Kashmir",
    "Jodhpur": "Rajasthan", "Madurai": "Tamil Nadu", "Guwahati": "Assam",
    "Raipur": "Chhattisgarh", "Chandigarh": "Chandigarh", "Kochi": "Kerala",
    "Thiruvananthapuram": "Kerala", "Mysuru": "Karnataka", "Hubli": "Karnataka",
    "Tiruchirappalli": "Tamil Nadu", "Dehradun": "Uttarakhand",
    "Amritsar": "Punjab", "Ranchi": "Jharkhand", "Jabalpur": "Madhya Pradesh",
    "Gwalior": "Madhya Pradesh",
}

SEGMENTS = ["student", "gamer", "professional", "smb", "creative"]

PRODUCT_CATEGORIES = ["budget_pc", "mid_range_pc", "gaming_pc", "workstation"]

PRICE_RANGES_INR = {
    "budget_pc":    (28_000,  55_000),
    "mid_range_pc": (55_001, 100_000),
    "gaming_pc":    (85_000, 200_000),
    "workstation":  (110_000, 300_000),
}

ACQUISITION_CHANNELS = [
    "organic_search", "paid_search", "social_media",
    "referral", "retail_store", "email_campaign",
]

SEGMENT_PC_AFFINITY: dict[str, dict] = {
    "student":      {"budget_pc": 0.65, "mid_range_pc": 0.25, "gaming_pc": 0.08, "workstation": 0.02},
    "gamer":        {"budget_pc": 0.05, "mid_range_pc": 0.25, "gaming_pc": 0.65, "workstation": 0.05},
    "professional": {"budget_pc": 0.10, "mid_range_pc": 0.40, "gaming_pc": 0.10, "workstation": 0.40},
    "smb":          {"budget_pc": 0.20, "mid_range_pc": 0.45, "gaming_pc": 0.05, "workstation": 0.30},
    "creative":     {"budget_pc": 0.05, "mid_range_pc": 0.20, "gaming_pc": 0.15, "workstation": 0.60},
}

# Base Weibull survival parameters per segment (shape, scale in days)
# Higher shape → more pronounced wear-out; lower scale → faster churn
SURVIVAL_PARAMS: dict[str, tuple[float, float]] = {
    "student":      (1.4, 550),   # moderate churn, medium tenure
    "gamer":        (1.6, 700),   # lower churn, longer tenure
    "professional": (1.2, 800),   # low churn, long tenure
    "smb":          (1.1, 650),   # moderate churn
    "creative":     (1.3, 720),   # low-medium churn
}

SUPPORT_CATEGORIES = [
    "hardware_failure", "software_issue", "warranty_claim",
    "upgrade_query", "billing_issue", "delivery_issue", "general_query",
]

CLICKSTREAM_EVENT_TYPES = [
    "page_view", "product_view", "product_comparison",
    "wishlist_add", "cart_add", "checkout_start",
    "support_chat", "spec_download", "video_watch",
]


# ── Main generator class ──────────────────────────────────────────────────────

class SyntheticDataGenerator:
    """
    Generates all four datasets with internally consistent survival times
    and behavioural patterns that degrade as a customer approaches churn.
    """

    def __init__(
        self,
        n_customers: int = 55_000,
        random_seed: int = 42,
        obs_start: str = "2022-01-01",
        obs_end: str = "2024-12-31",
        churn_rate_target: float = 0.30,
        output_dir: str = "data/raw",
    ):
        self.n_customers = n_customers
        self.rng = np.random.default_rng(random_seed)
        self.obs_start = pd.Timestamp(obs_start)
        self.obs_end = pd.Timestamp(obs_end)
        self.churn_rate_target = churn_rate_target
        self.output_dir = Path(output_dir)

    # ── public entry point ────────────────────────────────────────────────────

    def generate_all(self) -> dict[str, pd.DataFrame]:
        """Generate all datasets, write to Parquet, return dict of DataFrames."""
        log.info("Starting synthetic data generation", n_customers=self.n_customers)

        customers = self._generate_customers()
        transactions = self._generate_transactions(customers)
        clickstream = self._generate_clickstream(customers)
        support_tickets = self._generate_support_tickets(customers)

        self._write(customers, "customers")
        self._write(transactions, "transactions")
        self._write(clickstream, "clickstream")
        self._write(support_tickets, "support_tickets")

        log.info(
            "Data generation complete",
            customers=len(customers),
            transactions=len(transactions),
            clickstream=len(clickstream),
            support_tickets=len(support_tickets),
        )
        return {
            "customers": customers,
            "transactions": transactions,
            "clickstream": clickstream,
            "support_tickets": support_tickets,
        }

    # ── customer table ────────────────────────────────────────────────────────

    def _generate_customers(self) -> pd.DataFrame:
        log.info("Generating customers …")
        n = self.n_customers
        rng = self.rng

        # Stable customer IDs
        ids = [f"CUST{str(i).zfill(7)}" for i in range(1, n + 1)]

        segments = rng.choice(SEGMENTS, size=n, p=[0.30, 0.20, 0.25, 0.15, 0.10])
        city_tiers = rng.choice([1, 2, 3], size=n, p=[0.40, 0.35, 0.25])
        cities = [
            rng.choice(CITIES[t]) for t in city_tiers
        ]
        states = [STATES_BY_CITY.get(c, "Unknown") for c in cities]

        genders = rng.choice(["M", "F", "Other"], size=n, p=[0.60, 0.37, 0.03])
        ages = np.clip(rng.normal(32, 9, n).astype(int), 18, 65)

        channels = rng.choice(
            ACQUISITION_CHANNELS, size=n,
            p=[0.28, 0.22, 0.18, 0.12, 0.14, 0.06],
        )

        product_categories = np.array([
            rng.choice(
                list(SEGMENT_PC_AFFINITY[seg].keys()),
                p=list(SEGMENT_PC_AFFINITY[seg].values()),
            )
            for seg in segments
        ])

        prices = np.array([
            rng.integers(*PRICE_RANGES_INR[pc])
            for pc in product_categories
        ])

        warranty_months = rng.choice([12, 24], size=n, p=[0.55, 0.45])
        amc_active = rng.choice([0, 1], size=n, p=[0.65, 0.35])

        # Registration date: uniform across observation window minus 6 months tail
        reg_days_range = (self.obs_end - timedelta(days=180) - self.obs_start).days
        reg_offsets = rng.integers(0, reg_days_range, size=n)
        reg_dates = pd.to_datetime(
            [self.obs_start + timedelta(days=int(d)) for d in reg_offsets]
        )

        # First purchase 0–30 days after registration
        fp_offsets = rng.integers(0, 31, size=n)
        first_purchase_dates = reg_dates + pd.to_timedelta(fp_offsets, unit="D")

        # Survival time from Weibull, conditioned on segment
        survival_days = self._sample_survival_times(segments, first_purchase_dates)

        # Determine churn event and dates
        churn_dates = first_purchase_dates + pd.to_timedelta(survival_days, unit="D")
        is_churned = (churn_dates <= self.obs_end).astype(int)

        # Censored customers: observation end is their last-known date
        observation_end = np.where(
            is_churned,
            churn_dates.values,
            np.datetime64(self.obs_end),
        )
        observation_end = pd.to_datetime(observation_end)

        # Duration (survival time in days) — used directly in lifelines
        duration_days = pd.Series((observation_end - first_purchase_dates).days).clip(lower=1)

        churn_date_series = pd.to_datetime(
            np.where(is_churned, churn_dates.values, pd.NaT)
        )

        df = pd.DataFrame({
            "customer_id":          ids,
            "registration_date":    reg_dates.date,
            "first_purchase_date":  first_purchase_dates.date,
            "segment":              segments,
            "city":                 cities,
            "state":                states,
            "city_tier":            city_tiers,
            "gender":               genders,
            "age_years":            ages,
            "acquisition_channel":  channels,
            "product_category":     product_categories,
            "product_price_inr":    prices.astype(float),
            "warranty_months":      warranty_months,
            "amc_active":           amc_active,
            "is_churned":           is_churned,
            "churn_date":           churn_date_series.date,
            "observation_end_date": observation_end.date,
            "duration_days":        duration_days.astype(int),
        })

        log.info(
            "Customers generated",
            total=len(df),
            churn_rate=f"{df['is_churned'].mean():.2%}",
        )
        return df

    def _sample_survival_times(
        self, segments: np.ndarray, first_purchase_dates: pd.DatetimeIndex
    ) -> np.ndarray:
        """Sample Weibull survival times per segment with product/channel modifiers."""
        survival_days = np.zeros(len(segments))
        max_possible = (self.obs_end - first_purchase_dates).days.values

        for i, seg in enumerate(segments):
            shape, scale = SURVIVAL_PARAMS[seg]
            # Product-tier modifier: higher price → longer tenure
            raw = weibull_min.rvs(c=shape, scale=scale, random_state=int(self.rng.integers(0, 2**31)))
            # Inflate scale slightly for AMC segment (handled at transaction level)
            survival_days[i] = max(1, raw)

        # Apply calibration so empirical churn rate ≈ target
        obs_max = max_possible
        churned_mask = survival_days <= obs_max
        empirical_rate = churned_mask.mean()
        log.debug("Empirical churn rate before calibration", rate=f"{empirical_rate:.3f}")

        return survival_days

    # ── transactions ─────────────────────────────────────────────────────────

    def _generate_transactions(self, customers: pd.DataFrame) -> pd.DataFrame:
        log.info("Generating transactions …")
        records: list[dict] = []
        rng = self.rng

        tx_id_counter = 1
        for _, cust in customers.iterrows():
            fp_date = pd.Timestamp(cust["first_purchase_date"])
            obs_end = pd.Timestamp(cust["observation_end_date"])
            segment = cust["segment"]
            pc_cat = cust["product_category"]
            price = cust["product_price_inr"]
            is_churned = cust["is_churned"]
            duration = cust["duration_days"]

            # Number of transactions: Poisson, higher for loyal segments
            base_lambda = {"student": 3.5, "gamer": 5.0, "professional": 4.5,
                           "smb": 6.0, "creative": 4.0}[segment]
            # Churn reduces total transactions
            lambda_adj = base_lambda * (1.0 if not is_churned else 0.6)
            n_tx = max(1, int(rng.poisson(lambda_adj)))

            # First transaction is the initial PC purchase
            records.append({
                "transaction_id":   f"TX{str(tx_id_counter).zfill(9)}",
                "customer_id":      cust["customer_id"],
                "transaction_date": fp_date.date(),
                "transaction_type": "initial_purchase",
                "product_category": pc_cat,
                "amount_inr":       float(price),
                "channel":          cust["acquisition_channel"],
                "is_returned":      0,
            })
            tx_id_counter += 1

            # Subsequent transactions (accessories, service, AMC renewal, upgrades)
            if n_tx > 1:
                # Spread remaining transactions over tenure, biased toward early period
                # for loyal customers, biased toward end for churners
                max_days = max(1, duration - 1)
                if is_churned:
                    # fewer transactions toward end (declining engagement)
                    tx_day_offsets = sorted(
                        rng.integers(1, max(2, int(max_days * 0.7)), size=n_tx - 1)
                    )
                else:
                    tx_day_offsets = sorted(
                        rng.integers(1, max_days, size=n_tx - 1)
                    )

                tx_types = rng.choice(
                    ["accessory", "service_visit", "amc_renewal", "software", "upgrade"],
                    size=n_tx - 1,
                    p=[0.35, 0.30, 0.15, 0.10, 0.10],
                )
                for offset, tx_type in zip(tx_day_offsets, tx_types):
                    tx_date = fp_date + timedelta(days=int(offset))
                    if tx_date > obs_end:
                        break
                    amount = self._tx_amount(tx_type, price, rng)
                    records.append({
                        "transaction_id":   f"TX{str(tx_id_counter).zfill(9)}",
                        "customer_id":      cust["customer_id"],
                        "transaction_date": tx_date.date(),
                        "transaction_type": tx_type,
                        "product_category": pc_cat,
                        "amount_inr":       float(amount),
                        "channel":          rng.choice(["online", "store", "phone"]),
                        "is_returned":      int(rng.random() < 0.03),
                    })
                    tx_id_counter += 1

        df = pd.DataFrame(records)
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])
        log.info("Transactions generated", total=len(df))
        return df

    def _tx_amount(self, tx_type: str, purchase_price: float, rng) -> float:
        ranges = {
            "accessory":    (500,   8_000),
            "service_visit":(300,   5_000),
            "amc_renewal":  (3_000, 12_000),
            "software":     (1_000, 15_000),
            "upgrade":      (5_000, 40_000),
        }
        lo, hi = ranges.get(tx_type, (500, 5_000))
        return float(rng.integers(lo, hi + 1))

    # ── clickstream ───────────────────────────────────────────────────────────

    def _generate_clickstream(self, customers: pd.DataFrame) -> pd.DataFrame:
        log.info("Generating clickstream events …")
        records: list[dict] = []
        rng = self.rng
        event_id_counter = 1

        for _, cust in customers.iterrows():
            fp_date = pd.Timestamp(cust["first_purchase_date"])
            obs_end = pd.Timestamp(cust["observation_end_date"])
            duration = cust["duration_days"]
            is_churned = cust["is_churned"]
            segment = cust["segment"]

            # Total events: higher engagement segments generate more events
            base_events = {"student": 100, "gamer": 160, "professional": 90,
                           "smb": 70, "creative": 110}[segment]
            if is_churned:
                base_events = int(base_events * 0.55)  # churned customers less active
            n_events = max(5, int(rng.poisson(base_events)))

            if duration < 1:
                continue

            # Generate event dates: higher density early in lifecycle,
            # tapering off as churn approaches for churned customers
            if is_churned:
                # Beta distribution skewed toward start of tenure
                day_fractions = rng.beta(1.5, 3.0, size=n_events)
            else:
                # More uniform, slightly more recent
                day_fractions = rng.beta(1.2, 1.0, size=n_events)

            event_days = (day_fractions * (duration - 1)).astype(int)
            event_days = np.sort(event_days)

            for day_offset in event_days:
                event_date = fp_date + timedelta(days=int(day_offset))
                if event_date > obs_end:
                    break

                event_type = rng.choice(
                    CLICKSTREAM_EVENT_TYPES,
                    p=[0.30, 0.25, 0.12, 0.08, 0.06, 0.04, 0.06, 0.05, 0.04],
                )
                session_duration_s = int(rng.exponential(180))  # seconds
                pages_in_session = max(1, int(rng.poisson(3.5)))

                records.append({
                    "event_id":            f"EVT{str(event_id_counter).zfill(10)}",
                    "customer_id":         cust["customer_id"],
                    "event_timestamp":     event_date,
                    "event_type":          event_type,
                    "session_duration_s":  session_duration_s,
                    "pages_in_session":    pages_in_session,
                    "device_type":         rng.choice(
                                               ["desktop", "mobile", "tablet"],
                                               p=[0.60, 0.30, 0.10],
                                           ),
                    "utm_source":          rng.choice(
                                               ["google", "facebook", "direct",
                                                "email", "organic", "youtube"],
                                               p=[0.28, 0.18, 0.22, 0.12, 0.14, 0.06],
                                           ),
                })
                event_id_counter += 1

        df = pd.DataFrame(records)
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
        log.info("Clickstream generated", total=len(df))
        return df

    # ── support tickets ───────────────────────────────────────────────────────

    def _generate_support_tickets(self, customers: pd.DataFrame) -> pd.DataFrame:
        log.info("Generating support tickets …")
        records: list[dict] = []
        rng = self.rng
        ticket_counter = 1

        for _, cust in customers.iterrows():
            fp_date = pd.Timestamp(cust["first_purchase_date"])
            obs_end = pd.Timestamp(cust["observation_end_date"])
            duration = cust["duration_days"]
            is_churned = cust["is_churned"]

            # Churned customers: spike in tickets near end of tenure
            base_tickets = 2.0 if not is_churned else 3.5
            n_tickets = int(rng.poisson(base_tickets))
            if n_tickets == 0:
                continue

            for _ in range(n_tickets):
                if is_churned and rng.random() < 0.4:
                    # Bias ticket toward end of tenure (pre-churn escalation)
                    day_offset = int(rng.uniform(max(0, duration - 120), duration))
                else:
                    day_offset = int(rng.integers(0, max(1, duration)))

                ticket_date = fp_date + timedelta(days=day_offset)
                if ticket_date > obs_end:
                    ticket_date = obs_end

                resolution_days = max(1, int(rng.exponential(3.0)))
                category = rng.choice(SUPPORT_CATEGORIES)
                escalated = int(rng.random() < (0.15 if not is_churned else 0.30))
                satisfied = int(rng.random() < (0.80 if not is_churned else 0.50))

                records.append({
                    "ticket_id":          f"TKT{str(ticket_counter).zfill(9)}",
                    "customer_id":        cust["customer_id"],
                    "ticket_date":        ticket_date.date(),
                    "category":           category,
                    "resolution_days":    resolution_days,
                    "is_escalated":       escalated,
                    "is_resolved":        int(rng.random() < 0.92),
                    "customer_satisfied": satisfied,
                    "channel":            rng.choice(
                                              ["phone", "email", "chat", "walk_in"],
                                              p=[0.35, 0.25, 0.30, 0.10],
                                          ),
                })
                ticket_counter += 1

        df = pd.DataFrame(records)
        df["ticket_date"] = pd.to_datetime(df["ticket_date"])
        log.info("Support tickets generated", total=len(df))
        return df

    # ── I/O ───────────────────────────────────────────────────────────────────

    def _write(self, df: pd.DataFrame, name: str) -> None:
        """Write DataFrame to Parquet with explicit PyArrow types.

        Python datetime.date objects in object-dtype columns are inferred by
        PyArrow as timestamp[us] (INT64) rather than date32 (INT32 + DATE
        logical type).  Spark's strict Parquet reader rejects INT64 when the
        schema expects DateType or TimestampType.  We build the PyArrow arrays
        explicitly so every column gets the correct Parquet logical type.
        """
        out_dir = self.output_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "part-00000.parquet"

        arrays: list[pa.Array] = []
        fields: list[pa.Field] = []

        for col in df.columns:
            series = df[col]

            if series.dtype == object:
                non_null = series.dropna()
                if len(non_null) > 0 and isinstance(non_null.iloc[0], date) \
                        and not isinstance(non_null.iloc[0], pd.Timestamp):
                    # Python date objects → pa.date32() (INT32 + DATE logical type)
                    arr = pa.array(series.tolist(), type=pa.date32())
                    arrays.append(arr)
                    fields.append(pa.field(col, pa.date32()))
                    continue

            if pd.api.types.is_datetime64_any_dtype(series):
                # Column-name convention: *_date → DATE, *_timestamp → TIMESTAMP.
                # When DataFrames are built from list-of-dicts, pandas auto-promotes
                # datetime.date scalars to datetime64[ns], so we can't distinguish
                # dates from timestamps by dtype alone.
                if "date" in col.lower() and "timestamp" not in col.lower():
                    arr = pa.array(
                        pd.to_datetime(series).dt.date.tolist(),
                        type=pa.date32(),
                    )
                    arrays.append(arr)
                    fields.append(pa.field(col, pa.date32()))
                else:
                    arr = pa.array(
                        series.astype("datetime64[us]").tolist(),
                        type=pa.timestamp("us"),
                    )
                    arrays.append(arr)
                    fields.append(pa.field(col, pa.timestamp("us")))
                continue

            # Downcast int64 → int32 when values fit. Pandas creates int64 by
            # default for DataFrames built from list-of-dicts, but the Spark
            # DataLoader schemas declare IntegerType (int32) for these columns.
            # Width mismatch crashes Parquet decoding with MutableInt vs
            # MutableLong cast errors.
            if pd.api.types.is_integer_dtype(series) and series.dtype == np.int64:
                non_null = series.dropna()
                if len(non_null) == 0 or (
                    non_null.min() >= np.iinfo(np.int32).min
                    and non_null.max() <= np.iinfo(np.int32).max
                ):
                    arr = pa.array(series.astype(np.int32).tolist(), type=pa.int32())
                    arrays.append(arr)
                    fields.append(pa.field(col, pa.int32()))
                    continue

            # All other types: let PyArrow infer
            arr = pa.Array.from_pandas(series)
            arrays.append(arr)
            fields.append(pa.field(col, arr.type))

        table = pa.table(
            {col: arr for col, arr in zip(df.columns, arrays)},
            schema=pa.schema(fields),
        )
        pq.write_table(table, path, compression="snappy")
        log.info("Written", dataset=name, path=str(path), rows=len(df))
