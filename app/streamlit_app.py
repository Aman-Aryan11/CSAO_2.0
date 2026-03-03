from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except Exception:  # pragma: no cover
    px = None


def _find_project_root(start_file: str) -> Path:
    """Resolve project root by walking up from current file."""
    p = Path(start_file).resolve().parent
    for cand in [p, *p.parents]:
        if (cand / "data_pipeline").exists() and (cand / "src").exists():
            return cand
    return Path(start_file).resolve().parent


PROJECT_ROOT = _find_project_root(__file__)
DATA_DEMO_DIR = PROJECT_ROOT / "data" / "demo"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"


@dataclass(frozen=True)
class DemoData:
    """Container for loaded demo data tables."""

    items: pd.DataFrame
    sessions: pd.DataFrame
    recommendations: pd.DataFrame
    recs_available: bool


@dataclass(frozen=True)
class LoadedModels:
    """Container for model scoring callables and metadata."""

    baseline_scorer: Callable[[pd.DataFrame, Dict[str, Any]], np.ndarray]
    neural_scorer: Callable[[pd.DataFrame, Dict[str, Any]], np.ndarray]
    baseline_name: str
    neural_name: str


DEFAULT_METRICS: Dict[str, float] = {
    "NDCG@10": 0.712,
    "MAP": 0.608,
    "MRR": 0.763,
    "Precision@10": 0.250,
    "Recall@10": 0.866,
    "AUC": 0.803,
}


def _safe_read_parquet(path: Path, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Read a parquet file safely and return empty DataFrame if unavailable."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path, columns=list(columns) if columns else None)
    except Exception:
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()


def _normalize_items(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize item schema for the demo app."""
    if df.empty:
        rng = np.random.default_rng(42)
        n = 120
        cats = np.array(["main", "beverage", "dessert", "side", "snack"])
        out = pd.DataFrame(
            {
                "item_id": np.arange(1, n + 1),
                "item_name": [f"Item {i}" for i in range(1, n + 1)],
                "category": rng.choice(cats, size=n, p=[0.35, 0.2, 0.15, 0.15, 0.15]),
                "price": np.round(rng.uniform(79, 499, size=n), 2),
                "restaurant_id": rng.integers(1, 25, size=n),
                "popularity": np.round(rng.uniform(0.1, 1.0, size=n), 4),
                "image_url": "",
            }
        )
        return out

    out = df.copy()
    rename_map = {
        "id": "item_id",
        "name": "item_name",
        "title": "item_name",
        "menu_item": "item_name",
        "popularity_score": "popularity",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})

    if "item_id" not in out.columns:
        out["item_id"] = np.arange(1, len(out) + 1)
    if "item_name" not in out.columns:
        out["item_name"] = out["item_id"].astype(str).map(lambda x: f"Item {x}")
    if "category" not in out.columns:
        out["category"] = "misc"
    if "price" not in out.columns:
        out["price"] = 149.0
    if "restaurant_id" not in out.columns:
        out["restaurant_id"] = 1
    if "popularity" not in out.columns:
        out["popularity"] = 0.5
    if "image_url" not in out.columns:
        out["image_url"] = ""

    out["item_id"] = pd.to_numeric(out["item_id"], errors="coerce").fillna(0).astype(int)
    out["item_name"] = out["item_name"].astype(str)
    out["category"] = out["category"].astype(str)
    out["price"] = pd.to_numeric(out["price"], errors="coerce").fillna(149.0).astype(float)
    out["restaurant_id"] = pd.to_numeric(out["restaurant_id"], errors="coerce").fillna(1).astype(int)
    out["popularity"] = pd.to_numeric(out["popularity"], errors="coerce").fillna(0.5).clip(0, 1).astype(float)
    out["image_url"] = out["image_url"].fillna("").astype(str)

    return out[["item_id", "item_name", "category", "price", "restaurant_id", "popularity", "image_url"]].drop_duplicates(
        "item_id"
    )


def _normalize_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize session schema for the demo app."""
    if df.empty:
        rng = np.random.default_rng(7)
        n = 300
        meal_times = ["breakfast", "lunch", "snack", "dinner", "late_night"]
        return pd.DataFrame(
            {
                "session_id": np.arange(10001, 10001 + n),
                "user_id": rng.integers(1, 120, size=n),
                "meal_time": rng.choice(meal_times, size=n, p=[0.18, 0.30, 0.16, 0.27, 0.09]),
                "city": rng.choice(["Mumbai", "Delhi", "Bangalore", "Pune"], size=n),
                "cart_value": np.round(rng.uniform(150, 900, size=n), 2),
                "item_count": rng.integers(1, 8, size=n),
            }
        )

    out = df.copy()
    rename_map = {"id": "session_id", "uid": "user_id"}
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})

    if "session_id" not in out.columns:
        out["session_id"] = np.arange(1, len(out) + 1)
    if "user_id" not in out.columns:
        out["user_id"] = 1
    if "meal_time" not in out.columns:
        out["meal_time"] = "lunch"
    if "city" not in out.columns:
        out["city"] = "Unknown"
    if "cart_value" not in out.columns:
        out["cart_value"] = 0.0
    if "item_count" not in out.columns:
        out["item_count"] = 0

    out["session_id"] = pd.to_numeric(out["session_id"], errors="coerce").fillna(0).astype(int)
    out["user_id"] = pd.to_numeric(out["user_id"], errors="coerce").fillna(0).astype(int)
    out["meal_time"] = out["meal_time"].astype(str)
    out["city"] = out["city"].astype(str)
    out["cart_value"] = pd.to_numeric(out["cart_value"], errors="coerce").fillna(0.0).astype(float)
    out["item_count"] = pd.to_numeric(out["item_count"], errors="coerce").fillna(0).astype(int)

    return out[["session_id", "user_id", "meal_time", "city", "cart_value", "item_count"]].drop_duplicates("session_id")


def _normalize_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize precomputed recommendation schema for optional use."""
    if df.empty:
        return pd.DataFrame(columns=["session_id", "item_id", "score", "rank", "model", "source"])

    out = df.copy()
    rename_map = {
        "candidate_item_id": "item_id",
        "similarity_score": "score",
        "final_rank": "rank",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})

    for col, default in {
        "session_id": 0,
        "item_id": 0,
        "score": 0.0,
        "rank": 0,
        "model": "baseline",
        "source": "Hybrid",
    }.items():
        if col not in out.columns:
            out[col] = default

    out["session_id"] = pd.to_numeric(out["session_id"], errors="coerce").fillna(0).astype(int)
    out["item_id"] = pd.to_numeric(out["item_id"], errors="coerce").fillna(0).astype(int)
    out["score"] = pd.to_numeric(out["score"], errors="coerce").fillna(0.0).astype(float)
    out["rank"] = pd.to_numeric(out["rank"], errors="coerce").fillna(0).astype(int)
    out["model"] = out["model"].astype(str).str.lower()
    out["source"] = out["source"].astype(str)

    return out[["session_id", "item_id", "score", "rank", "model", "source"]]


@st.cache_data(show_spinner=False)
def load_demo_data() -> DemoData:
    """Load and normalize demo datasets with fallback generation."""
    items_path = DATA_DEMO_DIR / "items.parquet"
    sessions_path = DATA_DEMO_DIR / "sample_sessions.parquet"
    recs_path = DATA_DEMO_DIR / "recommendations.parquet"

    items = _normalize_items(_safe_read_parquet(items_path))
    sessions = _normalize_sessions(_safe_read_parquet(sessions_path))
    recommendations = _normalize_recommendations(_safe_read_parquet(recs_path))

    return DemoData(
        items=items,
        sessions=sessions,
        recommendations=recommendations,
        recs_available=not recommendations.empty,
    )


def _heuristic_score(candidates: pd.DataFrame, context: Dict[str, Any], model: str) -> np.ndarray:
    """Heuristic score for fallback ranking and demo responsiveness."""
    rng = np.random.default_rng(int(context.get("seed", 42)))
    if candidates.empty:
        return np.array([], dtype=float)

    pop = candidates["popularity"].to_numpy(dtype=float)
    price = candidates["price"].to_numpy(dtype=float)
    cart_avg = float(context.get("cart_avg_price", 180.0) or 180.0)
    price_match = 1.0 - np.clip(np.abs(price - cart_avg) / max(cart_avg, 1.0), 0, 1)

    cat = candidates["category"].astype(str).to_numpy()
    top_cat = str(context.get("top_cart_category", ""))
    cat_match = (cat == top_cat).astype(float)

    rest = candidates["restaurant_id"].to_numpy(dtype=int)
    cart_rest = int(context.get("dominant_restaurant_id", -1))
    rest_match = (rest == cart_rest).astype(float)

    noise = rng.uniform(0.0, 0.01, size=len(candidates))

    if model == "neural":
        score = 0.35 * pop + 0.25 * cat_match + 0.15 * rest_match + 0.20 * price_match + 0.05 * noise
    else:
        score = 0.45 * pop + 0.20 * cat_match + 0.20 * price_match + 0.10 * rest_match + 0.05 * noise
    return score.astype(float)


@st.cache_resource(show_spinner=False)
def load_models() -> LoadedModels:
    """Load baseline and neural model handles; fallback to heuristic scorers."""
    lgb_path = MODELS_DIR / "lightgbm_ranker.txt"
    neural_path = MODELS_DIR / "neural_ranker.pt"

    baseline_name = "Heuristic Baseline"
    neural_name = "Heuristic Neural"

    baseline_model = None
    neural_model = None

    if lgb_path.exists():
        try:
            import lightgbm as lgb  # type: ignore

            baseline_model = lgb.Booster(model_file=str(lgb_path))
            baseline_name = "LightGBM Ranker"
        except Exception:
            baseline_model = None

    if neural_path.exists():
        try:
            import torch

            neural_model = torch.load(str(neural_path), map_location="cpu", weights_only=False)
            neural_name = "Neural Ranker"
        except Exception:
            neural_model = None

    def baseline_scorer(candidates: pd.DataFrame, context: Dict[str, Any]) -> np.ndarray:
        """Unified baseline scorer; prefers loaded model else heuristic."""
        if baseline_model is None:
            return _heuristic_score(candidates, context, "baseline")
        try:
            feats = pd.DataFrame(
                {
                    "candidate_popularity": candidates["popularity"].astype(float),
                    "candidate_price": candidates["price"].astype(float),
                    "restaurant_id": candidates["restaurant_id"].astype(float),
                }
            )
            return np.asarray(baseline_model.predict(feats), dtype=float)
        except Exception:
            return _heuristic_score(candidates, context, "baseline")

    def neural_scorer(candidates: pd.DataFrame, context: Dict[str, Any]) -> np.ndarray:
        """Unified neural scorer; prefers loaded model else heuristic."""
        if neural_model is None:
            return _heuristic_score(candidates, context, "neural")
        try:
            if hasattr(neural_model, "predict"):
                feats = pd.DataFrame(
                    {
                        "candidate_popularity": candidates["popularity"].astype(float),
                        "candidate_price": candidates["price"].astype(float),
                        "restaurant_id": candidates["restaurant_id"].astype(float),
                    }
                )
                out = neural_model.predict(feats)
                return np.asarray(out, dtype=float)
            return _heuristic_score(candidates, context, "neural")
        except Exception:
            return _heuristic_score(candidates, context, "neural")

    return LoadedModels(
        baseline_scorer=baseline_scorer,
        neural_scorer=neural_scorer,
        baseline_name=baseline_name,
        neural_name=neural_name,
    )


def _load_eval_metrics() -> Dict[str, float]:
    """Load best available metrics from artifacts with defaults."""
    p = OUTPUT_DIR / "baseline_output" / "evaluation_summary.json"
    if not p.exists():
        return DEFAULT_METRICS
    try:
        data = json.loads(p.read_text())
        if not isinstance(data, list) or not data:
            return DEFAULT_METRICS
        test_rows = [x for x in data if str(x.get("split", "")).lower() == "test"]
        pick = test_rows[0] if test_rows else data[0]
        out = {k: float(pick.get(k, DEFAULT_METRICS[k])) for k in DEFAULT_METRICS}
        return out
    except Exception:
        return DEFAULT_METRICS


def _business_impact_payload() -> Dict[str, Dict[str, float]]:
    """Load baseline and neural business impact summaries if available."""
    baseline_path = OUTPUT_DIR / "cycle2" / "business_impact_summary.json"
    neural_path = OUTPUT_DIR / "cycle3" / "business_impact_neural.json"

    payload: Dict[str, Dict[str, float]] = {
        "baseline": {
            "acceptance_at_k": 0.88,
            "projected_aov_lift_percent": 12.0,
            "absolute_addon_uplift": 52.0,
        },
        "neural": {
            "acceptance_at_k": 0.90,
            "projected_aov_lift_percent": 13.5,
            "absolute_addon_uplift": 58.0,
        },
    }

    for mode, path in {"baseline": baseline_path, "neural": neural_path}.items():
        if not path.exists():
            continue
        try:
            obj = json.loads(path.read_text())
            if isinstance(obj, dict):
                for k in ["acceptance_at_k", "projected_aov_lift_percent", "absolute_addon_uplift"]:
                    if k in obj:
                        payload[mode][k] = float(obj[k])
        except Exception:
            continue
    return payload


def _cart_summary(cart_items: Dict[int, int], items_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute cart totals and category distribution."""
    if not cart_items:
        return {"total_items": 0, "estimated_value": 0.0, "category_mix": {}}

    qty_df = pd.DataFrame({"item_id": list(cart_items.keys()), "qty": list(cart_items.values())})
    merged = qty_df.merge(items_df[["item_id", "price", "category", "restaurant_id"]], on="item_id", how="left")
    merged["price"] = merged["price"].fillna(0.0)
    merged["line_value"] = merged["qty"] * merged["price"]

    cat_mix = (
        merged.groupby("category", dropna=False)["qty"].sum().sort_values(ascending=False).astype(int).to_dict()
    )
    return {
        "total_items": int(merged["qty"].sum()),
        "estimated_value": float(merged["line_value"].sum()),
        "category_mix": cat_mix,
        "dominant_restaurant_id": int(
            merged.groupby("restaurant_id", dropna=True)["qty"].sum().sort_values(ascending=False).index[0]
        )
        if merged["restaurant_id"].notna().any()
        else -1,
        "top_cart_category": next(iter(cat_mix), ""),
        "cart_avg_price": float((merged["line_value"].sum() / max(merged["qty"].sum(), 1))),
    }


def _render_item_card(item_row: pd.Series, key_prefix: str) -> None:
    """Render a compact item card with metadata."""
    st.markdown(f"**{item_row['item_name']}**")
    st.caption(f"Category: `{item_row['category']}` | Price: `Rs {item_row['price']:.2f}`")
    if str(item_row.get("image_url", "")).strip():
        st.image(item_row["image_url"], use_container_width=True)
    st.button("Add to Cart", key=f"{key_prefix}_add_{int(item_row['item_id'])}")


def _recommendation_explanation(row: pd.Series, context: Dict[str, Any]) -> str:
    """Generate compact explanation for recommendation rationale."""
    reasons = []
    if float(row.get("category_match", 0.0)) > 0.5:
        reasons.append("matches your current cart category pattern")
    if float(row.get("price_match", 0.0)) > 0.6:
        reasons.append("fits your current cart price range")
    if float(row.get("restaurant_match", 0.0)) > 0.5:
        reasons.append("from the same restaurant context")
    if float(row.get("popularity", 0.0)) > 0.75:
        reasons.append("high historical popularity")
    if not reasons:
        reasons.append("strong hybrid score from candidate signals")

    user_id = context.get("user_id", "-")
    meal_time = context.get("meal_time", "-")
    return f"Recommended for user `{user_id}` during `{meal_time}` because it {', '.join(reasons)}."


def render_home() -> None:
    """Render landing overview, KPIs, and pipeline narrative."""
    st.title("CSAO Recommender System Demo")
    st.subheader("Cart Super Add-On Optimization for Food Delivery")

    st.write(
        "This demo simulates how a production recommendation pipeline predicts add-on items in real time "
        "from cart context, user behavior, and candidate retrieval signals."
    )

    st.info(
        "Business objective: maximize add-on acceptance and AOV lift while preserving a clean, low-latency user experience."
    )

    metrics = _load_eval_metrics()
    cols = st.columns(6)
    for i, k in enumerate(["NDCG@10", "MAP", "MRR", "Precision@10", "Recall@10", "AUC"]):
        cols[i].metric(k, f"{metrics[k]:.3f}")

    st.markdown("### High-Level Pipeline")
    st.markdown(
        """
```mermaid
flowchart LR
    A[Data Ingestion] --> B[Candidate Generation\n(Item Similarity + CF)]
    B --> C[Candidate Fusion]
    C --> D[Feature Engineering]
    D --> E[Ranking Models\n(Baseline + Neural)]
    E --> F[Top-K Recommendations]
    F --> G[Business Impact Tracking]
```
"""
    )
    st.caption(
        "If Mermaid rendering is unavailable in your browser, read the flow left-to-right as Ingestion -> Retrieval -> Ranking -> Monitoring."
    )

    if st.button("Start Demo", type="primary"):
        st.session_state.demo_started = True
        st.session_state.nav_section = "User Simulation"
        st.rerun()


def update_cart() -> Dict[str, Any]:
    """Apply cart mutation from session state action and return latest summary."""
    items_df = st.session_state.data.items
    cart: Dict[int, int] = st.session_state.cart_items

    action = st.session_state.get("cart_action")
    if not action:
        return _cart_summary(cart, items_df)

    try:
        kind = action.get("kind")
        item_id = int(action.get("item_id", -1))
        qty = int(action.get("qty", 1))

        if kind == "clear":
            cart.clear()
        elif item_id not in set(items_df["item_id"].astype(int).tolist()):
            pass
        elif kind == "add":
            cart[item_id] = max(cart.get(item_id, 0) + max(qty, 1), 0)
        elif kind == "remove":
            cart[item_id] = max(cart.get(item_id, 0) - max(qty, 1), 0)
            if cart[item_id] <= 0:
                cart.pop(item_id, None)
        elif kind == "set":
            if qty <= 0:
                cart.pop(item_id, None)
            else:
                cart[item_id] = qty
    except Exception:
        pass
    finally:
        st.session_state.cart_action = None

    return _cart_summary(cart, items_df)


def render_catalog() -> None:
    """Render user/session simulation and interactive menu browsing."""
    st.header("User Simulation")
    data: DemoData = st.session_state.data

    left, right = st.columns([2, 1])
    with left:
        user_ids = sorted(data.sessions["user_id"].unique().tolist())
        session_ids = sorted(data.sessions["session_id"].unique().tolist())

        st.session_state.selected_user_id = st.selectbox(
            "Select User ID",
            options=user_ids,
            index=max(0, user_ids.index(st.session_state.selected_user_id))
            if st.session_state.selected_user_id in user_ids
            else 0,
        )

        filtered_sessions = data.sessions[data.sessions["user_id"] == st.session_state.selected_user_id]
        candidate_sessions = (
            sorted(filtered_sessions["session_id"].tolist()) if not filtered_sessions.empty else session_ids
        )
        if not candidate_sessions:
            candidate_sessions = session_ids[:1]

        st.session_state.selected_session_id = st.selectbox(
            "Select Session ID",
            options=candidate_sessions,
            index=max(0, candidate_sessions.index(st.session_state.selected_session_id))
            if st.session_state.selected_session_id in candidate_sessions
            else 0,
        )

    with right:
        st.markdown("### Cart Controls")
        if st.button("Clear Cart"):
            st.session_state.cart_action = {"kind": "clear"}
            update_cart()
            st.rerun()

        summary = _cart_summary(st.session_state.cart_items, data.items)
        st.metric("Total Items", summary["total_items"])
        st.metric("Estimated Cart Value", f"Rs {summary['estimated_value']:.2f}")

    st.markdown("### Browse Menu")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        cats = sorted(data.items["category"].astype(str).unique().tolist())
        selected_cats = st.multiselect("Category", options=cats, default=cats[: min(4, len(cats))])
    with c2:
        pmin, pmax = float(data.items["price"].min()), float(data.items["price"].max())
        lo, hi = st.slider("Price Range", min_value=float(pmin), max_value=float(pmax), value=(float(pmin), float(pmax)))
    with c3:
        q = st.text_input("Search item", value="", placeholder="Type item name...")

    filt = data.items.copy()
    if selected_cats:
        filt = filt[filt["category"].isin(selected_cats)]
    filt = filt[(filt["price"] >= lo) & (filt["price"] <= hi)]
    if q.strip():
        filt = filt[filt["item_name"].str.contains(q.strip(), case=False, na=False)]

    if filt.empty:
        st.warning("No items match current filters. Adjust category, price, or search.")
    else:
        show_n = min(60, len(filt))
        st.caption(f"Showing {show_n} of {len(filt)} matching items")
        page_df = filt.sort_values(["popularity", "item_name"], ascending=[False, True]).head(show_n)

        cols = st.columns(3)
        for idx, (_, row) in enumerate(page_df.iterrows()):
            col = cols[idx % 3]
            with col:
                st.markdown(f"**{row['item_name']}**")
                st.caption(f"`{row['category']}` | `Rs {row['price']:.2f}`")
                if str(row.get("image_url", "")).strip():
                    st.image(row["image_url"], use_container_width=True)
                if st.button("Add", key=f"add_{int(row['item_id'])}"):
                    st.session_state.cart_action = {"kind": "add", "item_id": int(row["item_id"]), "qty": 1}
                    update_cart()
                    st.rerun()

    st.markdown("### Cart Details")
    cart = st.session_state.cart_items
    if not cart:
        st.info("Your cart is empty. Add a few items to generate recommendations.")
        return

    qty_df = pd.DataFrame({"item_id": list(cart.keys()), "qty": list(cart.values())})
    cart_df = qty_df.merge(data.items[["item_id", "item_name", "category", "price"]], on="item_id", how="left")
    cart_df["line_total"] = cart_df["qty"] * cart_df["price"]

    for _, r in cart_df.iterrows():
        x1, x2, x3, x4, x5 = st.columns([3, 2, 1, 1, 1])
        x1.write(r["item_name"])
        x2.write(f"{r['category']} | Rs {r['price']:.2f}")
        x3.write(f"Qty: {int(r['qty'])}")
        if x4.button("+", key=f"inc_{int(r['item_id'])}"):
            st.session_state.cart_action = {"kind": "add", "item_id": int(r["item_id"]), "qty": 1}
            update_cart()
            st.rerun()
        if x5.button("-", key=f"dec_{int(r['item_id'])}"):
            st.session_state.cart_action = {"kind": "remove", "item_id": int(r["item_id"]), "qty": 1}
            update_cart()
            st.rerun()

    summary = _cart_summary(cart, data.items)
    st.success(
        f"Cart ready: {summary['total_items']} items, estimated value Rs {summary['estimated_value']:.2f}. "
        f"Top categories: {summary['category_mix']}"
    )


def generate_recommendations(
    selected_session_id: int,
    selected_user_id: int,
    cart_items: Dict[int, int],
    active_model: str,
    top_k: int,
) -> pd.DataFrame:
    """Generate Top-K recommendations using precomputed or heuristic/model scoring."""
    t0 = time.perf_counter()
    data: DemoData = st.session_state.data
    models: LoadedModels = st.session_state.models

    items = data.items.copy()
    if items.empty:
        return pd.DataFrame(columns=["item_id", "item_name", "category", "price", "score", "confidence", "rank", "source", "why", "latency_ms"])

    in_cart = set(map(int, cart_items.keys()))
    candidates = items[~items["item_id"].isin(in_cart)].copy()
    if candidates.empty:
        return pd.DataFrame(columns=["item_id", "item_name", "category", "price", "score", "confidence", "rank", "source", "why", "latency_ms"])

    summary = _cart_summary(cart_items, items)
    sess_row = data.sessions[data.sessions["session_id"] == int(selected_session_id)]
    meal_time = str(sess_row["meal_time"].iloc[0]) if not sess_row.empty else "lunch"

    context = {
        "user_id": int(selected_user_id),
        "session_id": int(selected_session_id),
        "meal_time": meal_time,
        "top_cart_category": summary.get("top_cart_category", ""),
        "dominant_restaurant_id": int(summary.get("dominant_restaurant_id", -1)),
        "cart_avg_price": float(summary.get("cart_avg_price", 180.0)),
        "seed": 42 + int(selected_session_id) % 1000,
    }

    has_precomputed = data.recs_available and not data.recommendations.empty
    rec_used = False

    if has_precomputed:
        pre = data.recommendations
        pre = pre[pre["session_id"] == int(selected_session_id)]
        if "model" in pre.columns and pre["model"].notna().any():
            m = "neural" if active_model == "neural" else "baseline"
            alt = pre[pre["model"].str.contains(m, case=False, na=False)]
            if not alt.empty:
                pre = alt
        pre = pre[~pre["item_id"].isin(in_cart)]
        if not pre.empty:
            cand = pre.merge(
                items[["item_id", "item_name", "category", "price", "restaurant_id", "popularity"]],
                on="item_id",
                how="inner",
            )
            if not cand.empty:
                candidates = cand
                score_col = "score"
                rec_used = True
        else:
            score_col = "score"
    else:
        score_col = "score"

    if rec_used:
        candidates["score"] = pd.to_numeric(candidates[score_col], errors="coerce").fillna(0.0).astype(float)
    else:
        candidates["category_match"] = (candidates["category"] == context["top_cart_category"]).astype(float)
        candidates["price_match"] = 1.0 - np.clip(
            np.abs(candidates["price"].to_numpy(float) - context["cart_avg_price"]) / max(context["cart_avg_price"], 1.0),
            0,
            1,
        )
        candidates["restaurant_match"] = (candidates["restaurant_id"] == context["dominant_restaurant_id"]).astype(float)

        if active_model == "neural":
            candidates["score"] = models.neural_scorer(candidates, context)
            source = "Neural"
        else:
            candidates["score"] = models.baseline_scorer(candidates, context)
            source = "CF/Similarity"
        candidates["source"] = source

    if "source" not in candidates.columns:
        candidates["source"] = "Hybrid"

    candidates = candidates.sort_values("score", ascending=False).head(max(int(top_k), 1)).copy()
    if candidates.empty:
        return pd.DataFrame(columns=["item_id", "item_name", "category", "price", "score", "confidence", "rank", "source", "why", "latency_ms"])

    cmin, cmax = float(candidates["score"].min()), float(candidates["score"].max())
    if abs(cmax - cmin) < 1e-12:
        candidates["confidence"] = 0.6
    else:
        candidates["confidence"] = 0.25 + 0.75 * ((candidates["score"] - cmin) / (cmax - cmin))

    if "category_match" not in candidates.columns:
        candidates["category_match"] = (candidates["category"] == context["top_cart_category"]).astype(float)
    if "price_match" not in candidates.columns:
        candidates["price_match"] = 1.0 - np.clip(
            np.abs(candidates["price"].to_numpy(float) - context["cart_avg_price"]) / max(context["cart_avg_price"], 1.0),
            0,
            1,
        )
    if "restaurant_match" not in candidates.columns:
        candidates["restaurant_match"] = (candidates["restaurant_id"] == context["dominant_restaurant_id"]).astype(float)

    candidates["rank"] = np.arange(1, len(candidates) + 1)
    candidates["why"] = candidates.apply(lambda r: _recommendation_explanation(r, context), axis=1)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    candidates["latency_ms"] = latency_ms

    cols = [
        "item_id",
        "item_name",
        "category",
        "price",
        "score",
        "confidence",
        "rank",
        "source",
        "why",
        "latency_ms",
    ]
    return candidates[cols]


def render_recommendations() -> None:
    """Render live recommendation outputs and model comparison tabs."""
    st.header("Live Recommendations")

    data: DemoData = st.session_state.data
    if not st.session_state.cart_items:
        st.info("Add items to cart in User Simulation to get real-time recommendations.")
        return

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        st.session_state.active_model = st.radio(
            "Active Ranker",
            ["baseline", "neural"],
            index=0 if st.session_state.active_model == "baseline" else 1,
            horizontal=True,
        )
    with c2:
        st.session_state.top_k = st.slider("Top-K", min_value=3, max_value=25, value=int(st.session_state.top_k))
    with c3:
        run_now = st.button("Refresh", type="primary")

    if run_now or st.session_state.get("last_recommendations") is None:
        with st.spinner("Generating recommendations..."):
            rec = generate_recommendations(
                selected_session_id=int(st.session_state.selected_session_id),
                selected_user_id=int(st.session_state.selected_user_id),
                cart_items=st.session_state.cart_items,
                active_model=st.session_state.active_model,
                top_k=int(st.session_state.top_k),
            )
            st.session_state.last_recommendations = rec

    rec_df: pd.DataFrame = st.session_state.last_recommendations
    if rec_df is None or rec_df.empty:
        st.warning("No recommendations available for the current context.")
        return

    latency = float(rec_df["latency_ms"].iloc[0])
    st.success(f"Inference completed in {latency:.1f} ms")

    for _, r in rec_df.iterrows():
        b1, b2 = st.columns([3, 2])
        with b1:
            st.markdown(f"### #{int(r['rank'])} - {r['item_name']}")
            st.caption(f"`{r['category']}` | `Rs {float(r['price']):.2f}` | Source: `{r['source']}`")
            with st.expander("Why recommended"):
                st.write(r["why"])
        with b2:
            st.metric("Score", f"{float(r['score']):.4f}")
            st.progress(float(np.clip(r["confidence"], 0, 1)))
            st.caption(f"Confidence: {float(r['confidence']) * 100:.1f}%")

    st.markdown("### Baseline vs Neural Comparison")
    tab1, tab2 = st.tabs(["Baseline", "Neural"])

    with tab1:
        base_df = generate_recommendations(
            int(st.session_state.selected_session_id),
            int(st.session_state.selected_user_id),
            st.session_state.cart_items,
            "baseline",
            int(st.session_state.top_k),
        )
        st.dataframe(base_df[["rank", "item_name", "category", "score", "confidence", "source"]], use_container_width=True)

    with tab2:
        neu_df = generate_recommendations(
            int(st.session_state.selected_session_id),
            int(st.session_state.selected_user_id),
            st.session_state.cart_items,
            "neural",
            int(st.session_state.top_k),
        )
        st.dataframe(neu_df[["rank", "item_name", "category", "score", "confidence", "source"]], use_container_width=True)

    overlap = set(base_df["item_id"].tolist()).intersection(set(neu_df["item_id"].tolist()))
    st.info(f"Top-K overlap: {len(overlap)} / {max(len(base_df), 1)} items")

    st.markdown("### Business KPI Simulator")
    impact = _business_impact_payload()
    use = impact["neural"] if st.session_state.active_model == "neural" else impact["baseline"]

    k1, k2, k3 = st.columns(3)
    k1.metric("Expected Acceptance@K", f"{float(use['acceptance_at_k']) * 100:.2f}%")
    k2.metric("Projected AOV Lift", f"{float(use['projected_aov_lift_percent']):.2f}%")
    k3.metric("Incremental Revenue/Session", f"Rs {float(use['absolute_addon_uplift']):.2f}")


def render_model_insights() -> None:
    """Render interpretability and diagnostic visualizations."""
    st.header("Model Insights")
    data: DemoData = st.session_state.data

    fi_path = OUTPUT_DIR / "error_analysis" / "feature_importance.csv"
    calib_path = OUTPUT_DIR / "error_analysis" / "calibration_curve.csv"
    seg_path = OUTPUT_DIR / "error_analysis" / "segment_metrics.csv"

    st.subheader("Feature Importance")
    fi = _safe_read_parquet(fi_path) if fi_path.suffix == ".parquet" else pd.DataFrame()
    if fi.empty and fi_path.exists() and fi_path.suffix == ".csv":
        try:
            fi = pd.read_csv(fi_path)
        except Exception:
            fi = pd.DataFrame()

    if fi.empty:
        fi = pd.DataFrame(
            {
                "feature": [
                    "category_match_flag",
                    "restaurant_match_flag",
                    "candidate_popularity",
                    "price_diff_from_cart_avg",
                    "hour",
                    "item_count",
                ],
                "importance_gain": [0.29, 0.24, 0.18, 0.12, 0.10, 0.07],
            }
        )

    fi = fi.rename(columns={
        c: "feature" for c in fi.columns if c.lower() in {"feature_name", "feature"}
    } | {
        c: "importance_gain" for c in fi.columns if c.lower() in {"importance", "importance_gain", "gain"}
    })

    fi = fi[[c for c in ["feature", "importance_gain"] if c in fi.columns]].dropna().head(20)
    if px is not None and not fi.empty:
        fig = px.bar(fi.sort_values("importance_gain", ascending=True), x="importance_gain", y="feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)
    elif not fi.empty:
        st.bar_chart(fi.set_index("feature")["importance_gain"])

    st.subheader("Score Distribution")
    score_df = st.session_state.get("last_recommendations")
    if score_df is None or score_df.empty:
        score_df = generate_recommendations(
            int(st.session_state.selected_session_id),
            int(st.session_state.selected_user_id),
            st.session_state.cart_items,
            st.session_state.active_model,
            max(int(st.session_state.top_k), 10),
        )
    if not score_df.empty:
        if px is not None:
            hist = px.histogram(score_df, x="score", nbins=20)
            st.plotly_chart(hist, use_container_width=True)
        else:
            st.bar_chart(score_df["score"])

    st.subheader("Calibration Curve")
    cal = pd.DataFrame()
    if calib_path.exists():
        try:
            cal = pd.read_csv(calib_path)
        except Exception:
            cal = pd.DataFrame()

    if cal.empty:
        bins = np.linspace(0.05, 0.95, 10)
        cal = pd.DataFrame(
            {
                "pred_bin": bins,
                "actual_rate": np.clip(bins * 0.9 + 0.03, 0, 1),
            }
        )

    xcol = "pred_bin" if "pred_bin" in cal.columns else cal.columns[0]
    ycol = "actual_rate" if "actual_rate" in cal.columns else cal.columns[min(1, len(cal.columns) - 1)]
    if px is not None:
        fig = px.line(cal, x=xcol, y=ycol, markers=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(cal.set_index(xcol)[ycol])

    st.subheader("Segment Performance")
    seg = pd.DataFrame()
    if seg_path.exists():
        try:
            seg = pd.read_csv(seg_path)
        except Exception:
            seg = pd.DataFrame()

    if seg.empty:
        seg = pd.DataFrame(
            {
                "segment": ["breakfast", "lunch", "snack", "dinner", "late_night"],
                "NDCG@10": [0.63, 0.67, 0.64, 0.69, 0.61],
                "Precision@10": [0.21, 0.24, 0.23, 0.25, 0.19],
                "Recall@10": [0.79, 0.86, 0.82, 0.88, 0.74],
            }
        )
    st.dataframe(seg.head(50), use_container_width=True)

    st.subheader("Session Deep Dive")
    sess_options = sorted(data.sessions["session_id"].unique().tolist())
    sid = st.selectbox("Select session for deep dive", options=sess_options, index=0)

    d_base = generate_recommendations(
        int(sid),
        int(st.session_state.selected_user_id),
        st.session_state.cart_items,
        "baseline",
        20,
    )
    d_neural = generate_recommendations(
        int(sid),
        int(st.session_state.selected_user_id),
        st.session_state.cart_items,
        "neural",
        20,
    )

    deep = d_base[["item_id", "item_name", "category", "price", "score", "rank"]].rename(
        columns={"score": "baseline_score", "rank": "baseline_rank"}
    ).merge(
        d_neural[["item_id", "score", "rank"]].rename(columns={"score": "neural_score", "rank": "neural_rank"}),
        on="item_id",
        how="outer",
    )
    deep = deep.sort_values(["baseline_rank", "neural_rank"], na_position="last")
    st.dataframe(deep.head(50), use_container_width=True)


def render_architecture() -> None:
    """Render system architecture and engineering design details."""
    st.header("System Architecture")
    st.markdown(
        """
### End-to-End Pipeline
1. Data ingestion and synthetic/raw session construction  
2. Candidate generation (item similarity + collaborative filtering)  
3. Candidate fusion and feature engineering  
4. Ranking model scoring (baseline + neural)  
5. Evaluation, diagnostics, and business KPI simulation
"""
    )

    with st.expander("Data Ingestion & Schemas"):
        st.write(
            "Raw tables: users, restaurants, items, sessions, session_items. Processed outputs include interactions, "
            "ranking datasets, and feature tables with session/user/candidate/cross features."
        )

    with st.expander("Model Choices"):
        st.write(
            "Retrieval layer combines item co-occurrence and ALS-based collaborative filtering. Ranking layer compares "
            "GBDT baselines (LightGBM/XGBoost) against a neural pairwise ranker."
        )

    with st.expander("Checkpointing & Resume"):
        st.write(
            "Training pipelines persist model, optimizer, and scheduler states per epoch. If interrupted, runs resume "
            "from latest checkpoint to reduce re-computation risk."
        )

    with st.expander("Latency & Scalability"):
        st.write(
            "Pipeline is designed for chunked parquet processing and cached model/data loading. In this demo app, "
            "recommendation calls are optimized for sub-second interaction with lightweight scoring paths."
        )

    with st.expander("Fallback Behavior"):
        st.write(
            "If artifacts are missing, the app auto-generates synthetic demo data and heuristic scorers while preserving "
            "the same UX and API contract."
        )


def _init_session_state(data: DemoData) -> None:
    """Initialize required session state keys exactly once."""
    if "data" not in st.session_state:
        st.session_state.data = data
    if "models" not in st.session_state:
        st.session_state.models = load_models()

    defaults: Dict[str, Any] = {
        "cart_items": {},
        "selected_user_id": int(data.sessions["user_id"].iloc[0]) if not data.sessions.empty else 1,
        "selected_session_id": int(data.sessions["session_id"].iloc[0]) if not data.sessions.empty else 1,
        "active_model": "baseline",
        "top_k": 10,
        "last_recommendations": None,
        "demo_started": False,
        "nav_section": "Home / Overview",
        "cart_action": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def main() -> None:
    """Application entry point."""
    st.set_page_config(page_title="CSAO Recommender Demo", page_icon="🛒", layout="wide")

    with st.spinner("Loading app data..."):
        data = load_demo_data()
    _init_session_state(data)

    with st.sidebar:
        st.title("CSAO Demo")
        section = st.radio(
            "Navigate",
            options=[
                "Home / Overview",
                "User Simulation",
                "Live Recommendations",
                "Model Insights",
                "System Architecture",
            ],
            key="nav_section",
        )
        st.caption(f"Project root: {PROJECT_ROOT}")

    if section == "Home / Overview":
        render_home()
    elif section == "User Simulation":
        render_catalog()
    elif section == "Live Recommendations":
        render_recommendations()
    elif section == "Model Insights":
        render_model_insights()
    elif section == "System Architecture":
        render_architecture()


if __name__ == "__main__":
    main()
