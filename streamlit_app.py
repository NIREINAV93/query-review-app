import json
import os
import re
import base64
import logging

import streamlit as st
import sqlglot
import pandas as pd
from dotenv import load_dotenv
from snowflake.snowpark import Session
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

load_dotenv(override=True)

st.set_page_config(page_title="Query Review", layout="wide")

DATA_PATH = "data/query_annotated.json"
FEEDBACK_TABLE = "machine_learning.ai_agent.query_review_feedback"

logger = logging.getLogger(__name__)


# --- Snowflake connection ---

def _create_session() -> Session:
    private_key_b64 = os.environ.get("SNOWFLAKE_PRIVATE_KEY")
    if not private_key_b64:
        raise ValueError("SNOWFLAKE_PRIVATE_KEY environment variable is required.")

    private_key_pem = base64.b64decode(private_key_b64).decode("utf-8")
    private_key = serialization.load_pem_private_key(
        private_key_pem.encode("utf-8"),
        password=None,
        backend=default_backend(),
    )
    private_key_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    params = {
        "account": os.environ.get("SNOWFLAKE_ACCOUNT", "riversidefm-analytics"),
        "user": os.environ.get("SNOWFLAKE_USER", "ML_PROD_AWS"),
        "role": os.environ.get("SNOWFLAKE_ROLE", "AI_TOOLS"),
        "private_key": private_key_bytes,
    }
    warehouse = os.environ.get("SNOWFLAKE_WAREHOUSE", "").strip()
    if warehouse:
        params["warehouse"] = warehouse

    return Session.builder.configs(params).create()


def _get_session():
    if "sf_session" not in st.session_state:
        st.session_state.sf_session = _create_session()
    return st.session_state.sf_session


def run_query(sql: str) -> list[dict]:
    session = _get_session()
    df = session.sql(sql)
    columns = df.columns
    rows = df.collect()
    if not rows:
        return []
    return [{col: row[i] for i, col in enumerate(columns)} for row in rows]


def run_query_raw(sql: str) -> None:
    session = _get_session()
    session.sql(sql).collect()


# --- Data helpers ---

@st.cache_data
def load_queries():
    with open(DATA_PATH) as f:
        return json.load(f)


def fetch_existing_feedback(reviewer):
    rows = run_query(
        f"SELECT cluster_id FROM {FEEDBACK_TABLE} WHERE reviewer = '{reviewer}'"
    )
    return {r["CLUSTER_ID"] for r in rows}


def submit_feedback(
    cluster_id,
    reviewer,
    question_status,
    question_feedback,
    query_status,
    query_feedback,
    business_question,
    query_category,
):
    def esc(s):
        return s.replace("'", "''") if s else ""

    run_query_raw(
        f"""
        MERGE INTO {FEEDBACK_TABLE} AS t
        USING (SELECT '{esc(cluster_id)}' AS cluster_id, '{esc(reviewer)}' AS reviewer) AS s
        ON t.cluster_id = s.cluster_id AND t.reviewer = s.reviewer
        WHEN MATCHED THEN UPDATE SET
            question_status = '{esc(question_status)}',
            question_feedback = '{esc(question_feedback)}',
            query_status = '{esc(query_status)}',
            query_feedback = '{esc(query_feedback)}',
            business_question = '{esc(business_question)}',
            query_category = '{esc(query_category)}',
            reviewed_at = CURRENT_TIMESTAMP()
        WHEN NOT MATCHED THEN INSERT
            (cluster_id, reviewer, question_status, question_feedback,
             query_status, query_feedback, business_question, query_category)
        VALUES (
            '{esc(cluster_id)}', '{esc(reviewer)}',
            '{esc(question_status)}', '{esc(question_feedback)}',
            '{esc(query_status)}', '{esc(query_feedback)}',
            '{esc(business_question)}', '{esc(query_category)}'
        )
        """
    )


# --- App ---

def main():
    queries = load_queries()

    all_authors = sorted({a for q in queries for a in q.get("authors", [])})

    # --- Sidebar ---
    st.sidebar.title("Query Review")
    reviewer = st.sidebar.selectbox("Analyst", [""] + all_authors)
    if not reviewer:
        st.info("Select your name in the sidebar to begin.")
        return

    # Filter queries to this author — analysis only
    author_queries = [
        q for q in queries
        if reviewer in q.get("authors", []) and q.get("query_category") == "analysis"
    ]
    if not author_queries:
        st.warning(f"No queries found for {reviewer}.")
        return

    reviewed_ids = fetch_existing_feedback(reviewer)

    # Review status filter
    review_filter = st.sidebar.radio(
        "Review status", ["All", "Pending", "Reviewed"], horizontal=True
    )

    filtered = author_queries
    if review_filter == "Pending":
        filtered = [q for q in filtered if q["cluster_id"] not in reviewed_ids]
    elif review_filter == "Reviewed":
        filtered = [q for q in filtered if q["cluster_id"] in reviewed_ids]

    # Progress
    reviewed_count = sum(1 for q in author_queries if q["cluster_id"] in reviewed_ids)
    st.sidebar.markdown(f"**Progress:** {reviewed_count} / {len(author_queries)} reviewed")
    st.sidebar.progress(reviewed_count / len(author_queries) if author_queries else 0)

    if not filtered:
        st.info("No queries match the current filters.")
        return

    # --- Navigation ---
    if "query_idx" not in st.session_state:
        st.session_state.query_idx = 0
    idx = st.session_state.query_idx
    idx = max(0, min(idx, len(filtered) - 1))

    col_prev, col_counter, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("← Previous", disabled=idx == 0):
            st.session_state.query_idx = idx - 1
            st.rerun()
    with col_counter:
        st.markdown(f"### Query {idx + 1} / {len(filtered)}")
    with col_next:
        if st.button("Next →", disabled=idx == len(filtered) - 1):
            st.session_state.query_idx = idx + 1
            st.rerun()

    q = filtered[idx]
    is_reviewed = q["cluster_id"] in reviewed_ids

    # --- Query display ---
    if is_reviewed:
        st.success("Already reviewed")

    raw_sql = q.get("query_text", "")

    # Truncate long IN (...) lists for display
    def truncate_in_lists(sql):
        def _replace(m):
            ids = m.group(1)
            count = ids.count("'") // 2 or ids.count(",") + 1
            return f"IN (/* ... {count} IDs ... */)"
        return re.sub(r"IN\s*\((\s*'[^)]{500,})\)", _replace, sql, flags=re.IGNORECASE | re.DOTALL)

    display_sql = truncate_in_lists(raw_sql)
    try:
        display_sql = sqlglot.transpile(display_sql, read="snowflake", pretty=True)[0]
    except Exception:
        pass
    if len(display_sql) > 3000:
        with st.expander(f"SQL ({len(raw_sql.splitlines())} lines — click to expand)"):
            st.code(display_sql, language="sql")
    else:
        st.code(display_sql, language="sql")

    st.markdown("**LLM-generated business question** — does this accurately describe what the query above does?\n\n")
    st.info(
        f"*\"{q.get('business_question', 'N/A')}\"*"
    )

    # --- Review form ---
    st.markdown("---")
    st.subheader("Review")

    col_q, col_s = st.columns(2)

    with col_q:
        st.markdown("**Business Question Alignment**")
        q_status = st.radio(
            "Does the business question match the SQL?",
            ["Aligned", "Misaligned"],
            key=f"q_status_{q['cluster_id']}",
            horizontal=True,
        )
        q_feedback = ""
        if q_status == "Misaligned":
            q_feedback = st.text_area(
                "What should the business question be?",
                key=f"q_feedback_{q['cluster_id']}",
            )

    with col_s:
        st.markdown("**Query Correctness**")
        s_status = st.radio(
            "Is the SQL correct and trustworthy?",
            ["Correct", "Incorrect"],
            key=f"s_status_{q['cluster_id']}",
            horizontal=True,
        )
        s_feedback = ""
        if s_status == "Incorrect":
            s_feedback = st.text_area(
                "What's wrong with this query?",
                key=f"s_feedback_{q['cluster_id']}",
            )

    # Validation
    can_submit = True
    if q_status == "Misaligned" and not q_feedback.strip():
        can_submit = False
    if s_status == "Incorrect" and not s_feedback.strip():
        can_submit = False

    if st.button("Submit Review", type="primary", disabled=not can_submit):
        submit_feedback(
            cluster_id=q["cluster_id"],
            reviewer=reviewer,
            question_status=q_status.lower(),
            question_feedback=q_feedback,
            query_status=s_status.lower(),
            query_feedback=s_feedback,
            business_question=q.get("business_question", ""),
            query_category=q.get("query_category", ""),
        )
        st.toast("Review submitted!", icon="✅")
        if idx < len(filtered) - 1:
            st.session_state.query_idx = idx + 1
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()
