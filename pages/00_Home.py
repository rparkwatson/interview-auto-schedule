import streamlit as st

st.set_page_config(page_title="Home", layout="wide")

st.title("🏠 Home")

has_file = "formatted_xlsx" in st.session_state

# Quick status
if has_file:
    st.success("Formatted workbook is ready from Step 1.")
else:
    st.info("No formatted workbook yet. Start with Step 1, or upload manually in Step 2.")

# Cards/links
c1, c2 = st.columns(2)

with c1:
    st.subheader("Step 1 — Build Workbook")
    st.write("Upload interviewer & adcom files, set Max Pairs, generate **formatted.xlsx**.")

    if st.button("Open Step 1 →", type="primary"):
        st.switch_page("pages/01_Builder.py")

with c2:
    st.subheader("Step 2 — Run Scheduler")
    st.write("Use the formatted workbook to run the solver, scan scenarios, and export reports.")

    # enable only if Step 1 produced bytes
    disabled = not has_file

    # High-contrast button (uses your theme's primary color)
    if st.button("Open Step 2 →", type="primary", disabled=disabled):
        # Requires Streamlit ≥ 1.37
        st.switch_page("pages/02_Scheduler.py")

    if disabled:
        st.caption("Complete Step 1 to enable.")

# Optional: one-click continue
if has_file and st.button("Continue where I left off"):
    try:
        st.switch_page("pages/02_Scheduler.py")
    except Exception:
        # Older Streamlit without switch_page — the page link above still works
        st.info("Use the Step 2 link above.")