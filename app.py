# app.py  — Home
import streamlit as st

st.set_page_config(page_title="Interview Scheduling")

home = st.Page("pages/00_Home.py", title="Home")
builder = st.Page("pages/01_Builder.py", title="Step 1 — Build Workbook")
scheduler = st.Page("pages/02_Scheduler.py", title="Step 2 — Run Scheduler")  # <-- match your real filename

nav = st.navigation([home, builder, scheduler])  # shows in the sidebar by default
nav.run()
