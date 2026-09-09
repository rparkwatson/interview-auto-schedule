"""Production Streamlit entry point for Interview Scheduler v2."""

import streamlit as st

from v2.interview_scheduler_v2.admin_app import main


def run_production_app() -> None:
    """Register only v2 and suppress automatic discovery of legacy pages."""

    page = st.Page(main, title="Interview Scheduler", default=True)
    navigation = st.navigation([page], position="hidden")
    navigation.run()


if __name__ == "__main__":
    run_production_app()
