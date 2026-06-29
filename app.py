"""Main Streamlit entry point with simple role-based login."""

from pathlib import Path
import sys

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from auth import (  # noqa: E402
    ROLE_PAGES,
    initialize_session_state,
    render_login_screen,
    render_sidebar_navigation,
)


st.set_page_config(page_title="Student Gaming Academic Dashboard", layout="wide")

initialize_session_state()

if not st.session_state.logged_in:
    render_login_screen()
    st.stop()

render_sidebar_navigation()

st.title("Student Gaming Academic Dashboard")
st.write(f"Welcome, {st.session_state.username}.")
st.write(f"Role: {st.session_state.role}")

st.subheader("Available Pages")
for label, page_path in ROLE_PAGES.get(st.session_state.role, []):
    st.page_link(page_path, label=label)
