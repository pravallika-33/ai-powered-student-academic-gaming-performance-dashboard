"""Simple session-state authentication and role-based access helpers."""

import streamlit as st


DEMO_USERS = {
    "admin": {"password": "admin123", "role": "Admin"},
    "analyst": {"password": "analyst123", "role": "Analyst"},
    "student": {"password": "student123", "role": "Student"},
}

ROLE_PAGES = {
    "Admin": [
        ("Analytics Dashboard", "pages/1_Analytics_Dashboard.py"),
        ("ML Analysis & Prediction", "pages/2_ML_Analysis_Prediction.py"),
    ],
    "Analyst": [
        ("Analytics Dashboard", "pages/1_Analytics_Dashboard.py"),
        ("ML Analysis & Prediction", "pages/2_ML_Analysis_Prediction.py"),
    ],
    "Student": [
        ("Student Prediction", "pages/3_Student_Prediction.py"),
    ],
}


def initialize_session_state() -> None:
    """Create the session-state keys used by the login system."""
    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("username", "")
    st.session_state.setdefault("role", "")


def login_user(username: str, password: str) -> bool:
    """Validate a hardcoded demo user and store login details in session state."""
    user = DEMO_USERS.get(username)

    if not user or user["password"] != password:
        return False

    st.session_state.logged_in = True
    st.session_state.username = username
    st.session_state.role = user["role"]
    return True


def logout_user() -> None:
    """Clear login details from session state."""
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.role = ""


def render_login_screen() -> None:
    """Display the login form when no user is authenticated."""
    initialize_session_state()

    st.title("Student Gaming Academic Dashboard")
    st.subheader("Login")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        if login_user(username.strip(), password):
            st.success("Login successful.")
            st.rerun()
        else:
            st.error("Invalid username or password.")

    st.info("Demo users: admin/admin123, analyst/analyst123, student/student123")


def render_sidebar_navigation() -> None:
    """Display logout controls and role-based page links in the sidebar."""
    initialize_session_state()

    st.sidebar.write(f"User: {st.session_state.username}")
    st.sidebar.write(f"Role: {st.session_state.role}")

    if st.sidebar.button("Logout"):
        logout_user()
        st.switch_page("app.py")

    st.sidebar.divider()
    st.sidebar.subheader("Navigation")

    for label, page_path in ROLE_PAGES.get(st.session_state.role, []):
        st.sidebar.page_link(page_path, label=label)


def require_login() -> None:
    """Stop page rendering when the user is not logged in."""
    initialize_session_state()

    if not st.session_state.logged_in:
        render_login_screen()
        st.stop()


def require_roles(allowed_roles: list[str]) -> None:
    """Stop page rendering when the logged-in user's role is not allowed."""
    require_login()

    if st.session_state.role not in allowed_roles:
        render_sidebar_navigation()
        st.error("You do not have permission to access this page.")
        st.stop()
