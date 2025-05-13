import streamlit as st
import re
from typing import Optional, Dict, Any, Tuple

# Import the database connection
from utils import get_database_session


def validate_email(email: str) -> bool:
    """
    Validates that a string is a properly formatted email address.

    Args:
        email: The email address to validate

    Returns:
        True if the email is valid, False otherwise
    """
    # Use a simple regex pattern for email validation
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def validate_password(password: str) -> Tuple[bool, str]:
    """
    Validates that a password meets security requirements.

    Args:
        password: The password to validate

    Returns:
        A tuple of (is_valid, message) where message explains any validation issues
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"

    # Check for at least one uppercase letter, one lowercase letter, and one digit
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"

    if not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"

    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one digit"

    return True, "Password meets requirements"


def init_session_state():
    """Initialize session state variables for user authentication"""
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "username" not in st.session_state:
        st.session_state.username = None
    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False
    if "is_logged_in" not in st.session_state:
        st.session_state.is_logged_in = False


def login_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Authenticates a user using database authentication.

    Args:
        username: The username to authenticate
        password: The password to verify

    Returns:
        Dict with user information if login successful, None otherwise
    """
    init_session_state()

    # If already logged in, return current user info
    if st.session_state.is_logged_in:
        return {
            "id": st.session_state.user_id,
            "username": st.session_state.username,
            "is_admin": st.session_state.is_admin,
            "is_logged_in": True
        }

    # Get database instance
    db = get_database_session()

    # Authenticate user
    user_id = db.authenticate_user(username, password)

    if user_id:
        # Get user details
        user_info = db.get_user(user_id)

        # Set session state
        st.session_state.user_id = user_id
        st.session_state.username = user_info['username']
        st.session_state.is_admin = user_info['is_admin']
        st.session_state.is_logged_in = True

        return {
            "id": user_id,
            "username": user_info['username'],
            "email": user_info['email'],
            "is_admin": user_info['is_admin'],
            "is_logged_in": True
        }

    return None


def logout_user():
    """
    Logs out the current user by clearing their session state.
    """
    if "user_id" in st.session_state:
        st.session_state.user_id = None
    if "username" in st.session_state:
        st.session_state.username = None
    if "is_admin" in st.session_state:
        st.session_state.is_admin = False
    if "is_logged_in" in st.session_state:
        st.session_state.is_logged_in = False


def register_user(username: str, email: str, password: str, confirm_password: str) -> Tuple[bool, str]:
    """
    Registers a new user in the database.

    Args:
        username: The desired username
        email: The user's email address
        password: The user's password
        confirm_password: Password confirmation to prevent typos

    Returns:
        Tuple of (success, message) where message provides details about the result
    """
    # Input validation
    if not username or len(username) < 3:
        return False, "Username must be at least 3 characters"

    if not validate_email(email):
        return False, "Please enter a valid email address"

    if password != confirm_password:
        return False, "Passwords do not match"

    password_valid, password_msg = validate_password(password)
    if not password_valid:
        return False, password_msg

    # Get database instance
    db = get_database_session()

    # Try to create user
    try:
        user_id = db.create_user(username, email, password)
        return True, f"User {username} registered successfully! You can now log in."
    except ValueError as e:
        # Handle existing username/email
        return False, str(e)
    except Exception as e:
        # Handle other errors
        return False, f"Registration failed: {str(e)}"


def change_user_password(current_password: str, new_password: str, confirm_password: str) -> Tuple[bool, str]:
    """
    Changes the password for the currently logged-in user.

    Args:
        current_password: The user's current password
        new_password: The new password
        confirm_password: Confirmation of the new password

    Returns:
        Tuple of (success, message) where message provides details about the result
    """
    if not st.session_state.is_logged_in:
        return False, "You must be logged in to change your password"

    if new_password != confirm_password:
        return False, "New passwords do not match"

    password_valid, password_msg = validate_password(new_password)
    if not password_valid:
        return False, password_msg

    # Get database instance
    db = get_database_session()

    # Try to change password
    success = db.user_manager.change_password(
        st.session_state.user_id,
        current_password,
        new_password
    )

    if success:
        return True, "Password changed successfully!"
    else:
        return False, "Current password is incorrect"


def display_login_ui():
    """
    Displays a login UI with tabs for login and registration.
    """
    init_session_state()

    if not st.session_state.is_logged_in:

        # Create tabs for login and registration
        login_tab, register_tab = st.tabs(["Login", "Register"])

        # Login tab
        with login_tab:
            username = st.text_input("Username", key="login_username")
            password = st.text_input(
                "Password", type="password", key="login_password")

            login_button = st.button("Login", use_container_width=True)

            if login_button and username and password:
                user_info = login_user(username, password)
                if user_info:
                    st.success(f"Welcome back, {user_info['username']}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")

        # Registration tab
        with register_tab:
            reg_username = st.text_input("Username", key="reg_username")
            reg_email = st.text_input("Email", key="reg_email")
            reg_password = st.text_input(
                "Password", type="password", key="reg_password")
            reg_confirm_password = st.text_input(
                "Confirm Password", type="password", key="reg_confirm_password")

            register_button = st.button(
                "Register", use_container_width=True)

            if register_button:
                if reg_username and reg_email and reg_password and reg_confirm_password:
                    success, message = register_user(
                        reg_username, reg_email, reg_password, reg_confirm_password
                    )
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                else:
                    st.warning("Please fill out all fields")
    else:
        # User is logged in - show user info and logout option
        st.write(f"Logged in as: **{st.session_state.username}**")

        # Create tabs for user actions
        profile_tab, password_tab = st.tabs(["Profile", "Change Password"])

        # Profile tab
        with profile_tab:
            st.write(f"Username: {st.session_state.username}")
            st.write(f"Admin: {'Yes' if st.session_state.is_admin else 'No'}")

            if st.button("Log Out"):
                logout_user()
                st.success("Logged out successfully!")
                st.rerun()

        # Password change tab
        with password_tab:
            current_pwd = st.text_input(
                "Current Password", type="password", key="current_pwd")
            new_pwd = st.text_input(
                "New Password", type="password", key="new_pwd")
            confirm_pwd = st.text_input(
                "Confirm New Password", type="password", key="confirm_pwd")

            if st.button("Change Password"):
                if current_pwd and new_pwd and confirm_pwd:
                    success, message = change_user_password(
                        current_pwd, new_pwd, confirm_pwd)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                else:
                    st.warning("Please fill out all fields")


def get_current_user() -> Optional[Dict[str, Any]]:
    """
    Get information about the currently logged-in user.

    Returns:
        Dict with user information if a user is logged in, None otherwise
    """
    init_session_state()

    if st.session_state.is_logged_in:
        return {
            "id": st.session_state.user_id,
            "username": st.session_state.username,
            "is_admin": st.session_state.is_admin,
            "is_logged_in": True
        }
    return None


def require_login():
    """
    Ensures that a user is logged in. If not, displays the login UI and halts further execution.

    This function is useful for pages that require authentication.
    """
    init_session_state()

    if not st.session_state.is_logged_in:
        display_login_ui()
        st.stop()
