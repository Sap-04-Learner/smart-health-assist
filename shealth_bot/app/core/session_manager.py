import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Any, Literal

# Import settings from config
from ..config.setting import get_settings


Role = Literal["user", "assistant"]


class SessionManager:
    """
    Manages user session history with secure file storage.
    Prevents directory traversal and validates all inputs.
    """
    def __init__(self, storage_path: str | None = None):
        """
        Initialize SessionManager.
        
        Args:
            storage_path: Directory path for storing session files.
                          Defaults to settings.session_storage_path
        """
        settings = get_settings()
        self.storage_path = storage_path or settings.session_storage_path
        os.makedirs(self.storage_path, exist_ok=True)

        # Pre-compile regex for performance
        self._id_pattern = re.compile(r'^[a-zA-Z0-9_-]{1,128}$')
        self._session_id_pattern = re.compile(r'^[a-zA-Z0-9_-]{1,64}$')


    def _validate_id(self, identifier: str, pattern: re.Pattern, name: str) -> None:
        """Validate identifier using pre-compiled pattern."""
        if not pattern.match(identifier):
            raise ValueError(
                f"Invalid {name}: must be 1+ alphanumeric, underscore, or dash chars. "
                f"Got: {identifier!r}"
            )


    def _get_session_file(self, user_id: str, session_id: str) -> Path:
        """Generate safe session file path with traversal protection."""
        self._validate_id(user_id, self._id_pattern, "user_id")
        self._validate_id(session_id, self._session_id_pattern, "session_id")

        file_path = (Path(self.storage_path) / f"{user_id}_{session_id}.json").resolve()
        storage_root = Path(self.storage_path).resolve()

        if not file_path.is_relative_to(storage_root):
            raise ValueError(f"Path traversal attempt detected: {file_path}")

        return file_path


    def get_history(
        self,
        user_id: str,
        session_id: str,
        max_messages: int = 50  
    ) -> list[dict[str, Any]]:
        """Retrieve last N messages from session history."""
        session_file = self._get_session_file(user_id, session_id)

        if not session_file.exists():
            return []

        try:
            with session_file.open('r', encoding='utf-8') as f:
                history = json.load(f)

            # Validate history is list of dicts
            if not isinstance(history, list):
                return []

            return history[-max_messages:] if len(history) > max_messages else history
        except (json.JSONDecodeError, OSError):
            return []


    def save_message(
        self,
        user_id: str,
        session_id: str,
        role: Role,
        content: str,
        max_history: int = 200  # Prevent unbounded growth
    ) -> None:
        """Save a message and optionally truncate old history."""
        if role not in {"user", "assistant"}:
            raise ValueError(f"Invalid role: {role}. Must be 'user' or 'assistant'")

        if not isinstance(content, str):
            raise ValueError("Content must be a string")

        session_file = self._get_session_file(user_id, session_id)

        # Load existing
        history = []
        if session_file.exists():
            try:
                with session_file.open('r', encoding='utf-8') as f:
                    history = json.load(f)
                if not isinstance(history, list):
                    history = []
            except (json.JSONDecodeError, OSError):
                history = []

        # Append new message
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        history.append(message)

        # Truncate to prevent infinite growth
        if len(history) > max_history:
            history = history[-max_history:]

        # Save
        try:
            with session_file.open('w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        except OSError as e:
            raise IOError(f"Failed to save session: {e}")


    def clear_session(self, user_id: str, session_id: str) -> None:
        """Delete entire session file."""
        session_file = self._get_session_file(user_id, session_id)
        if session_file.exists():
            try:
                session_file.unlink()
            except OSError:
                pass  # Best effort


    def list_sessions(self, user_id: str) -> list[str]:
        """List all session IDs for a user."""
        self._validate_id(user_id, self._id_pattern, "user_id")
        prefix = f"{user_id}_"

        sessions = []
        try:
            for entry in os.scandir(self.storage_path):
                if (entry.is_file()
                    and entry.name.startswith(prefix)
                    and entry.name.endswith('.json')):
                    session_id = entry.name[len(prefix):-5]  # Remove prefix + .json
                    sessions.append(session_id)
        except OSError:
            return []

        return sessions