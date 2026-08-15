---
name: Streamlit Preview startup
description: Replit workflow details needed for a reliable Streamlit Preview.
---

Streamlit can stop at its first-run email/onboarding prompt before opening the configured port. Keep usage statistics disabled and headless mode enabled in `.streamlit/config.toml`; bind the workflow command explicitly to `0.0.0.0` and the configured Preview port.

**Why:** A workflow can appear configured correctly but still fail its port readiness check if Streamlit waits for interactive terminal input.

**How to apply:** For Streamlit apps, verify both the workflow's `openPorts`/status and `/_stcore/health` after restarting. Use a successful HTTP response rather than searching for app text in Streamlit's shell HTML.