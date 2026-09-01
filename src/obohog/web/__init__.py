"""The HTTP layer: FastAPI JSON API and HTMX HTML pages.

Everything here is adapter code over :mod:`obohog.service` — imports of
FastAPI/Jinja2 stay inside this package so the core installs without the
``web`` extra.
"""
