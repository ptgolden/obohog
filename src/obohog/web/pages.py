"""HTML routes: full pages plus the HTMX search-results fragment.

Templates consume the service layer's models directly (op spans →
``<del>``/``<ins>``, commit metadata pre-styled), so this module is only
routing and query-string plumbing.
"""

from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .. import service
from .api import Handle

router = APIRouter(include_in_schema=False)

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
templates.env.filters["dateonly"] = lambda value: str(value)[:10]


def render_error(request: Request, status: int, detail: str) -> HTMLResponse:
    """Error content-negotiated for the HTML side: a bare fragment for
    HTMX requests (swapped into the page), a full page otherwise."""
    name = (
        "partials/error.html"
        if request.headers.get("HX-Request")
        else "error.html"
    )
    return templates.TemplateResponse(
        request, name, {"status": status, "detail": detail}, status_code=status
    )


@router.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request,
        "sources.html",
        {"sources": service.list_sources(request.app.state.config)},
    )


@router.get("/{src}/terms/{term_id}", response_class=HTMLResponse)
def term_page(
    request: Request,
    handle: Handle,
    src: str,
    term_id: str,
    tag: str | None = None,
    since: str | None = None,
    limit: int | None = Query(None, ge=1),
    full: bool = False,
):
    db, style = handle
    timeline = service.get_timeline(
        db, style, term_id,
        tag=tag, since=since, limit=limit, full=full,
    )
    if timeline is None:
        raise HTTPException(status_code=404, detail=f"no history for {term_id!r}")
    return templates.TemplateResponse(
        request, "term.html", {"src": src, "timeline": timeline}
    )


def _fragment_query_string(params: service.SearchParams) -> str:
    """The current search restated as a query string, minus the cursor —
    the load-more sentinel appends its own ``after``."""
    fields = params.model_dump(exclude_defaults=True, exclude={"after"})
    return urlencode(
        {k: (str(v).lower() if isinstance(v, bool) else v) for k, v in fields.items()}
    )


class _FormParams(service.SearchParams):
    """The search form before a query is typed: ``q`` optional.

    ``order`` additionally accepts the form's combined values: a select
    can only set one param, so ``newest``/``oldest`` mean date order
    with the commit-time direction baked in (date order is newest-first
    by default, ``reverse`` makes it oldest-first).
    """

    q: str | None = None
    order: Literal["term", "date", "newest", "oldest"] = "term"

    def to_search_params(self) -> service.SearchParams:
        data = self.model_dump()
        if data["order"] == "newest":
            data["order"], data["reverse"] = "date", False
        elif data["order"] == "oldest":
            data["order"], data["reverse"] = "date", True
        return service.SearchParams(**data)


@router.get("/{src}/search", response_class=HTMLResponse)
def search_page(
    request: Request,
    handle: Handle,
    src: str,
    params: Annotated[_FormParams, Query()] = None,
):
    db, style = handle
    context: dict = {
        "src": src,
        "params": None,
        "page": None,
        "facets": request.app.state.registry.facets(src),
    }
    if params is not None and params.q:
        sp = params.to_search_params()
        context["params"] = sp
        context["page"] = service.search(db, style, sp)
        context["qs"] = _fragment_query_string(sp)
    return templates.TemplateResponse(request, "search.html", context)


@router.get("/{src}/search/results", response_class=HTMLResponse)
def search_results(
    request: Request,
    handle: Handle,
    src: str,
    params: Annotated[service.SearchParams, Query()],
):
    db, style = handle
    page = service.search(db, style, params)
    return templates.TemplateResponse(
        request,
        "partials/search_results.html",
        {
            "src": src,
            "params": params,
            "page": page,
            "qs": _fragment_query_string(params),
        },
    )


@router.get("/{src}/commits/{sha}", response_class=HTMLResponse)
def commit_page(
    request: Request,
    handle: Handle,
    src: str,
    sha: str,
    namespace: str | None = None,
    full: bool = False,
):
    db, style = handle
    view = service.get_commit(db, style, sha, namespace=namespace, full=full)
    if view is None:
        raise HTTPException(
            status_code=404, detail=f"no indexed changes for commit {sha!r}"
        )
    return templates.TemplateResponse(
        request, "commit.html", {"src": src, "view": view}
    )


@router.get("/{src}/prs/{number}", response_class=HTMLResponse)
def pr_page(request: Request, handle: Handle, src: str, number: int):
    db, style = handle
    pr = service.get_pr(db, style, number)
    if pr is None:
        raise HTTPException(
            status_code=404, detail=f"no indexed changes for PR #{number}"
        )
    return templates.TemplateResponse(request, "pr.html", {"src": src, "pr": pr})
