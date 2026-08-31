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

_STYLE = Path(__file__).parent / "static" / "style.css"
# Cache-buster for the stylesheet link: the URL changes whenever the file
# does, so browsers can cache hard yet never serve a stale sheet.
templates.env.globals["style_v"] = lambda: int(_STYLE.stat().st_mtime)


def _commit_noun(style) -> str:
    """What one history point is called in copy: synthetic histories
    (releases, BioPortal) have versions, git sources have commits."""
    return "version" if style.hide_sha else "commit"


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
        request,
        "term.html",
        {"src": src, "timeline": timeline, "commit_noun": _commit_noun(style)},
    )


@router.get("/{src}/terms/{term_id}/state", response_class=HTMLResponse)
def state_page(
    request: Request, handle: Handle, src: str, term_id: str, at: str
):
    """The reconstructed term stanza as of a ref — full page normally, a
    bare fragment for the HTMX inline expand on commit headers."""
    db, _ = handle
    state = service.get_state(db, term_id, at)
    if state is None:
        raise HTTPException(
            status_code=404,
            detail=f"{term_id!r} has no snapshot at or before {at!r}",
        )
    fragment = bool(request.headers.get("HX-Request"))
    return templates.TemplateResponse(
        request,
        "partials/term_state.html" if fragment else "state.html",
        {"src": src, "state": state, "fragment": fragment},
    )


def _fragment_query_string(params: service.SearchParams) -> str:
    """The current search restated as a query string, minus the cursor —
    the load-more sentinel appends its own ``after``. ``doseq`` so the
    repeatable ``has`` param survives as ``has=…&has=…``."""
    fields = params.model_dump(exclude_defaults=True, exclude={"after"})
    enc = lambda v: str(v).lower() if isinstance(v, bool) else v  # noqa: E731
    return urlencode(
        {
            k: [enc(x) for x in v] if isinstance(v, list) else enc(v)
            for k, v in fields.items()
        },
        doseq=True,
    )


def _form_clause(quantifier: str, tag: str | None, match: str, q: str | None) -> str:
    """The form's single predicate, spelled as a ``has`` clause."""
    prefix = "now:" if quantifier == "now" else ""
    value = q or ""
    if match == "exact":
        return f"{prefix}{tag or ''}={value}"
    if match == "regex":
        return f"{prefix}{tag or ''}~/{value}/"
    return f"{prefix}{tag or ''}~{value}"


class _FormParams(service.SearchParams):
    """The search form's dialect of :class:`service.SearchParams`.

    ``order`` additionally accepts the form's combined values: a select
    can only set one param, so ``newest``/``oldest`` mean date order
    with the commit-time direction baked in (date order is newest-first
    by default, ``reverse`` makes it oldest-first). The form defaults to
    newest-first — the git-log shape — where the API defaults to term
    order.

    ``scope`` and ``quantifier`` are the term-histories controls: in
    ``scope=terms`` the q/match/tag fields *are* the term-set predicate,
    translated into a ``has`` clause (extra hand-written ``has=`` params
    AND-compose with it). The blank terms-scope form translates to the
    match-anything clause ``~`` — browse every term's full history,
    paged — mirroring the changes-scope blank-form browse.
    """

    order: Literal["term", "date", "newest", "oldest"] = "newest"
    scope: Literal["changes", "terms"] = "changes"
    quantifier: Literal["ever", "now"] = "ever"

    def to_search_params(self) -> service.SearchParams:
        data = self.model_dump(exclude={"scope", "quantifier"})
        if self.scope == "terms" or self.has:
            if self.scope == "terms":
                data["has"] = [
                    _form_clause(self.quantifier, self.tag, self.match, self.q),
                    *data["has"],
                ]
            data["q"] = None
            data["tag"] = None
            data["order"], data["reverse"] = "term", False
        elif data["order"] == "newest":
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
        "form": None,
        "page": None,
        "facets": request.app.state.registry.facets(src),
        "commit_noun": _commit_noun(style),
    }
    # A bare URL shows the quiet form; any submitted params — even all
    # blank — run the search (a blank form browses everything, paged).
    if params is not None and request.url.query:
        sp = params.to_search_params()
        # The form re-renders from the raw dialect (`form`): translation
        # nulls q/tag in terms scope, so `sp` can't refill the inputs.
        context["form"] = params
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
    # Older/newer chevrons: seqs are dense from 0 through HEAD's.
    seq, last = view.commit.commit_seq, db.resolve_ref("HEAD")
    return templates.TemplateResponse(
        request,
        "commit.html",
        {
            "src": src,
            "view": view,
            "ref": sha,
            "namespace": namespace,
            "full": full,
            "prev_url": f"/{src}/commits/{seq - 1}" if seq > 0 else None,
            "next_url": f"/{src}/commits/{seq + 1}" if seq < last else None,
        },
    )


@router.get("/{src}/commits/{sha}/terms", response_class=HTMLResponse)
def commit_terms(
    request: Request,
    handle: Handle,
    src: str,
    sha: str,
    after: str,
    namespace: str | None = None,
    full: bool = False,
):
    """The commit page's load-more fragment: the next window of term
    sections plus a fresh sentinel while more remain."""
    db, style = handle
    view = service.get_commit(
        db, style, sha, namespace=namespace, full=full, after=after
    )
    if view is None:
        raise HTTPException(
            status_code=404, detail=f"no indexed changes for commit {sha!r}"
        )
    return templates.TemplateResponse(
        request,
        "partials/commit_terms.html",
        {
            "src": src,
            "view": view,
            "ref": sha,
            "namespace": namespace,
            "full": full,
        },
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
