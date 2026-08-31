"""The JSON API: thin FastAPI routes over :mod:`obohog.service`.

Every handler resolves its source through the registry dependency, calls
one service function, and either returns its Pydantic model or raises
404 when the service says "no such thing". Error mapping for the typed
query exceptions lives in :mod:`.app`.
"""

from typing import Annotated, Iterator

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from .. import service
from ..query import HistoryDB
from ..views import SourceStyle

router = APIRouter()


def _source_handle(
    src: str, request: Request
) -> Iterator[tuple[HistoryDB, SourceStyle]]:
    """Yield a forked per-request handle; close it after the response."""
    db, style = request.app.state.registry.acquire(src)
    try:
        yield db, style
    finally:
        db.close()


Handle = Annotated[tuple[HistoryDB, SourceStyle], Depends(_source_handle)]


def _or_404(result, detail: str):
    if result is None:
        raise HTTPException(status_code=404, detail=detail)
    return result


@router.get("/sources")
def sources(request: Request) -> list[service.SourceInfo]:
    """All configured sources with build status."""
    return service.list_sources(request.app.state.config)


@router.get("/sources/{src}/releases")
def releases(handle: Handle) -> list[service.ReleaseOut]:
    db, _ = handle
    return service.list_releases(db)


@router.get("/sources/{src}/facets")
def facets(src: str, request: Request) -> service.FacetsOut:
    """Distinct predicate/namespace values in the source, for filter UIs."""
    return request.app.state.registry.facets(src)


@router.get("/sources/{src}/terms/{term_id}")
def term_timeline(
    handle: Handle,
    term_id: str,
    predicate: str | None = None,
    since: str | None = None,
    limit: int | None = Query(None, ge=1),
    full: bool = False,
) -> service.TimelineOut:
    """A term's change history, commit-grouped, oldest first."""
    db, style = handle
    return _or_404(
        service.get_timeline(
            db, style, term_id,
            predicate=predicate, since=since, limit=limit, full=full,
        ),
        f"no history for {term_id!r}",
    )


@router.get("/sources/{src}/terms/{term_id}/state")
def term_state(handle: Handle, term_id: str, at: str) -> service.StateOut:
    """The term's reconstructed clause set as of a ref."""
    db, _ = handle
    return _or_404(
        service.get_state(db, term_id, at),
        f"{term_id!r} has no snapshot at or before {at!r}",
    )


@router.get("/sources/{src}/search")
def search(
    handle: Handle, params: Annotated[service.SearchParams, Query()]
) -> service.PageOut[service.TermSectionOut] | service.PageOut[service.CommitSectionOut]:
    """One page of events matching a query; `next_cursor` resumes."""
    db, style = handle
    return service.search(db, style, params)


@router.get("/sources/{src}/diff")
def diff(
    handle: Handle,
    a: str,
    b: str,
    term: str | None = None,
    namespace: str | None = None,
    limit: int = Query(service.DEFAULT_PAGE, ge=1, le=service.MAX_PAGE),
    after: str | None = None,
    full: bool = False,
) -> service.PageOut[service.TermSectionOut]:
    """One page of changes between two refs, grouped by term."""
    db, style = handle
    return service.diff(
        db, style, a, b,
        term=term, namespace=namespace, limit=limit, after=after, full=full,
    )


@router.get("/sources/{src}/commits/{sha}")
def commit(
    handle: Handle, sha: str, namespace: str | None = None, full: bool = False
) -> service.CommitViewOut:
    """One commit's changes (sha prefix ok), grouped by term."""
    db, style = handle
    return _or_404(
        service.get_commit(db, style, sha, namespace=namespace, full=full),
        f"no indexed changes for commit {sha!r}",
    )


@router.get("/sources/{src}/prs/{number}")
def pr(handle: Handle, number: int) -> service.PrOut:
    """Terms changed by a pull request's commits."""
    db, style = handle
    return _or_404(
        service.get_pr(db, style, number),
        f"no indexed changes for PR #{number}",
    )
