"""Context PR engine — create, review, merge context pull requests.

A Context PR is a proposed change to witch.yaml that goes through:
1. Branch creation
2. Semantic diff generation
3. Eval gate validation
4. Review & approval
5. Merge with version bump
6. Canary deploy orchestration (future)
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from ctxwitch.core.context import ContextDiff, ContextDiffEntry


class PRStatus(Enum):
    OPEN = "open"
    EVAL_RUNNING = "eval_running"
    EVAL_PASSED = "eval_passed"
    EVAL_FAILED = "eval_failed"
    REVIEW_PENDING = "review_pending"
    APPROVED = "approved"
    MERGED = "merged"
    CLOSED = "closed"


@dataclass
class PRComment:
    author: str
    body: str
    timestamp: str
    path: Optional[str] = None  # context path the comment refers to


@dataclass
class ContextPR:
    id: str
    number: int
    title: str
    author: str
    branch: str
    base: str
    status: PRStatus
    created_at: str
    diff: Optional[ContextDiff] = None
    eval_result: Optional[Dict[str, Any]] = None
    comments: List[PRComment] = field(default_factory=list)
    reviewers: List[str] = field(default_factory=list)
    approvals: List[str] = field(default_factory=list)
    merged_at: Optional[str] = None
    merged_by: Optional[str] = None


class PRStore:
    """Local file-based PR store (.ctxwitch/prs/)."""

    def __init__(self, root: Path):
        self.root = root
        self.pr_dir = root / ".ctxwitch" / "prs"
        self.pr_dir.mkdir(parents=True, exist_ok=True)

    def _next_number(self) -> int:
        existing = list(self.pr_dir.glob("*.json"))
        if not existing:
            return 1
        numbers = []
        for f in existing:
            try:
                with open(f) as fh:
                    data = json.load(fh)
                    numbers.append(data.get("number", 0))
            except (json.JSONDecodeError, KeyError):
                continue
        return max(numbers, default=0) + 1

    def create(
        self,
        title: str,
        author: str,
        branch: str,
        base: str = "main",
        diff: Optional[ContextDiff] = None,
    ) -> ContextPR:
        pr_id = uuid.uuid4().hex[:8]
        number = self._next_number()

        pr = ContextPR(
            id=pr_id,
            number=number,
            title=title,
            author=author,
            branch=branch,
            base=base,
            status=PRStatus.OPEN,
            created_at=datetime.now(timezone.utc).isoformat(),
            diff=diff,
        )

        self._save(pr)
        return pr

    def get(self, number: int) -> Optional[ContextPR]:
        for f in self.pr_dir.glob("*.json"):
            with open(f) as fh:
                data = json.load(fh)
                if data.get("number") == number:
                    return self._from_dict(data)
        return None

    def list_prs(self, status: Optional[PRStatus] = None) -> List[ContextPR]:
        prs = []
        for f in sorted(self.pr_dir.glob("*.json")):
            with open(f) as fh:
                data = json.load(fh)
                pr = self._from_dict(data)
                if status is None or pr.status == status:
                    prs.append(pr)
        return prs

    def update_status(self, number: int, status: PRStatus) -> Optional[ContextPR]:
        pr = self.get(number)
        if pr:
            pr.status = status
            if status == PRStatus.MERGED:
                pr.merged_at = datetime.now(timezone.utc).isoformat()
            self._save(pr)
        return pr

    def add_comment(self, number: int, comment: PRComment) -> Optional[ContextPR]:
        pr = self.get(number)
        if pr:
            pr.comments.append(comment)
            self._save(pr)
        return pr

    def approve(self, number: int, reviewer: str) -> Optional[ContextPR]:
        pr = self.get(number)
        if pr:
            if reviewer not in pr.approvals:
                pr.approvals.append(reviewer)
            pr.status = PRStatus.APPROVED
            self._save(pr)
        return pr

    def _save(self, pr: ContextPR) -> None:
        data = {
            "id": pr.id,
            "number": pr.number,
            "title": pr.title,
            "author": pr.author,
            "branch": pr.branch,
            "base": pr.base,
            "status": pr.status.value,
            "created_at": pr.created_at,
            "comments": [
                {
                    "author": c.author,
                    "body": c.body,
                    "timestamp": c.timestamp,
                    "path": c.path,
                }
                for c in pr.comments
            ],
            "reviewers": pr.reviewers,
            "approvals": pr.approvals,
            "merged_at": pr.merged_at,
            "merged_by": pr.merged_by,
        }
        if pr.diff:
            data["diff"] = {
                "old_version": pr.diff.old_version,
                "new_version": pr.diff.new_version,
                "entries": [
                    {
                        "path": e.path,
                        "old_value": e.old_value,
                        "new_value": e.new_value,
                        "change_type": e.change_type,
                    }
                    for e in pr.diff.entries
                ],
            }
        if pr.eval_result:
            data["eval_result"] = pr.eval_result

        filepath = self.pr_dir / f"{pr.id}.json"
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _from_dict(self, data: Dict[str, Any]) -> ContextPR:
        diff = None
        if "diff" in data and data["diff"]:
            d = data["diff"]
            entries = [
                ContextDiffEntry(
                    path=e["path"],
                    old_value=e["old_value"],
                    new_value=e["new_value"],
                    change_type=e["change_type"],
                )
                for e in d.get("entries", [])
            ]
            diff = ContextDiff(
                old_version=d["old_version"],
                new_version=d["new_version"],
                old_sha=d.get("old_sha", ""),
                new_sha=d.get("new_sha", ""),
                entries=entries,
            )

        comments = [
            PRComment(
                author=c["author"],
                body=c["body"],
                timestamp=c["timestamp"],
                path=c.get("path"),
            )
            for c in data.get("comments", [])
        ]

        return ContextPR(
            id=data["id"],
            number=data["number"],
            title=data["title"],
            author=data["author"],
            branch=data["branch"],
            base=data["base"],
            status=PRStatus(data["status"]),
            created_at=data["created_at"],
            diff=diff,
            eval_result=data.get("eval_result"),
            comments=comments,
            reviewers=data.get("reviewers", []),
            approvals=data.get("approvals", []),
            merged_at=data.get("merged_at"),
            merged_by=data.get("merged_by"),
        )
