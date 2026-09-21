"""
Prototype identity handling.

There is no real session/token system yet — the frontend's mock login
(services/mock/authService.ts) derives a user's role from their email
("starts with admin" -> admin, everything else -> relationship_manager)
and sends that identity on every request via X-User-ID / X-User-Email.
This module mirrors that exact rule so both sides agree today.

Swap point for real auth: replace get_current_identity's body with real
session/JWT/OIDC claim extraction. Every route that needs identity or
admin authorization already goes through these two dependencies, so
nothing else changes.
"""

from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException


@dataclass
class Identity:
    user_id: str
    email: str
    is_admin: bool


def get_current_identity(
    x_user_id: str | None = Header(default=None),
    x_user_email: str | None = Header(default=None),
) -> Identity:
    if not x_user_id or not x_user_email:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "code": "UNAUTHENTICATED",
                    "message": "Sign in required.",
                }
            },
        )

    return Identity(
        user_id=x_user_id,
        email=x_user_email,
        is_admin=x_user_email.lower().startswith("admin"),
    )


def require_admin(identity: Identity = Depends(get_current_identity)) -> Identity:
    if not identity.is_admin:
        raise HTTPException(
            status_code=403,
            detail={
                "error": {
                    "code": "FORBIDDEN",
                    "message": "Admin access required.",
                }
            },
        )
    return identity
