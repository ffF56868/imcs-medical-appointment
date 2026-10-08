"""Shared access rules for conversations and knowledge records.

The first version of the project only had feature permissions (admin versus
regular user). These helpers add data permissions without changing the
meaning of the existing public knowledge base.
"""

from sqlmodel import select

from app.legacy.models import KnowledgeDocument, User


PUBLIC_KNOWLEDGE_BASE_ID = "global"
PUBLIC_VISIBILITY = "public"


def can_access_document(document: KnowledgeDocument, user: User) -> bool:
    if user.is_admin or document.visibility == PUBLIC_VISIBILITY:
        return True
    return document.owner_user_id == user.id


def accessible_documents_statement(user: User):
    statement = select(KnowledgeDocument)
    if user.is_admin:
        return statement
    return statement.where(
        (KnowledgeDocument.visibility == PUBLIC_VISIBILITY)
        | (KnowledgeDocument.owner_user_id == user.id)
    )


def build_vector_access_filter(user: User) -> dict | None:
    """Build the vector filter for a regular user's public/private scope."""
    if user.is_admin:
        return None
    return {
        "$or": [
            {"visibility": PUBLIC_VISIBILITY},
            {
                "$and": [
                    {"visibility": "private"},
                    {"owner_user_id": user.id},
                ]
            },
        ]
    }
