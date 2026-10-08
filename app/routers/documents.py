from datetime import UTC, datetime

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlmodel import Session, select

from app.access import accessible_documents_statement, can_access_document
from app.cache import invalidate_knowledge_status_cache
from app.database import get_session
from app.document_parsers import (
    MAX_BATCH_BYTES,
    ParsedDocument,
    parse_uploaded_file,
    parse_web_page,
)
from app.legacy.models import KnowledgeDocument, User
from app.routers.auth import get_current_user, require_admin
from app.legacy.schemas import (
    DocumentUploadBatchResponse,
    DocumentUploadError,
    DocumentUploadItem,
    DocumentWebImportRequest,
    KnowledgeDocumentCreate,
    KnowledgeDocumentRead,
)


router = APIRouter(
    prefix="/documents",
    tags=["documents"],
    dependencies=[Depends(get_current_user)],
)

MAX_BATCH_FILES = 20


async def parse_upload(upload: UploadFile) -> tuple[str, list[ParsedDocument], int]:
    filename = (upload.filename or "").strip()
    if not filename:
        raise ValueError("文件名不能为空")
    raw_content = await upload.read()
    return filename, parse_uploaded_file(filename, raw_content), len(raw_content)


def build_uploaded_document(parsed: ParsedDocument) -> KnowledgeDocument:
    return KnowledgeDocument(
        title=parsed.title,
        content=parsed.content,
        source=parsed.source,
        source_url=parsed.source_url,
        source_tier="unverified",
        page_number=parsed.page_number,
        updated_at=datetime.now(UTC),
    )


def ensure_unique_document_titles(
    parsed_documents: list[ParsedDocument],
    session: Session,
    seen_titles: set[str],
) -> None:
    for parsed in parsed_documents:
        if parsed.title in seen_titles:
            raise ValueError("本次上传中存在同名资料标题")
        duplicate = session.exec(
            select(KnowledgeDocument).where(KnowledgeDocument.title == parsed.title)
        ).first()
        if duplicate is not None:
            raise ValueError(f"资料标题“{parsed.title}”已经存在")
    seen_titles.update(parsed.title for parsed in parsed_documents)


@router.post("", response_model=KnowledgeDocumentRead, status_code=201)
def create_document(
    document_data: KnowledgeDocumentCreate,
    current_user: User = Depends(require_admin),
    session: Session = Depends(get_session),
):
    duplicate = session.exec(
        select(KnowledgeDocument).where(
            KnowledgeDocument.title == document_data.title
        )
    ).first()

    if duplicate is not None:
        raise HTTPException(status_code=409, detail="知识文档标题已经存在")

    document = KnowledgeDocument.model_validate(document_data)
    if document.visibility == "private":
        document.owner_user_id = current_user.id
    else:
        document.owner_user_id = None
    document.updated_at = datetime.now(UTC)
    session.add(document)
    session.commit()
    invalidate_knowledge_status_cache()
    session.refresh(document)
    return document


@router.post("/upload", response_model=KnowledgeDocumentRead, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    _admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
):
    try:
        _filename, parsed_documents, _size = await parse_upload(file)
        ensure_unique_document_titles(parsed_documents, session, set())
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    documents = [build_uploaded_document(parsed) for parsed in parsed_documents]
    session.add_all(documents)
    session.commit()
    invalidate_knowledge_status_cache()
    for document in documents:
        session.refresh(document)
    return documents[0]


@router.post(
    "/upload-batch",
    response_model=DocumentUploadBatchResponse,
    status_code=201,
)
async def upload_documents(
    files: list[UploadFile] = File(...),
    _admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
):
    """Import several text files and report per-file validation failures."""
    if not files:
        raise HTTPException(status_code=400, detail="至少选择一个文件")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"一次最多上传 {MAX_BATCH_FILES} 个文件",
        )

    created_files: list[tuple[str, list[KnowledgeDocument]]] = []
    errors: list[DocumentUploadError] = []
    total_bytes = 0
    seen_titles: set[str] = set()

    for upload in files:
        filename = (upload.filename or "未命名文件").strip() or "未命名文件"
        try:
            filename, parsed_documents, raw_size = await parse_upload(upload)
            if total_bytes + raw_size > MAX_BATCH_BYTES:
                raise ValueError("批量上传文件总大小不能超过 20MB")
            total_bytes += raw_size
            ensure_unique_document_titles(parsed_documents, session, seen_titles)
            documents = [build_uploaded_document(parsed) for parsed in parsed_documents]
            session.add_all(documents)
            created_files.append((filename, documents))
        except ValueError as error:
            errors.append(DocumentUploadError(filename=filename, detail=str(error)))

    session.commit()
    invalidate_knowledge_status_cache()
    items: list[DocumentUploadItem] = []
    for filename, documents in created_files:
        for document in documents:
            session.refresh(document)
        items.append(
            DocumentUploadItem(
                filename=filename,
                title=documents[0].title,
                document_id=documents[0].id,
                created_document_count=len(documents),
            )
        )

    return DocumentUploadBatchResponse(
        created_count=len(items),
        created_document_count=sum(len(documents) for _, documents in created_files),
        failed_count=len(errors),
        items=items,
        errors=errors,
    )


@router.get("", response_model=list[KnowledgeDocumentRead])
def list_documents(
    keyword: str | None = Query(default=None, max_length=100),
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    statement = accessible_documents_statement(current_user).order_by(
        KnowledgeDocument.id
    )

    if keyword and keyword.strip():
        statement = statement.where(
            KnowledgeDocument.title.contains(keyword.strip())
        )

    return session.exec(statement).all()


@router.post("/import-web", response_model=KnowledgeDocumentRead, status_code=201)
def import_web_document(
    import_data: DocumentWebImportRequest,
    _admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
):
    try:
        parsed = parse_web_page(import_data.url, import_data.title)
        ensure_unique_document_titles([parsed], session, set())
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    document = build_uploaded_document(parsed)
    session.add(document)
    session.commit()
    invalidate_knowledge_status_cache()
    session.refresh(document)
    return document


@router.get("/{document_id}", response_model=KnowledgeDocumentRead)
def get_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    document = session.get(KnowledgeDocument, document_id)

    if document is None or not can_access_document(document, current_user):
        raise HTTPException(status_code=404, detail="知识文档不存在")

    return document


@router.put("/{document_id}", response_model=KnowledgeDocumentRead)
def update_document(
    document_id: int,
    document_data: KnowledgeDocumentCreate,
    current_user: User = Depends(require_admin),
    session: Session = Depends(get_session),
):
    document = session.get(KnowledgeDocument, document_id)

    if document is None:
        raise HTTPException(status_code=404, detail="知识文档不存在")

    duplicate = session.exec(
        select(KnowledgeDocument).where(
            KnowledgeDocument.title == document_data.title,
            KnowledgeDocument.id != document_id,
        )
    ).first()

    if duplicate is not None:
        raise HTTPException(status_code=409, detail="知识文档标题已经存在")

    document.title = document_data.title
    document.content = document_data.content
    document.source = document_data.source
    document.source_url = document_data.source_url
    document.source_tier = document_data.source_tier
    document.knowledge_base_id = document_data.knowledge_base_id
    document.visibility = document_data.visibility
    document.page_number = document_data.page_number
    document.owner_user_id = (
        current_user.id if document_data.visibility == "private" else None
    )
    document.updated_at = datetime.now(UTC)
    session.add(document)
    session.commit()
    invalidate_knowledge_status_cache()
    session.refresh(document)
    return document


@router.delete("/{document_id}", status_code=204)
def delete_document(
    document_id: int,
    _admin: User = Depends(require_admin),
    session: Session = Depends(get_session),
):
    document = session.get(KnowledgeDocument, document_id)

    if document is None:
        raise HTTPException(status_code=404, detail="知识文档不存在")

    session.delete(document)
    session.commit()
    invalidate_knowledge_status_cache()
