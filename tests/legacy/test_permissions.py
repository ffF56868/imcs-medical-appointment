from sqlmodel import Session, select

from app.legacy.models import ChatMessage, KnowledgeDocument, User


def register_user(client, account="user@example.com"):
    response = client.post(
        "/auth/register",
        json={
            "account": account,
            "password": "health123",
            "confirm_password": "health123",
        },
    )
    assert response.status_code == 201
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def test_knowledge_requires_login(auth_client):
    response = auth_client.get("/conditions")

    assert response.status_code == 401


def test_regular_user_can_query_but_cannot_change_knowledge(auth_client):
    headers = register_user(auth_client)

    listed = auth_client.get("/conditions", headers=headers)
    assert listed.status_code == 200

    created = auth_client.post(
        "/conditions",
        headers=headers,
        json={
            "name": "普通用户不应新增的病症",
            "symptoms": "测试症状",
            "treatment": "测试建议",
        },
    )
    assert created.status_code == 403
    assert created.json()["detail"] == "需要管理员权限"


def test_admin_can_change_knowledge(client):
    response = client.post(
        "/conditions",
        json={
            "name": "管理员可以新增的病症",
            "symptoms": "测试症状",
            "treatment": "测试建议",
        },
    )

    assert response.status_code == 201


def test_only_admin_can_read_security_audit_logs(client, auth_client):
    denied = auth_client.get("/auth/audit-logs")
    assert denied.status_code == 401

    allowed = client.get("/auth/audit-logs")
    assert allowed.status_code == 200
    assert "logs" in allowed.json()


def test_regular_users_cannot_read_another_users_conversation(
    auth_client,
    test_engine,
):
    first_headers = register_user(auth_client, "first@example.com")
    second_headers = register_user(auth_client, "second@example.com")

    with Session(test_engine) as session:
        users = session.exec(select(User).order_by(User.id)).all()
        session.add_all(
            [
                ChatMessage(
                    user_id=users[0].id,
                    conversation_id="shared-id",
                    role="user",
                    content="第一位用户的私密问题",
                ),
                ChatMessage(
                    user_id=users[1].id,
                    conversation_id="shared-id",
                    role="user",
                    content="第二位用户的私密问题",
                ),
            ]
        )
        session.commit()

    first_messages = auth_client.get(
        "/conversations/shared-id/messages",
        headers=first_headers,
    )
    second_messages = auth_client.get(
        "/conversations/shared-id/messages",
        headers=second_headers,
    )

    assert first_messages.status_code == 200
    assert second_messages.status_code == 200
    assert [item["content"] for item in first_messages.json()] == [
        "第一位用户的私密问题"
    ]
    assert [item["content"] for item in second_messages.json()] == [
        "第二位用户的私密问题"
    ]


def test_regular_users_cannot_list_another_users_private_document(
    auth_client,
    test_engine,
):
    with Session(test_engine) as session:
        session.add(
            KnowledgeDocument(
                title="仅限其他用户的资料",
                content="私有内容",
                owner_user_id=999,
                visibility="private",
            )
        )
        session.commit()

    headers = register_user(auth_client, "reader@example.com")
    response = auth_client.get("/documents", headers=headers)

    assert response.status_code == 200
    assert all(item["title"] != "仅限其他用户的资料" for item in response.json())
