from app.security import MAX_ACTIVE_SESSIONS, MAX_LOGIN_FAILURES


def test_register_phone_normalizes_to_china_country_code(auth_client):
    response = auth_client.post(
        "/auth/register",
        json={
            "account": "18242643780",
            "password": "health123",
            "confirm_password": "health123",
        },
    )

    assert response.status_code == 201
    data = response.json()
    assert data["user"]["account"] == "+8618242643780"
    assert data["token_type"] == "bearer"
    assert data["access_token"]


def test_register_email_is_case_insensitive(auth_client):
    response = auth_client.post(
        "/auth/register",
        json={
            "account": "Health.Example@Example.COM",
            "password": "health123",
            "confirm_password": "health123",
        },
    )

    assert response.status_code == 201
    assert response.json()["user"]["account"] == "health.example@example.com"


def test_repeated_login_failures_are_temporarily_locked(auth_client):
    payload = {
        "account": "locked@example.com",
        "password": "health123",
        "confirm_password": "health123",
    }
    assert auth_client.post("/auth/register", json=payload).status_code == 201

    for _ in range(MAX_LOGIN_FAILURES - 1):
        response = auth_client.post(
            "/auth/login",
            json={"account": payload["account"], "password": "wrong123"},
        )
        assert response.status_code == 401

    limited = auth_client.post(
        "/auth/login",
        json={"account": payload["account"], "password": "wrong123"},
    )
    assert limited.status_code == 429

    still_limited = auth_client.post(
        "/auth/login",
        json={"account": payload["account"], "password": payload["password"]},
    )
    assert still_limited.status_code == 429


def test_oldest_sessions_are_revoked_after_session_limit(auth_client):
    payload = {
        "account": "sessions@example.com",
        "password": "health123",
        "confirm_password": "health123",
    }
    first = auth_client.post("/auth/register", json=payload)
    first_token = first.json()["access_token"]

    for _ in range(MAX_ACTIVE_SESSIONS):
        response = auth_client.post(
            "/auth/login",
            json={"account": payload["account"], "password": payload["password"]},
        )
        assert response.status_code == 200

    response = auth_client.get(
        "/auth/me", headers={"Authorization": f"Bearer {first_token}"}
    )
    assert response.status_code == 401


def test_register_rejects_password_without_letters_or_digits(auth_client):
    no_letter = auth_client.post(
        "/auth/register",
        json={
            "account": "18242643780",
            "password": "12345678",
            "confirm_password": "12345678",
        },
    )
    no_digit = auth_client.post(
        "/auth/register",
        json={
            "account": "test@example.com",
            "password": "abcdefgh",
            "confirm_password": "abcdefgh",
        },
    )

    assert no_letter.status_code == 422
    assert "英文字母" in no_letter.text
    assert no_digit.status_code == 422
    assert "数字" in no_digit.text


def test_register_rejects_mismatched_passwords(auth_client):
    response = auth_client.post(
        "/auth/register",
        json={
            "account": "test@example.com",
            "password": "health123",
            "confirm_password": "health456",
        },
    )

    assert response.status_code == 422
    assert "两次输入的密码不一致" in response.text


def test_duplicate_account_is_rejected(auth_client):
    payload = {
        "account": "18242643780",
        "password": "health123",
        "confirm_password": "health123",
    }
    assert auth_client.post("/auth/register", json=payload).status_code == 201

    duplicate = auth_client.post("/auth/register", json=payload)

    assert duplicate.status_code == 409
    assert duplicate.json()["detail"] == "该账号已经注册"


def test_login_me_and_logout_manage_one_session(auth_client):
    register = auth_client.post(
        "/auth/register",
        json={
            "account": "test@example.com",
            "password": "health123",
            "confirm_password": "health123",
        },
    )
    token = register.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    me = auth_client.get("/auth/me", headers=headers)
    assert me.status_code == 200
    assert me.json()["account"] == "test@example.com"

    login = auth_client.post(
        "/auth/login",
        json={"account": "TEST@EXAMPLE.COM", "password": "health123"},
    )
    assert login.status_code == 200
    assert login.json()["user"]["account"] == "test@example.com"

    logout = auth_client.post("/auth/logout", headers=headers)
    assert logout.status_code == 204
    assert auth_client.get("/auth/me", headers=headers).status_code == 401


def test_auth_does_not_require_a_captcha_field(auth_client):
    response = auth_client.post(
        "/auth/register",
        json={
            "account": "test@example.com",
            "password": "health123",
            "confirm_password": "health123",
        },
    )

    assert response.status_code == 201
