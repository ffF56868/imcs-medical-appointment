from sqlalchemy import inspect

from app.database import build_engine, create_db_and_tables
from app.legacy.models import Condition


def test_sqlite_engine_uses_sqlite_compatible_connection_options():
    test_engine = build_engine("sqlite://")
    create_db_and_tables(test_engine)

    assert test_engine.url.get_backend_name() == "sqlite"
    assert "condition" in inspect(test_engine).get_table_names()
    assert inspect(test_engine).get_columns("condition")


def test_models_remain_available_for_schema_creation():
    assert Condition.__tablename__ == "condition"
