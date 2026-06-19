import pytest
import os
import app as app_module

@pytest.fixture()
def temp_db(tmp_path):
    dbf = tmp_path / "test_ppe.db"
    app_module.DB_PATH = str(dbf)
    app_module.init_db()
    return str(dbf)
