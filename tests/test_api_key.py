import unittest
import os
import tempfile
from fastapi import HTTPException
from app.api_key import (
    init_db,
    generate_api_key,
    verify_key,
    list_api_keys,
    revoke_api_key,
)


class TestApiKeyManagement(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_api_keys.db")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_generate_and_verify_key(self):
        raw_key, prefix = generate_api_key("Test App", db_path=self.db_path)
        self.assertTrue(raw_key.startswith("kt_"))
        self.assertEqual(prefix, raw_key[:7])

        # Verify correct key succeeds
        self.assertTrue(verify_key(raw_key, db_path=self.db_path))

        # Verify invalid key fails
        self.assertFalse(verify_key("kt_invalidkey12345", db_path=self.db_path))

    def test_list_keys(self):
        generate_api_key("App One", db_path=self.db_path)
        generate_api_key("App Two", db_path=self.db_path)

        keys = list_api_keys(db_path=self.db_path)
        self.assertEqual(len(keys), 2)
        names = [k["name"] for k in keys]
        self.assertIn("App One", names)
        self.assertIn("App Two", names)

    def test_revoke_key(self):
        raw_key, prefix = generate_api_key("Revoke Me", db_path=self.db_path)
        self.assertTrue(verify_key(raw_key, db_path=self.db_path))

        # Revoke key using prefix
        success = revoke_api_key(prefix, db_path=self.db_path)
        self.assertTrue(success)

        # Verification should now fail
        self.assertFalse(verify_key(raw_key, db_path=self.db_path))


if __name__ == "__main__":
    unittest.main()
