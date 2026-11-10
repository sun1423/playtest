import os
import tempfile
import unittest
from pathlib import Path

_ROOT = tempfile.TemporaryDirectory(prefix="studio-vace-tests-")
os.environ["STUDIO_ROOT"] = str(Path(_ROOT.name) / "data")
os.environ["ADMIN_PASSWORD_HASH"] = "test-salt$test-hash"
os.environ["SESSION_SECRET"] = "vace-test-session-secret"
os.environ["VACE_RUNNER_TOKEN"] = "vace-test-runner-secret"

from fastapi.testclient import TestClient
import app


class VaceRunnerApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app.app)
        cls.client.__enter__()
        cls.client.cookies.set("studio_session", app.make_session(), path="/api")
        cls.runner_headers = {
            "Authorization": "Bearer " + os.environ["VACE_RUNNER_TOKEN"]
        }

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        _ROOT.cleanup()

    def test_status_reports_runner_configuration(self):
        response = self.client.get("/api/vace/status")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["runner_configured"])
        self.assertEqual(response.json()["video_model"], "Wan2.1-VACE-1.3B")

    def test_runner_endpoints_reject_missing_token(self):
        unauthenticated = TestClient(app.app)
        response = unauthenticated.post("/api/vace/runner/next")
        self.assertEqual(response.status_code, 401)

    def test_authenticated_runner_can_poll_empty_queue(self):
        response = self.client.post(
            "/api/vace/runner/next", headers=self.runner_headers
        )
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["job"])

    def test_all_remote_runner_routes_are_registered(self):
        paths = app.app.openapi()["paths"]
        expected = {
            "/api/vace/status",
            "/api/vace/video",
            "/api/vace/runner/next",
            "/api/vace/runner/jobs/{job_id}/input",
            "/api/vace/runner/jobs/{job_id}/voice",
            "/api/vace/runner/jobs/{job_id}/progress",
            "/api/vace/runner/jobs/{job_id}/complete",
            "/api/vace/runner/jobs/{job_id}/fail",
        }
        self.assertTrue(expected.issubset(paths))


if __name__ == "__main__":
    unittest.main()
