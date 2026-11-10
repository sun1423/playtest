import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
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
        with TestClient(app.app) as guest:
            response = guest.post("/api/vace/runner/next")
        self.assertEqual(response.status_code, 401)

    def test_authenticated_runner_can_poll_empty_queue(self):
        response = self.client.post(
            "/api/vace/runner/next", headers=self.runner_headers
        )
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["job"])

    def test_job_can_be_claimed_and_completed_with_voice_audio(self):
        voice_path = app.VOICES / "ci_voice.wav"
        voice_path.write_bytes(b"RIFF-test-wav-reference")
        fields = {
            "prompt": "Transform the scene into a soft watercolor animation",
            "narration": "नमस्ते, यह एक वीडियो परीक्षण है।",
            "language": "hi",
            "voice_id": "ci_voice",
            "motion_mode": "flow",
            "keep_source_audio": "false",
        }
        with patch("app.subprocess.run", return_value=SimpleNamespace(stdout="2.0\n")):
            queued = self.client.post(
                "/api/vace/video",
                data=fields,
                files={"file": ("source.mp4", b"source-video-test", "video/mp4")},
            )
        self.assertEqual(queued.status_code, 200, queued.text)
        job_id = queued.json()["id"]

        claim = self.client.post(
            "/api/vace/runner/next", headers=self.runner_headers
        )
        self.assertEqual(claim.status_code, 200, claim.text)
        job = claim.json()["job"]
        self.assertEqual(job["id"], job_id)
        self.assertEqual(job["voice_id"], "ci_voice")
        self.assertEqual(
            self.client.get(job["input_url"], headers=self.runner_headers).status_code,
            200,
        )
        self.assertEqual(
            self.client.get(job["voice_url"], headers=self.runner_headers).status_code,
            200,
        )

        progress = self.client.post(
            f"/api/vace/runner/jobs/{job_id}/progress",
            headers=self.runner_headers,
            json={"stage": "Testing output upload", "progress": 90},
        )
        self.assertEqual(progress.status_code, 200, progress.text)
        with patch(
            "app.subprocess.run",
            return_value=SimpleNamespace(stdout="video\naudio\n"),
        ):
            completed = self.client.post(
                f"/api/vace/runner/jobs/{job_id}/complete",
                headers=self.runner_headers,
                files={"file": ("final.mp4", b"test-mp4-output", "video/mp4")},
            )
        self.assertEqual(completed.status_code, 200, completed.text)
        item = next(row for row in self.client.get("/api/jobs").json() if row["id"] == job_id)
        self.assertEqual(item["status"], "completed")
        self.assertEqual(item["progress"], 100)
        self.assertEqual(
            self.client.get(f"/api/jobs/{job_id}/output").content,
            b"test-mp4-output",
        )

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
