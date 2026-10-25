"""Needs: pip install locust.   locust -f scripts/loadtest.py --host http://localhost:8000
Set TOKEN in the environment first. Expect HTTP 429 once /predict exceeds 60/min."""

import os

from locust import HttpUser, between, task

FLOW = {"src_ip": "10.0.0.5", "dst_ip": "10.0.0.9", "src_port": 4444, "dst_port": 80}


class Api(HttpUser):
    wait_time = between(0.1, 0.3)

    @task
    def predict(self):
        with self.client.post(
            "/api/v1/predict",
            json=FLOW,
            catch_response=True,
            headers={"Authorization": f"Bearer {os.environ['TOKEN']}"},
        ) as r:
            if r.status_code in (200, 429):
                r.success()
