from locust import HttpUser, constant, task

class StableDiffusionUser(HttpUser):
    wait_time = constant(1)

    @task
    def predict(self):
        params = {
            "prompt": "photograph of an astronaut riding a horse",
            "batch_size": 1
        }

        _ = self.client.post(
            "/image/generate", params=params
        )
