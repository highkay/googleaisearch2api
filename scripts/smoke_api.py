from __future__ import annotations

import argparse
import json

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a smoke test against a local googleaisearch2api server."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--token", default="change-me-google-search")
    args = parser.parse_args()

    headers = {
        "Authorization": f"Bearer {args.token}",
        "Content-Type": "application/json",
    }

    with httpx.Client(base_url=args.base_url, timeout=120.0) as client:
        health = client.get("/healthz")
        models = client.get("/v1/models", headers=headers)
        completion = client.post(
            "/v1/chat/completions",
            headers=headers,
            json={
                "model": "google-search",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "What is the difference between Responses API and "
                            "Chat Completions API? summarize in 3 points"
                        ),
                    }
                ],
            },
        )

    print(
        json.dumps(
            {
                "health": health.json(),
                "models": models.json(),
                "completion": completion.json(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
