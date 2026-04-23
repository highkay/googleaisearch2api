from __future__ import annotations

import argparse
import json

from googleaisearch2api.browser import GoogleAiRunner
from googleaisearch2api.config import ServiceConfig, get_settings
from googleaisearch2api.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a live Patchright probe against Google AI search."
    )
    parser.add_argument(
        "--prompt",
        default=(
            "What is the difference between Responses API and Chat Completions API? "
            "summarize in 3 points"
        ),
        help="Prompt to submit to Google AI search.",
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.app_log_level)
    config = ServiceConfig.from_settings(settings)
    result = GoogleAiRunner().run_prompt(config, args.prompt)
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
