"""
ofa/push_to_hub.py
------------------
MANUAL push of the trained student from the Modal volume to the HuggingFace Hub.

Deliberately separate from training: weights are never auto-published. Run this
only when you've inspected the student and want it on the Hub. The repo (org or
personal, public or private) is decided here at push time, not baked into config.

Usage (from a machine with the volume pulled, or via a Modal function):
    python -m ofa.push_to_hub \
        --local-dir ./ofa_student/final \
        --repo build-small-hackathon/one-for-all\
        --private
"""

from __future__ import annotations

import argparse


def push(local_dir: str, repo: str, private: bool = True,
         commit_message: str = "One for All student") -> str:
    """Upload a local student folder to the Hub. Returns the repo URL."""
    from huggingface_hub import HfApi, create_repo

    create_repo(repo, repo_type="model", private=private, exist_ok=True)
    api = HfApi()
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo,
        repo_type="model",
        commit_message=commit_message,
    )
    url = f"https://huggingface.co/{repo}"
    print(f"Pushed to {url}")
    return url


def main():
    ap = argparse.ArgumentParser(description="Push OFA student to HF Hub.")
    ap.add_argument("--local-dir", required=True,
                    help="path to the saved student folder (e.g. .../final)")
    ap.add_argument("--repo", required=True,
                    help="target repo id (org/name or user/name)")
    ap.add_argument("--private", action="store_true",
                    help="create a private repo")
    ap.add_argument("--message", default="One for All student")
    args = ap.parse_args()
    push(args.local_dir, args.repo, args.private, args.message)


if __name__ == "__main__":
    main()
