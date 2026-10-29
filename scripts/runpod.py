"""Single-command entry point for persistent nf-core runs on RunPod."""

import subprocess
import sys

from harmonize.runpod import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "Disconnected locally. A started remote job continues; use status/resume/fetch with its run ID.",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"RunPod launcher: {exc}", file=sys.stderr)
        print(
            "A connection failure does not cancel a remote job. Check status before resuming.",
            file=sys.stderr,
        )
        print("RunPod billing continues until you stop the pod.", file=sys.stderr)
        raise SystemExit(1)
