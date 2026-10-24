#!/usr/bin/env bash
# Build every stale or missing artifact (heavy ones included, in dependency order), render the book,
# then preview it. The work runs in a screen session; detach with Ctrl-a d and this script keeps
# waiting, starting the preview when everything succeeded.
session=render
status=artifacts/render.status

if screen -list | grep -q "\.$session[[:space:]]"; then
    echo "Render already in progress (screen session '$session'); reattach with: screen -r $session" >&2
    exit 1
fi

mkdir -p artifacts
rm -f "$status"

# The render runs without RUN_HEAVY: after the build step nothing should be stale, and if something
# is, it should fail loudly rather than start a hidden training run.
screen -S "$session" bash -c \
    "uv run python -m convfinqa.artifacts build --all && quarto render docs/ --no-execute-daemon --log artifacts/render.log; echo \$? > $status"

# Wait for the exit status of build + render. Give up if the session ended without writing one (e.g. killed).
while [ ! -f "$status" ]; do
    if ! screen -list | grep -q "\.$session[[:space:]]"; then
        [ -f "$status" ] && break
        echo "Render session ended without a status; see artifacts/render.log" >&2
        exit 1
    fi
    sleep 5
done

if [ "$(cat "$status")" = 0 ]; then
    # Serve the site, re-rendering a chapter when its file changes. ALLOW_STALE lets prose edits keep
    # working after a code change made an artifact stale (a warning box marks the old result).
    # --no-execute-daemon: Quarto otherwise keeps each chapter's Jupyter kernel alive for 5 minutes, and a
    # reused kernel keeps the environment it started with, so RUN_HEAVY/ALLOW_STALE changes would be ignored.
    ALLOW_STALE=1 quarto preview docs/ --no-execute-daemon
else
    echo "Build or render failed (exit code $(cat "$status")); see artifacts/render.log and artifacts/<name>.tmp/build.log" >&2
    exit 1
fi
