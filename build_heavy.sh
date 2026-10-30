#!/usr/bin/env bash
# Build every stale or missing artifact, heavy ones included (RUN_HEAVY=1), and render the book, in a screen
# session that stays open afterwards: detach with Ctrl-a d, and reattach with: screen -r heavy
#
# The `build --all` step builds the artifacts that chapters name with a literal get("..."). Artifacts named
# by a formula (untuned_{split}_{variant}_..., one per run) are built by the render itself, when a chapter
# asks for one that is stale or missing, which is why the render also runs with RUN_HEAVY=1. Unlike build.sh
# this does not wait for the session or start a preview.
session=heavy

if screen -list | grep -q "\.$session[[:space:]]"; then
    echo "Session '$session' already exists; reattach with: screen -r $session" >&2
    exit 1
fi

mkdir -p artifacts

screen -S "$session" bash -c '
    export RUN_HEAVY=1
    uv run python -m convfinqa.artifacts build --all \
        && quarto render docs/ --no-execute-daemon --log artifacts/render.log
    status=$?
    echo
    if [ $status -eq 0 ]; then
        echo "Build and render finished."
    else
        echo "Build or render failed; see artifacts/render.log and artifacts/<name>.tmp/build.log" >&2
    fi
    echo "This session stays open: detach with Ctrl-a d, or type exit to close it."
    exec bash
'
