#!/usr/bin/env bash
# Build the book (heavy artifacts included) in a screen session, then preview it.
# Detach with Ctrl-a d; this script keeps waiting and starts the preview when the render succeeds.
session=render
status=artifacts/render.status

if screen -list | grep -q "\.$session[[:space:]]"; then
    echo "Render already in progress (screen session '$session'); reattach with: screen -r $session" >&2
    exit 1
fi

mkdir -p artifacts
rm -f "$status"

RUN_HEAVY=1 screen -S "$session" bash -c \
    "quarto render docs/ --log artifacts/render.log; echo \$? > $status"

# Wait for the render's exit status. Give up if the session ended without writing one (e.g. killed).
while [ ! -f "$status" ]; do
    if ! screen -list | grep -q "\.$session[[:space:]]"; then
        [ -f "$status" ] && break
        echo "Render session ended without a status; see artifacts/render.log" >&2
        exit 1
    fi
    sleep 5
done

if [ "$(cat "$status")" = 0 ]; then
    quarto preview docs/    # serve the rendered site, refreshing on changes (no RUN_HEAVY)
else
    echo "Render failed (exit code $(cat "$status")); see artifacts/render.log" >&2
    exit 1
fi
