from __future__ import annotations


EXPERIMENT_RECORDING_UI_MARKER = "mujoco-experiment-recording-ui"


EXPERIMENT_RECORDING_CSS = r"""
<style id="mujoco-experiment-recording-ui-style">
  .experiment-recording-control {
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
    flex-wrap: wrap;
  }

  .experiment-recording-indicator {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    min-width: 7.5rem;
    font-size: 0.86rem;
    color: #64748b;
  }

  .experiment-recording-dot {
    width: 0.68rem;
    height: 0.68rem;
    border-radius: 999px;
    background: #94a3b8;
    box-shadow: 0 0 0 0 rgba(220, 38, 38, 0);
  }

  .experiment-recording-control[data-state="recording"]
    .experiment-recording-dot {
    background: #dc2626;
    box-shadow: 0 0 0 0.24rem rgba(220, 38, 38, 0.16);
    animation: experiment-recording-pulse 1.35s ease-in-out infinite;
  }

  .experiment-recording-control[data-state="starting"]
    .experiment-recording-dot,
  .experiment-recording-control[data-state="finalizing"]
    .experiment-recording-dot,
  .experiment-recording-control[data-state="unknown"]
    .experiment-recording-dot {
    background: #f59e0b;
  }

  .experiment-recording-control[data-state="error"]
    .experiment-recording-dot {
    background: #dc2626;
  }

  @keyframes experiment-recording-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.45; }
  }

  @media (prefers-reduced-motion: reduce) {
    .experiment-recording-control[data-state="recording"]
      .experiment-recording-dot {
      animation: none;
    }
  }
</style>
""".strip()


EXPERIMENT_RECORDING_SCRIPT = r"""
<script id="mujoco-experiment-recording-ui">
(() => {
  "use strict";

  const STATUS_TYPE = "experiment_recording_status_v1";
  const ACTIVE_RECORDING_STATES = ["starting", "recording", "finalizing"];
  const TERMINAL_RECORDING_STATES = ["completed", "error"];
  const sessionNode = document.getElementById("session-id");
  const startGeminiButton = document.getElementById("generate-btn");
  if (!sessionNode || !startGeminiButton || !startGeminiButton.parentElement) {
    return;
  }

  const control = document.createElement("span");
  control.className = "experiment-recording-control";
  control.dataset.state = "idle";

  const button = document.createElement("button");
  button.id = "experiment-recording-btn";
  button.className = "secondary";
  button.type = "button";
  button.disabled = true;
  button.textContent = "Record Experiment";
  button.title =
    "Explicitly record the robot scene, your microphone audio, and the assistant audio on this server.";

  const indicator = document.createElement("span");
  indicator.id = "experiment-recording-indicator";
  indicator.className = "experiment-recording-indicator";
  indicator.setAttribute("role", "status");
  indicator.setAttribute("aria-live", "polite");

  const dot = document.createElement("span");
  dot.className = "experiment-recording-dot";
  dot.setAttribute("aria-hidden", "true");

  const label = document.createElement("span");
  label.textContent = "Not recording";
  indicator.append(dot, label);
  control.append(button, indicator);
  startGeminiButton.parentElement.append(control);

  let uiGeneration = 0;
  let recordingId = "";
  let recordingSessionId = "";
  let startRequestId = "";
  let pollTimer = null;
  const REQUEST_TIMEOUT_MS = 10000;

  function currentSessionId() {
    const value = String(sessionNode.textContent || "").trim();
    return value && value !== "-" ? value : "";
  }

  function stopPolling() {
    if (pollTimer !== null) {
      clearTimeout(pollTimer);
      pollTimer = null;
    }
  }

  function render(state, message = "") {
    control.dataset.state = state;
    if (state === "recording") {
      button.disabled = false;
      button.textContent = "Stop Recording";
      label.textContent = "Recording";
      return;
    }
    if (state === "starting") {
      button.disabled = true;
      button.textContent = "Starting…";
      label.textContent = "Preparing recorder";
      return;
    }
    if (state === "finalizing") {
      button.disabled = true;
      button.textContent = "Finalizing…";
      label.textContent = "Saving recording";
      return;
    }
    if (state === "completed") {
      button.disabled = !currentSessionId();
      button.textContent = "Record Experiment";
      label.textContent = message || "Recording saved";
      return;
    }
    if (state === "error") {
      button.disabled = !currentSessionId();
      button.textContent = recordingId ? "Retry Stop" : "Retry Recording";
      label.textContent = message || "Recording failed";
      return;
    }
    if (state === "unknown") {
      button.disabled = false;
      button.textContent = recordingId ? "Retry Stop" : "Retry Recording";
      label.textContent = message || "Recording status unknown";
      return;
    }
    button.disabled = !currentSessionId();
    button.textContent = "Record Experiment";
    label.textContent = "Not recording";
  }

  function statusEnvelopeIsValid(payload) {
    if (!payload || payload.type !== STATUS_TYPE) {
      return false;
    }
    if (ACTIVE_RECORDING_STATES.includes(payload.state)) {
      return payload.active === true;
    }
    if (TERMINAL_RECORDING_STATES.includes(payload.state)) {
      return payload.active === false;
    }
    return false;
  }

  function responseIsCurrent(payload, generation, sessionId, expectedRecordingId) {
    if (generation !== uiGeneration || sessionId !== currentSessionId()) {
      return false;
    }
    if (!statusEnvelopeIsValid(payload) || payload.session_id !== sessionId) {
      return false;
    }
    if (!expectedRecordingId || payload.recording_id !== expectedRecordingId) {
      return false;
    }
    if (startRequestId && payload.request_id !== startRequestId) {
      return false;
    }
    return true;
  }

  async function fetchWithTimeout(resource, options) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    try {
      return await fetch(resource, { ...options, signal: controller.signal });
    } finally {
      clearTimeout(timeout);
    }
  }

  async function readJson(response) {
    let payload = null;
    try {
      payload = await response.json();
    } catch (_error) {
      throw new Error(`Recording service returned HTTP ${response.status}`);
    }
    if (!response.ok) {
      const error = new Error(
        String(payload.message || payload.error_code || `HTTP ${response.status}`)
      );
      if (payload && payload.type === STATUS_TYPE) {
        error.recordingPayload = payload;
      }
      throw error;
    }
    return payload;
  }

  function newRequestId() {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return window.crypto.randomUUID();
    }
    const random = Math.random().toString(16).slice(2);
    return `${Date.now().toString(16)}-${random}`;
  }

  function bestEffortStop(sessionId, ownedRecordingId) {
    if (!sessionId || !ownedRecordingId) {
      return;
    }
    fetch("/api/experiment-recording/stop", {
      method: "POST",
      cache: "no-store",
      credentials: "same-origin",
      keepalive: true,
      headers: {
        "Content-Type": "application/json",
        "X-Experiment-Recording": "1",
      },
      body: JSON.stringify({
        session_id: sessionId,
        recording_id: ownedRecordingId,
      }),
    }).catch(() => {});
  }

  function schedulePoll(generation, sessionId) {
    stopPolling();
    pollTimer = setTimeout(async () => {
      try {
        const query = new URLSearchParams({ recording_id: recordingId });
        const response = await fetchWithTimeout(
          `/api/experiment-recording/status/${encodeURIComponent(sessionId)}?${query}`,
          { method: "GET", cache: "no-store", credentials: "same-origin" }
        );
        const payload = await readJson(response);
        if (generation !== uiGeneration || sessionId !== currentSessionId()) {
          return;
        }
        if (!responseIsCurrent(payload, generation, sessionId, recordingId)) {
          throw new Error("Recording service returned an invalid status response");
        }
        if (["starting", "recording", "finalizing"].includes(payload.state)) {
          render(payload.state, payload.message || "");
          schedulePoll(generation, sessionId);
        } else {
          recordingId = "";
          recordingSessionId = "";
          startRequestId = "";
          render(payload.state, payload.message || "");
        }
      } catch (error) {
        if (generation === uiGeneration && sessionId === currentSessionId()) {
          // A transient status failure is not proof that recording stopped.
          // This includes namespaced 4xx responses: they reject the status
          // request but do not prove that the background job is terminal.
          // Retain the ownership token so the operator can still stop it.
          render("unknown", error?.message || "Recording status unknown");
          schedulePoll(generation, sessionId);
        }
      }
    }, 1000);
  }

  async function startRecording() {
    const sessionId = currentSessionId();
    if (!sessionId) {
      render("error", "Connect WebRTC first");
      return;
    }
    const generation = ++uiGeneration;
    stopPolling();
    recordingId = "";
    recordingSessionId = sessionId;
    startRequestId = startRequestId || newRequestId();
    render("starting");
    try {
      const response = await fetchWithTimeout("/api/experiment-recording/start", {
        method: "POST",
        cache: "no-store",
        credentials: "same-origin",
        headers: {
          "Content-Type": "application/json",
          "X-Experiment-Recording": "1",
        },
        body: JSON.stringify({
          session_id: sessionId,
          request_id: startRequestId,
        }),
      });
      const payload = await readJson(response);
      if (generation !== uiGeneration || sessionId !== currentSessionId()) {
        bestEffortStop(sessionId, String(payload?.recording_id || ""));
        return;
      }
      if (
        !statusEnvelopeIsValid(payload) ||
        payload.session_id !== sessionId ||
        payload.request_id !== startRequestId
      ) {
        throw new Error("Recording service returned an invalid response");
      }
      recordingId = String(payload.recording_id || "");
      if (
        !recordingId ||
        !["starting", "recording", "finalizing"].includes(payload.state)
      ) {
        throw new Error(String(payload.message || "Recorder did not start"));
      }
      render(payload.state, payload.message || "");
      schedulePoll(generation, sessionId);
    } catch (error) {
      if (generation === uiGeneration && sessionId === currentSessionId()) {
        recordingId = "";
        const payload = error?.recordingPayload;
        const precommitError = [
          "recording_unavailable",
          "recording_request_forbidden",
          "invalid_recording_request",
          "session_not_recordable",
        ].includes(String(payload?.error_code || ""));
        const ownedTerminal =
          statusEnvelopeIsValid(payload) &&
          TERMINAL_RECORDING_STATES.includes(payload.state) &&
          payload?.session_id === sessionId &&
          payload?.request_id === startRequestId &&
          typeof payload?.recording_id === "string" &&
          payload.recording_id.length > 0;
        if (ownedTerminal && payload?.state === "completed") {
          recordingSessionId = "";
          startRequestId = "";
          render("completed", payload.message || "Recording stopped before capture began");
        } else if (
          precommitError ||
          (ownedTerminal && String(payload?.error_code || ""))
        ) {
          // A parsed HTTP error is a definite server decision. A new click must
          // use a fresh idempotency key after the operator fixes the cause.
          recordingSessionId = "";
          startRequestId = "";
          render("error", error.message || "Recording could not start");
        } else {
          // The response may have been lost after the server committed start.
          // Keep the idempotency key so Retry Recording recovers that exact job.
          render("unknown", error?.message || "Recording start status unknown");
        }
      }
    }
  }

  async function stopRecording() {
    const sessionId = recordingSessionId;
    const ownedRecordingId = recordingId;
    if (!sessionId || !ownedRecordingId) {
      render("idle");
      return;
    }
    const generation = ++uiGeneration;
    stopPolling();
    render("finalizing");
    try {
      const response = await fetchWithTimeout("/api/experiment-recording/stop", {
        method: "POST",
        cache: "no-store",
        credentials: "same-origin",
        headers: {
          "Content-Type": "application/json",
          "X-Experiment-Recording": "1",
        },
        body: JSON.stringify({
          session_id: sessionId,
          recording_id: ownedRecordingId,
        }),
      });
      const payload = await readJson(response);
      if (generation !== uiGeneration || sessionId !== currentSessionId()) {
        return;
      }
      if (!responseIsCurrent(payload, generation, sessionId, ownedRecordingId)) {
        throw new Error("Recording service returned an invalid response");
      }
      recordingId = ownedRecordingId;
      if (["starting", "recording", "finalizing"].includes(payload.state)) {
        render(payload.state, payload.message || "");
        schedulePoll(generation, sessionId);
      } else {
        recordingId = "";
        recordingSessionId = "";
        startRequestId = "";
        render(payload.state, payload.message || "");
      }
    } catch (error) {
      if (generation === uiGeneration && sessionId === currentSessionId()) {
        // An HTTP rejection only describes this control request; it does not
        // prove the background recorder stopped. Preserve the exact owner token
        // until a successful status/stop response reports a terminal state.
        render("unknown", error?.message || "Recording stop status unknown");
        schedulePoll(generation, sessionId);
      }
    }
  }

  button.addEventListener("click", () => {
    if (recordingId) {
      stopRecording();
    } else {
      startRecording();
    }
  });

  const sessionObserver = new MutationObserver(() => {
    const sessionId = currentSessionId();
    if (recordingSessionId && sessionId !== recordingSessionId) {
      ++uiGeneration;
      stopPolling();
      bestEffortStop(recordingSessionId, recordingId);
      recordingId = "";
      recordingSessionId = "";
      startRequestId = "";
      render("idle");
      return;
    }
    if (!recordingId) {
      render(control.dataset.state === "error" ? "error" : "idle", label.textContent);
    }
  });
  sessionObserver.observe(sessionNode, { childList: true, characterData: true, subtree: true });
  window.addEventListener("beforeunload", () => {
    bestEffortStop(recordingSessionId, recordingId);
  });
  window.addEventListener("pagehide", () => {
    bestEffortStop(recordingSessionId, recordingId);
  });
  render("idle");
})();
</script>
""".strip()


def inject_experiment_recording_ui(html: str) -> tuple[str, bool]:
    """Add the explicit recording control without touching dialogue UI state."""

    style_marker = 'id="mujoco-experiment-recording-ui-style"'
    script_marker = f'id="{EXPERIMENT_RECORDING_UI_MARKER}"'
    style_count = html.count(style_marker)
    script_count = html.count(script_marker)
    if style_count or script_count:
        if style_count == 1 and script_count == 1:
            return html, False
        raise RuntimeError("experiment recording UI is partially or multiply installed")

    head_end = html.lower().find("</head>")
    body_end = html.lower().find("</body>")
    if head_end < 0 or body_end < 0:
        raise RuntimeError("cannot install experiment recording UI: HTML anchors missing")
    patched = html[:head_end] + EXPERIMENT_RECORDING_CSS + "\n" + html[head_end:]
    body_end = patched.lower().find("</body>")
    patched = patched[:body_end] + EXPERIMENT_RECORDING_SCRIPT + "\n" + patched[body_end:]
    if patched.count(style_marker) != 1 or patched.count(script_marker) != 1:
        raise RuntimeError("experiment recording UI postcondition failed")
    return patched, True
