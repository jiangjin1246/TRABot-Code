"""
SoulX-FlashHead WebRTC streaming server.

基于 gradio_app_streaming.py 的推理流程改造而来：
- 保留模型加载、音频切片、逐 chunk 推理逻辑
- 将 MP4 分段预览改为 aiortc 音视频轨实时推送
- 额外保留完整 MP4 落盘，便于回看

运行方式:
    python webrtc_app_streaming.py --host 0.0.0.0 --port 8080
"""
import argparse
import asyncio
import base64
import concurrent.futures
import fractions
import json
import os
import queue
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import imageio
import librosa
import numpy as np
import torch
from aiohttp import WSMsgType, web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame, VideoFrame
from loguru import logger

try:
    from google import genai
    from google.genai import types
    GENAI_IMPORT_ERROR = None
except ImportError as exc:
    genai = None
    types = None
    GENAI_IMPORT_ERROR = exc

from flash_head.inference import (
    get_audio_embedding,
    get_base_data,
    get_infer_params,
    get_pipeline,
    run_pipeline,
)
from flash_head_robot_bridge import (
    DEFAULT_PRESENTATION_GESTURE_ID,
    PRESENTATION_GESTURE_IDS,
    pick_robot_action,
    robot_runtime,
    trigger_robot_action,
)


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>NoorAct | CVLab@UAEU</title>
  <style>
    :root {
      --bg: #f3f6fb;
      --panel: rgba(248, 251, 255, 0.9);
      --card: rgba(255, 255, 255, 0.78);
      --text: #112033;
      --muted: #5b6b7d;
      --line: rgba(17, 32, 51, 0.1);
      --accent: #004b87;
      --accent-strong: #002d62;
      --ok: #0f7b57;
      --warn: #b86a18;
      --shadow: 0 24px 56px rgba(18, 44, 86, 0.12);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(59, 130, 246, 0.14), transparent 28%),
        radial-gradient(circle at top right, rgba(0, 91, 150, 0.18), transparent 24%),
        linear-gradient(145deg, #f5f9ff 0%, #edf3fb 48%, #e9eef7 100%);
      min-height: 100vh;
      padding: 24px;
    }

    .shell {
      max-width: 1320px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }

    .content-grid {
      display: grid;
      grid-template-columns: minmax(340px, 390px) minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }

    .panel {
      background: var(--panel);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.5);
      border-radius: 28px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .masthead {
      padding: 18px 22px;
      background:
        radial-gradient(circle at top left, rgba(66, 121, 196, 0.16), transparent 22%),
        linear-gradient(135deg, rgba(255, 255, 255, 0.94), rgba(236, 244, 255, 0.92));
    }

    .masthead-main {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 24px;
    }

    .masthead-copy {
      min-width: 0;
      max-width: 860px;
    }

    .masthead-badge {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      padding: 8px 14px;
      border-radius: 999px;
      background: rgba(0, 75, 135, 0.08);
      border: 1px solid rgba(0, 75, 135, 0.12);
      font-size: 12px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--accent-strong);
      font-weight: 700;
      margin-bottom: 16px;
    }

    .masthead-title {
      margin: 0;
      font-size: clamp(24px, 2.8vw, 38px);
      line-height: 1.08;
      letter-spacing: -0.04em;
      max-width: 18ch;
    }

    .masthead-subtitle {
      margin: 10px 0 0;
      max-width: 58ch;
      font-size: 15px;
      line-height: 1.6;
      color: var(--muted);
    }

    .masthead-logos {
      display: flex;
      align-items: center;
      gap: 14px;
      flex-shrink: 0;
      align-self: stretch;
    }

    .masthead-logos img {
      display: block;
      height: 58px;
      width: auto;
      object-fit: contain;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.96);
      padding: 10px 14px;
      border: 1px solid rgba(0, 75, 135, 0.1);
      box-shadow: 0 14px 28px rgba(20, 55, 102, 0.08);
    }

    .controls {
      padding: 18px;
    }

    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 16px;
      margin-top: 14px;
    }

    .card h2 {
      margin: 0 0 14px;
      font-size: 15px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--muted);
    }

    .field {
      display: grid;
      gap: 8px;
      margin-bottom: 14px;
    }

    .field:last-child {
      margin-bottom: 0;
    }

    .field.compact {
      margin-bottom: 10px;
    }

    label {
      font-size: 13px;
      color: var(--muted);
      font-weight: 600;
    }

    input[type="text"],
    input[type="password"],
    input[type="number"],
    select,
    textarea {
      width: 100%;
      border: 1px solid rgba(37, 29, 24, 0.12);
      border-radius: 14px;
      padding: 12px 14px;
      font-size: 14px;
      color: var(--text);
      background: rgba(255, 255, 255, 0.9);
    }

    input[type="file"] {
      width: 100%;
      border: 1px dashed rgba(37, 29, 24, 0.18);
      border-radius: 14px;
      padding: 12px;
      background: rgba(255, 255, 255, 0.55);
    }

    .row-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }

    .ref-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 112px;
      gap: 12px;
      align-items: start;
    }

    .ref-preview {
      border-radius: 18px;
      overflow: hidden;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.92), rgba(240, 246, 255, 0.94));
      min-height: 112px;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
    }

    .ref-preview img {
      width: 100%;
      height: 112px;
      object-fit: cover;
      display: block;
    }

    .ref-preview span {
      position: absolute;
      inset: auto 8px 8px 8px;
      font-size: 11px;
      line-height: 1.35;
      color: #f4f8ff;
      background: rgba(9, 23, 38, 0.72);
      border-radius: 10px;
      padding: 6px 8px;
      text-align: center;
      backdrop-filter: blur(8px);
    }

    .inline {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }

    .buttons {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 18px;
    }

    button {
      border: none;
      border-radius: 999px;
      padding: 12px 18px;
      font-size: 14px;
      font-weight: 700;
      cursor: pointer;
      transition: transform 0.16s ease, box-shadow 0.16s ease, opacity 0.16s ease;
    }

    button:hover {
      transform: translateY(-1px);
    }

    button:disabled {
      cursor: not-allowed;
      opacity: 0.55;
      transform: none;
    }

    .primary {
      background: linear-gradient(135deg, #005ea6 0%, #003f78 100%);
      color: #f7fbff;
      box-shadow: 0 12px 24px rgba(0, 63, 120, 0.22);
    }

    .secondary {
      background: rgba(255, 255, 255, 0.85);
      color: var(--text);
      border: 1px solid var(--line);
    }

    .ghost {
      background: rgba(183, 72, 43, 0.1);
      color: var(--accent-strong);
    }

    .stage {
      position: relative;
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .status-bar {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
      flex-wrap: wrap;
      padding: 12px 16px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.62);
      border: 1px solid var(--line);
    }

    .status-label {
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 4px;
    }

    .status-value {
      font-size: 15px;
      font-weight: 700;
    }

    .status-value.ok {
      color: var(--ok);
    }

    .status-value.warn {
      color: var(--warn);
    }

    .video-shell {
      position: relative;
      width: min(100%, 600px);
      aspect-ratio: 600 / 1024;
      margin: 0 auto;
      border-radius: 0;
      overflow: hidden;
      border: 1px solid rgba(255, 255, 255, 0.38);
      background:
        radial-gradient(circle at 20% 20%, rgba(88, 142, 214, 0.32), transparent 28%),
        linear-gradient(180deg, #123055 0%, #091726 100%);
      min-height: 0;
      max-height: min(1024px, calc(100vh - 48px));
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }

    video {
      width: 100%;
      height: 100%;
      min-height: 0;
      display: block;
      object-fit: cover;
      object-position: 50% 50%;
      background: transparent;
    }

    .live-player {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
    }

    .placeholder {
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
      text-align: center;
	      color: rgba(244, 248, 255, 0.9);
      font-size: 18px;
      line-height: 1.7;
      pointer-events: none;
      background:
        linear-gradient(180deg, rgba(21, 12, 8, 0.12), rgba(21, 12, 8, 0.32));
    }

    .meta {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }

    .meta-item {
      padding: 12px;
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.6);
      border: 1px solid var(--line);
    }

    .meta-item span {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
    }

    .meta-item strong {
      display: block;
      font-size: 14px;
      word-break: break-all;
    }

    .tips {
      font-size: 13px;
      color: var(--muted);
      line-height: 1.7;
    }

    .transcript-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }

    .transcript-card {
      padding: 14px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.68);
      border: 1px solid var(--line);
      min-height: 120px;
    }

    .transcript-card span {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }

    .transcript-card p {
      margin: 0;
      font-size: 14px;
      line-height: 1.7;
      white-space: pre-wrap;
      word-break: break-word;
    }

    .feed {
      min-height: 140px;
      max-height: 220px;
      overflow: auto;
      padding: 16px;
      border-radius: 18px;
      background: rgba(28, 23, 20, 0.95);
      color: #f7efe6;
      font-family: "SFMono-Regular", "Consolas", monospace;
      font-size: 12px;
      line-height: 1.7;
      border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .feed .line {
      margin-bottom: 6px;
      word-break: break-word;
    }

    .feed .line.ok {
      color: #8ef0db;
    }

    .feed .line.warn {
      color: #ffcb8f;
    }

    @media (max-width: 1180px) {
      .content-grid {
        grid-template-columns: 1fr;
      }

      .masthead-main {
        flex-direction: column;
        align-items: flex-start;
      }

      .masthead-logos {
        align-self: auto;
      }
    }

    @media (max-width: 1024px) {

      .video-shell,
      video {
        min-height: 0;
      }
    }

    @media (max-width: 640px) {
      body {
        padding: 16px;
      }

      .controls,
      .stage {
        padding: 18px;
      }

      .row-2,
      .meta,
      .transcript-grid {
        grid-template-columns: 1fr;
      }

      .ref-grid {
        grid-template-columns: 1fr;
      }

      .ref-preview {
        min-height: 140px;
      }

      .ref-preview img {
        height: 140px;
      }

      .masthead {
        padding: 18px;
      }

      .masthead-title {
        max-width: none;
      }

      .masthead-logos {
        width: 100%;
        flex-wrap: wrap;
      }

      .masthead-logos img {
        height: 52px;
      }
    }
    @media (max-width: 700px) and (orientation: portrait) {
      body {
        padding: 0;
        overflow: hidden;
        background: #000;
      }

      .masthead,
      .controls,
      .status-bar,
      .meta,
      .transcript-grid,
      .stage > .transcript-card,
      .feed {
        display: none;
      }

      .shell,
      .content-grid,
      .stage {
        width: 100vw;
        height: 100vh;
        max-width: none;
        margin: 0;
      }

      .stage {
        padding: 0;
        gap: 0;
        border: 0;
        border-radius: 0;
        box-shadow: none;
        background: #000;
      }

      .video-shell {
        width: 100vw;
        height: 100vh;
        max-height: none;
        aspect-ratio: auto;
        border: 0;
        margin: 0;
      }
    }

  </style>
</head>
<body>
  <div class="shell">
    <section class="panel masthead">
      <div class="masthead-main">
        <div class="masthead-copy">
          <div class="masthead-badge">Realtime Digital Human Demo</div>
          <h1 class="masthead-title">NoorAct - A Unified Platform Created by CVLab@UAEU</h1>
          <p class="masthead-subtitle">A unified research interface for live Gemini conversation, audio-driven avatar generation, and low-latency WebRTC delivery.</p>
          <p class="masthead-subtitle"><strong>TRABot</strong> (<em>Talk, Render, Act</em>, <a href="https://arxiv.org/abs/2610.06153" target="_blank" rel="noopener">arXiv:2610.06153</a>) is a research result of NoorAct.</p>
        </div>
        <div class="masthead-logos">
          <img src="__LOGO_ASSET_URL__" alt="NoorAct logo" />
          <img src="__UAEU_ASSET_URL__" alt="UAEU logo" />
        </div>
      </div>
    </section>

    <div class="content-grid">
      <section class="panel controls">
        <div class="card">
        <h2>Connection Settings</h2>
        <div class="field">
          <div class="inline">
            <input id="use-stun" type="checkbox" checked />
            <label for="use-stun">Use Google STUN for remote access across devices</label>
          </div>
        </div>
        <div class="buttons">
          <button id="connect-btn" class="secondary" type="button">Connect WebRTC</button>
          <button id="stop-btn" class="ghost" type="button" disabled>Stop Current Session</button>
        </div>
      </div>

      <div class="card">
        <h2>Avatar Settings</h2>
        <div class="ref-grid">
          <div>
            <div class="field compact">
              <label for="cond-image-select">Reference Preset</label>
              <select id="cond-image-select">
                __REF_PRESET_OPTIONS__
                <option value="__custom__">Custom path or upload</option>
              </select>
            </div>
            <div class="field compact">
              <label for="cond-image-upload">Upload Condition Image</label>
              <input id="cond-image-upload" type="file" accept="image/*" />
            </div>
            <div class="field">
              <label for="cond-image-path">Or provide an image path on the server</label>
              <input id="cond-image-path" type="text" value="__DEFAULT_COND_IMAGE__" />
            </div>
          </div>
          <div class="ref-preview">
            <img id="cond-image-preview" src="__DEFAULT_COND_IMAGE_PREVIEW__" alt="Reference preview" />
            <span id="cond-image-preview-label">Default reference</span>
          </div>
        </div>
        <div class="row-2">
          <div class="field">
            <label for="ckpt-dir">Checkpoint Directory</label>
            <input id="ckpt-dir" type="text" value="models/SoulX-FlashHead-1_3B" />
          </div>
          <div class="field">
            <label for="wav2vec-dir">Wav2Vec Directory</label>
            <input id="wav2vec-dir" type="text" value="models/wav2vec2-base-960h" />
          </div>
        </div>

        <div class="row-2">
          <div class="field">
            <label for="model-type">Model Type</label>
            <select id="model-type">
              <option value="lite" selected>lite</option>
              <option value="pro">pro</option>
            </select>
          </div>
          <div class="field">
            <label for="seed">Random Seed</label>
            <input id="seed" type="number" value="9999" step="1" />
          </div>
        </div>

        <div class="field">
          <div class="inline">
            <input id="use-face-crop" type="checkbox" />
            <label for="use-face-crop">Use Face Crop</label>
          </div>
        </div>
      </div>

      <div class="card">
        <h2>Gemini Live Settings</h2>
        <div class="field">
          <label for="api-key">Gemini API Key</label>
          <input id="api-key" type="password" placeholder="Leave blank to use the server-side GEMINI_API_KEY" />
        </div>
        <div class="row-2">
          <div class="field">
            <label for="gemini-model">Model</label>
            <input id="gemini-model" type="text" value="__DEFAULT_GEMINI_MODEL__" />
          </div>
          <div class="field">
            <label for="gemini-voice">Voice</label>
            <select id="gemini-voice">
              __GEMINI_VOICE_OPTIONS__
            </select>
          </div>
        </div>
        <div class="field">
          <label for="gemini-temperature">Temperature</label>
          <input id="gemini-temperature" type="number" value="__DEFAULT_GEMINI_TEMPERATURE__" min="0" max="2" step="0.1" />
        </div>
        <div class="field">
          <label for="system-prompt">System Prompt</label>
          <textarea id="system-prompt" rows="5">__DEFAULT_GEMINI_PROMPT__</textarea>
        </div>
        <div class="field">
          <label for="text-input">Optional Text Prompt</label>
          <textarea id="text-input" rows="3" placeholder='Optional: send Gemini a text instruction such as "Please introduce yourself briefly first."'></textarea>
        </div>

        <div class="buttons">
          <button id="generate-btn" class="primary" type="button" disabled>Start Gemini Conversation</button>
          <button id="send-text-btn" class="secondary" type="button" disabled>Send Text</button>
        </div>
      </div>

        <div class="card tips">
        This version no longer uses random audio files. Your browser sends live PCM microphone audio to Gemini, and Gemini's reply audio drives SoulX-FlashHead while streaming synchronized audio and video through WebRTC.
        </div>
      </section>

      <section class="panel stage">
        <div class="video-shell">
        <video id="player" class="live-player" autoplay playsinline controls></video>
        <div id="placeholder" class="placeholder">Connect first, then start the Gemini conversation. The first synchronized frames will appear here through WebRTC.</div>
        </div>

        <div class="status-bar">
        <div>
          <div class="status-label">Current Status</div>
          <div id="status" class="status-value">Disconnected</div>
        </div>
        <div>
          <div class="status-label">Session ID</div>
          <div id="session-id" class="status-value">-</div>
        </div>
        </div>

        <div class="meta">
        <div class="meta-item">
          <span>Peer Connection</span>
          <strong id="pc-state">new</strong>
        </div>
        <div class="meta-item">
          <span>Generation State</span>
          <strong id="gen-state">idle</strong>
        </div>
        <div class="meta-item">
          <span>Current Turn</span>
          <strong id="selected-audio">-</strong>
        </div>
        </div>

        <div class="transcript-grid">
        <div class="transcript-card">
          <span>Input Transcription</span>
          <p id="input-transcript">Waiting for your voice input.</p>
        </div>
        <div class="transcript-card">
          <span>Output Transcription</span>
          <p id="output-transcript">Waiting for Robot's reply.</p>
        </div>
        </div>

        <div class="transcript-card">
          <span>Robot Motion</span>
          <p id="robot-mode">Mode: waiting for robot event.</p>
          <p id="robot-action">Action: no action triggered yet.</p>
          <p id="robot-result">Result: idle.</p>
          <p id="robot-detail">Detail: the latest matched robot action will appear here.</p>
        </div>

        <div id="event-feed" class="feed"></div>
      </section>
    </div>
  </div>

  <script>
    const serverWsUrl = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
    const ICE_GATHER_TIMEOUT_MS = 1200;
    const DISCONNECTED_WARN_DELAY_MS = 2500;
    const DISCONNECTED_STATUS_GRACE_MS = 15000;
    const CONTROL_KEEPALIVE_MS = 15000;
    let pc = null;
    let remoteStream = null;
    let sessionId = null;
    let pollHandle = null;
    let connectPromise = null;
    let controlSocket = null;
    let captureAudioContext = null;
    let micStream = null;
    let micSource = null;
    let micProcessor = null;
    let micMuteNode = null;
    let micResampler = null;
    let isMicRunning = false;
    let pendingMicStart = false;
    let isGeminiConnected = false;
    let condPreviewObjectUrl = null;
    let disconnectStatusTimer = null;
    let disconnectLostStatusTimer = null;
    let controlKeepaliveHandle = null;
    let currentTurnHasRobotAction = false;

    function $(id) {
      return document.getElementById(id);
    }

    function pushFeed(text, tone = "") {
      const node = document.createElement("div");
      node.className = `line${tone ? " " + tone : ""}`;
      const stamp = new Date().toLocaleTimeString("en-GB", { hour12: false });
      node.textContent = `[${stamp}] ${text}`;
      $("event-feed").prepend(node);
    }

    function setStatus(text, cls = "") {
      const node = $("status");
      node.textContent = text;
      node.className = "status-value" + (cls ? " " + cls : "");
    }

    function setPcState(text) {
      $("pc-state").textContent = text;
    }

    function clearDisconnectStatusTimer() {
      if (disconnectStatusTimer) {
        clearTimeout(disconnectStatusTimer);
        disconnectStatusTimer = null;
      }
      if (disconnectLostStatusTimer) {
        clearTimeout(disconnectLostStatusTimer);
        disconnectLostStatusTimer = null;
      }
    }

    function stopControlKeepalive() {
      if (controlKeepaliveHandle) {
        clearInterval(controlKeepaliveHandle);
        controlKeepaliveHandle = null;
      }
    }

    function startControlKeepalive() {
      stopControlKeepalive();
      controlKeepaliveHandle = setInterval(() => {
        if (controlSocket && controlSocket.readyState === WebSocket.OPEN) {
          controlSocket.send(JSON.stringify({ type: "ping" }));
        }
      }, CONTROL_KEEPALIVE_MS);
    }

    function setGenState(text) {
      $("gen-state").textContent = text;
    }

    function setOutputPath(text) {
      const node = $("output-path");
      if (node) {
        node.textContent = text || "-";
      }
    }

    function setSelectedAudio(text) {
      $("selected-audio").textContent = text || "-";
    }

    function setInputTranscript(text) {
      $("input-transcript").textContent = text || "Waiting for your voice input.";
    }

    function setOutputTranscript(text) {
      $("output-transcript").textContent = text || "Waiting for Robot's reply.";
    }

    function setRobotMode(text) {
      $("robot-mode").textContent = text || "Mode: waiting for robot event.";
    }

    function setRobotAction(text) {
      $("robot-action").textContent = text || "Action: no action triggered yet.";
    }

    function setRobotResult(text) {
      $("robot-result").textContent = text || "Result: idle.";
    }

    function setRobotDetail(text) {
      $("robot-detail").textContent = text || "Detail: the latest matched robot action will appear here.";
    }

    function resetRobotPanel() {
      currentTurnHasRobotAction = false;
      setRobotMode("Mode: waiting for robot event.");
      setRobotAction("Action: no action triggered yet.");
      setRobotResult("Result: idle.");
      setRobotDetail("Detail: the latest matched robot action will appear here.");
    }

    function markRobotPanelWaitingForCurrentTurn() {
      if (currentTurnHasRobotAction) {
        return;
      }
      setRobotAction("Action: waiting for current turn.");
      setRobotResult("Result: no action triggered for this turn yet.");
      setRobotDetail("Detail: if the current turn matches a robot action, it will appear here.");
    }

    function updateRobotPanel(payload) {
      currentTurnHasRobotAction = Boolean(payload.executed);
      const runtime = payload.robot || {};
      const mode = runtime.simulationMode ? "simulation" : runtime.enabled ? "real robot" : "disabled";
      setRobotMode(`Mode: ${mode}`);

      const label = payload.label || payload.action || "no action";
      setRobotAction(`Action: ${label}`);

      if (payload.executed) {
        setRobotResult(`Result: ${payload.simulated ? "simulated trigger" : "executed on robot"}.`);
        const playback =
          payload.playbackSource === "builtin"
            ? "builtin action"
            : payload.playbackSource === "motion_file"
              ? "motion file"
              : payload.playbackSource === "preset"
                ? "preset pose"
                : "unknown";
        const detailParts = [
          `Source: ${playback}`,
          `Kind: ${payload.kind || "unknown"}`,
        ];
        if (Array.isArray(payload.command) && payload.command.length) {
          detailParts.push(`Command: ${payload.command.join(" ")}`);
        }
        setRobotDetail(`Detail: ${detailParts.join(" | ")}`);
        return;
      }

      if (payload.matched === false) {
        setRobotResult("Result: no mapped action matched.");
        setRobotDetail(`Detail: ${payload.reason || "Gemini reply did not match a configured robot action."}`);
        return;
      }

      if (payload.error) {
        setRobotResult("Result: trigger failed.");
        setRobotDetail(`Detail: ${payload.error}`);
        return;
      }

      setRobotResult("Result: idle.");
      setRobotDetail("Detail: the latest matched robot action will appear here.");
    }

    function serverPathToPreviewUrl(path) {
      if (!path) {
        return "";
      }
      if (path.startsWith("examples/")) {
        return `/${path}`;
      }
      if (path.startsWith("assets/")) {
        return `/${path}`;
      }
      return "";
    }

    function syncReferencePreset() {
      const preset = $("cond-image-select");
      const path = $("cond-image-path").value.trim();
      const matchingOption = Array.from(preset.options).find((option) => option.value === path);
      preset.value = matchingOption ? path : "__custom__";
    }

    function updateReferencePreview() {
      const image = $("cond-image-preview");
      const label = $("cond-image-preview-label");
      const upload = $("cond-image-upload").files[0];
      const path = $("cond-image-path").value.trim();

      if (condPreviewObjectUrl) {
        URL.revokeObjectURL(condPreviewObjectUrl);
        condPreviewObjectUrl = null;
      }

      if (upload) {
        condPreviewObjectUrl = URL.createObjectURL(upload);
        image.src = condPreviewObjectUrl;
        label.textContent = upload.name;
        return;
      }

      const previewUrl = serverPathToPreviewUrl(path) || "__DEFAULT_COND_IMAGE_PREVIEW__";
      image.src = previewUrl;
      label.textContent = path ? path.split("/").pop() : "Default reference";
    }

    function toggleButtons(connected) {
      $("generate-btn").disabled = !connected;
      $("stop-btn").disabled = !connected;
      $("send-text-btn").disabled = !connected || !isGeminiConnected;
    }

    function playerNode() {
      return $("player");
    }

    function clearPlayerNode() {
      const player = playerNode();
      player.onended = null;
      player.onerror = null;
      player.ontimeupdate = null;
      player.pause();
      player.srcObject = null;
      player.removeAttribute("src");
      player.load();
    }

    function bindRemoteStream(force = false) {
      const player = playerNode();
      if (!__ENABLE_LIVE_TRACK_PLAYBACK__) {
        return player;
      }
      if (!remoteStream) {
        return player;
      }
      if (force || player.srcObject !== remoteStream) {
        player.srcObject = remoteStream;
      }
      $("placeholder").style.display = "none";
      return player;
    }

    function resumePlayer() {
      const player = bindRemoteStream();
      if (player.paused) {
        player.play().catch(() => {});
      }
    }

    function bytesToBase64(bytes) {
      let binary = "";
      const chunk = 0x8000;
      for (let i = 0; i < bytes.length; i += chunk) {
        binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
      }
      return btoa(binary);
    }

    function int16ArrayToBase64(int16Array) {
      return bytesToBase64(new Uint8Array(int16Array.buffer));
    }

    function floatToInt16(sample) {
      const clamped = Math.max(-1, Math.min(1, sample));
      return clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    }

    class PCMDownsampler {
      constructor(sourceRate, targetRate) {
        this.sourceRate = sourceRate;
        this.targetRate = targetRate;
        this.tail = new Float32Array(0);
      }

      process(chunk) {
        if (!chunk || chunk.length === 0) {
          return new Int16Array(0);
        }
        const merged = new Float32Array(this.tail.length + chunk.length);
        merged.set(this.tail, 0);
        merged.set(chunk, this.tail.length);

        if (this.sourceRate === this.targetRate) {
          this.tail = new Float32Array(0);
          return this.toInt16(merged);
        }

        const ratio = this.sourceRate / this.targetRate;
        const outputLength = Math.floor(merged.length / ratio);
        if (outputLength <= 0) {
          this.tail = merged;
          return new Int16Array(0);
        }

        const output = new Int16Array(outputLength);
        for (let i = 0; i < outputLength; i += 1) {
          const start = Math.floor(i * ratio);
          const end = Math.min(Math.floor((i + 1) * ratio), merged.length);
          let sum = 0;
          let count = 0;
          for (let j = start; j < end; j += 1) {
            sum += merged[j];
            count += 1;
          }
          const sample = count > 0 ? sum / count : merged[start] || 0;
          output[i] = floatToInt16(sample);
        }

        const consumed = Math.floor(outputLength * ratio);
        this.tail = merged.slice(consumed);
        return output;
      }

      toInt16(floatArray) {
        const output = new Int16Array(floatArray.length);
        for (let i = 0; i < floatArray.length; i += 1) {
          output[i] = floatToInt16(floatArray[i]);
        }
        return output;
      }
    }

    async function waitForIceGathering() {
      if (pc.iceGatheringState === "complete") {
        return;
      }

      await new Promise((resolve) => {
        let settled = false;
        const finish = () => {
          if (settled) {
            return;
          }
          settled = true;
          pc.removeEventListener("icegatheringstatechange", checkState);
          clearTimeout(timer);
          resolve();
        };

        const checkState = () => {
          if (pc.iceGatheringState === "complete") {
            finish();
          }
        };

        const timer = setTimeout(() => {
          console.warn(`ICE gathering timeout after ${ICE_GATHER_TIMEOUT_MS}ms, proceeding with partial candidates`);
          finish();
        }, ICE_GATHER_TIMEOUT_MS);

        pc.addEventListener("icegatheringstatechange", checkState);
      });
    }

    async function negotiate() {
      pc.addTransceiver("video", { direction: "recvonly" });
      pc.addTransceiver("audio", { direction: "recvonly" });

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      await waitForIceGathering();

      const response = await fetch("/offer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sdp: pc.localDescription.sdp,
          type: pc.localDescription.type,
        }),
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || "offer failed");
      }

      const answer = await response.json();
      sessionId = answer.session_id;
      $("session-id").textContent = sessionId;
      await pc.setRemoteDescription(answer);
    }

    async function startConnection() {
      if (connectPromise) {
        return connectPromise;
      }
      if (pc && ["connecting", "connected"].includes(pc.connectionState)) {
        return;
      }

      connectPromise = (async () => {
        remoteStream = new MediaStream();
        const config = {
          sdpSemantics: "unified-plan",
          bundlePolicy: "max-bundle",
          iceCandidatePoolSize: 2,
        };
        if ($("use-stun").checked) {
          config.iceServers = [{ urls: ["stun:stun.l.google.com:19302"] }];
        }

        pc = new RTCPeerConnection(config);
        pc.addEventListener("track", (event) => {
          const exists = remoteStream.getTracks().some((track) => track.id === event.track.id);
          if (!exists) {
            remoteStream.addTrack(event.track);
          }
          event.track.onunmute = () => {
            resumePlayer();
          };
          bindRemoteStream(true);
          resumePlayer();
        });

        pc.addEventListener("connectionstatechange", () => {
          setPcState(pc.connectionState);
          clearDisconnectStatusTimer();
          if (pc.connectionState === "connected") {
            setStatus("WebRTC connected", "ok");
            toggleButtons(true);
          } else if (pc.connectionState === "disconnected") {
            disconnectStatusTimer = setTimeout(() => {
              if (pc && pc.connectionState === "disconnected") {
                setStatus("WebRTC temporarily disconnected. Waiting for it to recover...", "warn");
              }
            }, DISCONNECTED_WARN_DELAY_MS);
            disconnectLostStatusTimer = setTimeout(() => {
              if (pc && pc.connectionState === "disconnected") {
                setStatus("WebRTC connection looks lost. Please reconnect if it does not recover.", "warn");
                toggleButtons(false);
              }
            }, DISCONNECTED_STATUS_GRACE_MS);
          } else if (["failed", "closed"].includes(pc.connectionState)) {
            setStatus("Connection closed", "warn");
            toggleButtons(false);
          }
        });

        setStatus("Establishing WebRTC connection...");
        setPcState("connecting");
        await negotiate();
        setStatus("Connection ready. You can now start the Gemini conversation.", "ok");
        toggleButtons(true);
        startPolling();
      })();

      try {
        await connectPromise;
      } finally {
        connectPromise = null;
      }
    }

    async function prepareAvatarConfig() {
      const formData = new FormData();
      formData.append("session_id", sessionId);
      formData.append("ckpt_dir", $("ckpt-dir").value.trim());
      formData.append("wav2vec_dir", $("wav2vec-dir").value.trim());
      formData.append("model_type", $("model-type").value);
      formData.append("seed", $("seed").value.trim());
      formData.append("use_face_crop", $("use-face-crop").checked ? "true" : "false");
      formData.append("cond_image_path", $("cond-image-path").value.trim());

      const condImageUpload = $("cond-image-upload").files[0];
      if (condImageUpload) {
        formData.append("cond_image_upload", condImageUpload);
      }

      const response = await fetch("/api/generate", {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "prepare avatar request failed");
      }
      setGenState(payload.status || "prepared");
      setSelectedAudio(payload.selected_audio || "-");
      setStatus("Avatar settings are ready. Waiting for Gemini to join.", "ok");
      return payload;
    }

    async function ensureCaptureContext() {
      if (!captureAudioContext) {
        captureAudioContext = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (captureAudioContext.state !== "running") {
        await captureAudioContext.resume();
      }
    }

    async function requestUserMedia(constraints) {
      if (navigator.mediaDevices && typeof navigator.mediaDevices.getUserMedia === "function") {
        return navigator.mediaDevices.getUserMedia(constraints);
      }

      const legacyGetUserMedia =
        navigator.getUserMedia ||
        navigator.webkitGetUserMedia ||
        navigator.mozGetUserMedia;

      if (typeof legacyGetUserMedia === "function") {
        return new Promise((resolve, reject) => {
          legacyGetUserMedia.call(navigator, constraints, resolve, reject);
        });
      }

      const secureHint = window.isSecureContext
        ? ""
        : " Open this page through HTTPS or localhost to allow microphone access.";
      throw new Error(`Microphone access is unavailable in this browser.${secureHint}`);
    }

    async function prepareMicAccess() {
      await ensureCaptureContext();
      if (micStream) {
        return;
      }

      try {
        micStream = await requestUserMedia({
          audio: {
            channelCount: 1,
            noiseSuppression: true,
            echoCancellation: true,
            autoGainControl: true,
          },
        });
      } catch (error) {
        const message = error?.message || "Failed to access the microphone.";
        setStatus(message, "warn");
        pushFeed(message, "warn");
        throw error;
      }

      micSource = captureAudioContext.createMediaStreamSource(micStream);
      micProcessor = captureAudioContext.createScriptProcessor(4096, 1, 1);
      micMuteNode = captureAudioContext.createGain();
      micMuteNode.gain.value = 0;
      micResampler = new PCMDownsampler(captureAudioContext.sampleRate, 16000);

      micProcessor.onaudioprocess = (event) => {
        if (!isGeminiConnected || !isMicRunning || !controlSocket || controlSocket.readyState !== WebSocket.OPEN) {
          return;
        }
        const input = event.inputBuffer.getChannelData(0);
        const downsampled = micResampler.process(new Float32Array(input));
        if (downsampled.length > 0) {
          controlSocket.send(JSON.stringify({
            type: "audio_chunk",
            data: int16ArrayToBase64(downsampled),
          }));
        }
      };

      micSource.connect(micProcessor);
      micProcessor.connect(micMuteNode);
      micMuteNode.connect(captureAudioContext.destination);
    }

    function activateMicStreaming() {
      if (!micStream) {
        return;
      }
      isMicRunning = true;
      pendingMicStart = false;
      setStatus("Gemini connected. The microphone is live and reply audio is now driving the avatar.", "ok");
      pushFeed("Microphone started. The browser is streaming PCM audio.", "ok");
      toggleButtons(true);
    }

    async function stopMic(sendAudioStreamEnd = true) {
      if (!isMicRunning && !micStream && !pendingMicStart) {
        return;
      }

      const shouldNotify = sendAudioStreamEnd && isGeminiConnected && controlSocket && controlSocket.readyState === WebSocket.OPEN;
      pendingMicStart = false;
      isMicRunning = false;
      if (shouldNotify) {
        controlSocket.send(JSON.stringify({ type: "audio_stream_end" }));
        pushFeed("Sent audio_stream_end.");
      }
      if (micProcessor) {
        micProcessor.disconnect();
        micProcessor.onaudioprocess = null;
      }
      if (micMuteNode) {
        micMuteNode.disconnect();
      }
      if (micSource) {
        micSource.disconnect();
      }
      if (micStream) {
        for (const track of micStream.getTracks()) {
          track.stop();
        }
      }
      micProcessor = null;
      micMuteNode = null;
      micSource = null;
      micStream = null;
      micResampler = null;
    }

    async function closeGemini(notifyBackend = true) {
      await stopMic(false);
      if (notifyBackend && controlSocket && controlSocket.readyState === WebSocket.OPEN) {
        controlSocket.send(JSON.stringify({ type: "close_session" }));
      }
      if (controlSocket) {
        controlSocket.onclose = null;
        controlSocket.close();
      }
      stopControlKeepalive();
      controlSocket = null;
      isGeminiConnected = false;
      resetRobotPanel();
      toggleButtons(Boolean(pc && pc.connectionState === "connected"));
    }

    async function startGeminiConversation() {
      if (!sessionId || !pc || !["new", "connecting", "connected"].includes(pc.connectionState)) {
        await startConnection();
      }
      resumePlayer();
      await prepareAvatarConfig();
      await closeGemini(false);
      clearPlayerNode();
      $("placeholder").style.display = "flex";
      await prepareMicAccess();
      pendingMicStart = true;
      setInputTranscript("");
      setOutputTranscript("");
      resetRobotPanel();
      setOutputPath("-");
      setSelectedAudio("gemini_live_reply");
      controlSocket = new WebSocket(serverWsUrl);
      pushFeed("Connecting to the Gemini control channel.");

        controlSocket.onopen = () => {
          pushFeed("Gemini control channel connected.", "ok");
          startControlKeepalive();
          controlSocket.send(JSON.stringify({
            type: "start_session",
            session_id: sessionId,
            api_key: $("api-key").value.trim(),
            model: $("gemini-model").value.trim(),
            voice_name: $("gemini-voice").value,
            system_instruction: $("system-prompt").value.trim(),
            temperature: $("gemini-temperature").value.trim(),
          }));
        };

      controlSocket.onmessage = async (event) => {
        const payload = JSON.parse(event.data);
        switch (payload.type) {
          case "session_ready":
            isGeminiConnected = true;
            setGenState("live_ready");
            setSelectedAudio(payload.model || "gemini_live_reply");
            pushFeed(`Gemini Live session established. Model: ${payload.model || "unknown"}`, "ok");
            bindRemoteStream(true);
            resumePlayer();
            if (pendingMicStart && micStream) {
              activateMicStreaming();
            }
            break;
          case "input_transcription":
            setInputTranscript(payload.text);
            markRobotPanelWaitingForCurrentTurn();
            break;
          case "output_transcription":
            setOutputTranscript(payload.text);
            markRobotPanelWaitingForCurrentTurn();
            break;
          case "turn_complete":
            pushFeed("Gemini finished this response. Continuing with the live WebRTC stream.", "ok");
            setStatus("This reply has been received and the live stream is staying open for the next turn.", "ok");
            currentTurnHasRobotAction = false;
            break;
          case "interrupted":
            pushFeed("Gemini was interrupted. Switching to a new turn.", "warn");
            setStatus("Gemini was interrupted. Switching to new content.", "warn");
            currentTurnHasRobotAction = false;
            break;
          case "status":
            pushFeed(payload.message || "Status update received.", payload.level === "ok" ? "ok" : payload.level === "warn" ? "warn" : "");
            break;
          case "robot_action": {
            updateRobotPanel(payload);
            const label = payload.label || payload.action || "robot action";
            if (payload.executed) {
              const mode = payload.simulated ? "simulated" : "executed";
              pushFeed(`Robot ${mode}: ${label}`, payload.simulated ? "" : "ok");
              setStatus(`Gemini is speaking and the robot is ${payload.simulated ? "simulating" : "performing"}: ${label}.`, "ok");
            } else if (payload.matched === false) {
              pushFeed(payload.reason || "No mapped robot action matched this turn.");
            } else if (payload.error) {
              pushFeed(`Robot action failed: ${payload.error}`, "warn");
              setStatus(`Robot action failed: ${payload.error}`, "warn");
            }
            break;
          }
          case "pong":
            break;
          case "error":
            pushFeed(payload.message || "Gemini session error.", "warn");
            setStatus(payload.message || "Gemini session error.", "warn");
            setGenState("error");
            break;
          case "session_closed":
            pushFeed(payload.message || "Gemini session closed.");
            await closeGemini(false);
            break;
          default:
            pushFeed(`Received unknown event: ${payload.type || "unknown"}`);
            break;
        }
      };

      controlSocket.onerror = () => {
        pushFeed("Gemini control channel error.", "warn");
        setStatus("Gemini control channel error.", "warn");
      };

      controlSocket.onclose = async () => {
        pushFeed("Gemini control channel closed.");
        await closeGemini(false);
      };
    }

    async function stopSession(silent = false) {
      if (!sessionId) {
        return;
      }
      await closeGemini(true);
      await fetch("/api/stop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      });
      if (!silent) {
        setStatus("Stop requested for the current Gemini-driven session.", "warn");
        setGenState("stopping");
      }
    }

    async function pollStatus() {
      if (!sessionId) {
        return;
      }
      const response = await fetch(`/api/status/${sessionId}`);
      if (!response.ok) {
        return;
      }
      const payload = await response.json();
      setGenState(payload.status || "idle");
      setOutputPath(payload.output_path || "-");
      setSelectedAudio(payload.selected_audio || "-");
      if (payload.input_transcript) {
        setInputTranscript(payload.input_transcript);
      }
      if (payload.output_transcript) {
        setOutputTranscript(payload.output_transcript);
      }
      if (payload.status === "running" || payload.status === "draining") {
        const statusText = payload.status === "draining"
          ? "The WebRTC stream is flushing the remaining synchronized audio and video for this turn."
          : "Generating synchronized audio and video and pushing them through WebRTC in real time.";
        setStatus(statusText, "ok");
        if (__ENABLE_LIVE_TRACK_PLAYBACK__) {
          resumePlayer();
        }
      } else if (payload.status === "saving") {
        setStatus("Saving the archive copy for this turn while the live WebRTC stream stays available.", "ok");
      } else if (payload.status === "live_ready" || payload.status === "prepared") {
        setStatus("Session ready. Waiting for Gemini's next reply.", "ok");
      } else if (payload.status === "completed") {
        setStatus("Gemini-driven video generation for this turn is complete.", "ok");
        if (__ENABLE_LIVE_TRACK_PLAYBACK__) {
          resumePlayer();
        }
      } else if (payload.status === "error") {
        setStatus(payload.error || "Inference failed", "warn");
      } else if (payload.status === "stopped") {
        setStatus("Session stopped", "warn");
      }
    }

    function startPolling() {
      if (pollHandle) {
        clearInterval(pollHandle);
      }
      pollHandle = setInterval(() => {
        pollStatus().catch((error) => console.error(error));
      }, 1500);
    }

    async function closePeer() {
      clearDisconnectStatusTimer();
      if (pollHandle) {
        clearInterval(pollHandle);
        pollHandle = null;
      }
      await closeGemini(false);

      clearPlayerNode();
      $("placeholder").style.display = "flex";

      if (pc) {
        pc.close();
        pc = null;
      }
      remoteStream = null;
      sessionId = null;
      connectPromise = null;
      $("session-id").textContent = "-";
      toggleButtons(false);
    }

    $("connect-btn").addEventListener("click", () => {
      startConnection().catch((error) => {
        console.error(error);
        setStatus(error.message || "Connection failed", "warn");
      });
    });

    $("cond-image-select").addEventListener("change", (event) => {
      if (event.target.value !== "__custom__") {
        $("cond-image-path").value = event.target.value;
      }
      updateReferencePreview();
    });

    $("cond-image-path").addEventListener("input", () => {
      syncReferencePreset();
      updateReferencePreview();
    });

    $("cond-image-upload").addEventListener("change", () => {
      syncReferencePreset();
      updateReferencePreview();
    });

    $("generate-btn").addEventListener("click", () => {
      startGeminiConversation().catch((error) => {
        console.error(error);
        setStatus(error.message || "Failed to start the Gemini conversation", "warn");
        setGenState("error");
      });
    });

    $("send-text-btn").addEventListener("click", () => {
      const text = $("text-input").value.trim();
      if (!text || !controlSocket || controlSocket.readyState !== WebSocket.OPEN) {
        return;
      }
      controlSocket.send(JSON.stringify({ type: "text", text }));
      pushFeed(`Sent text: ${text}`);
      $("text-input").value = "";
    });

    $("stop-btn").addEventListener("click", () => {
      stopSession().catch((error) => {
        console.error(error);
        setStatus(error.message || "Failed to stop the session", "warn");
      });
    });

    window.addEventListener("beforeunload", () => {
      closePeer();
    });

    pushFeed("Page ready. Connect WebRTC first, then start the Gemini conversation.");
    syncReferencePreset();
    updateReferencePreview();
    startConnection().catch((error) => {
      console.warn("Initial WebRTC preconnect failed:", error);
      setStatus("Page loaded. Click Connect WebRTC to retry.", "warn");
    });
  </script>
</body>
</html>
"""


AUDIO_PTIME = 0.020
VIDEO_CLOCK_RATE = 90000
VIDEO_QUEUE_MAXSIZE = 96
AUDIO_QUEUE_MAXSIZE = 96
# Keep enough queued media to absorb GPU chunk-generation jitter; a very small
# live queue causes audible gaps and lip-sync drift whenever one chunk arrives
# slightly later than real time.
MAX_LIVE_QUEUE_SECONDS = 0.75
TARGET_LIVE_QUEUE_SECONDS = 0.45
WEBRTC_DISCONNECT_GRACE_SECONDS = 12.0
ENABLE_LIVE_TRACK_PLAYBACK = True
SEGMENT_PLAYBACK_TARGET_SECONDS = 1.0
PLAYBACK_PREFILL_SECONDS = 0.55
PLAYBACK_MIN_BUFFER_SECONDS = 0.22
PLAYBACK_MAX_BUFFER_SECONDS = 1.10
DEFAULT_INFER_PARAMS = get_infer_params()
DEFAULT_SAMPLE_RATE = DEFAULT_INFER_PARAMS["sample_rate"]
DEFAULT_TGT_FPS = DEFAULT_INFER_PARAMS["tgt_fps"]
DEFAULT_CKPT_DIR = "models/SoulX-FlashHead-1_3B"
DEFAULT_WAV2VEC_DIR = "models/wav2vec2-base-960h"
DEFAULT_MODEL_TYPE = "lite"
DEFAULT_SEED = 9999
DEFAULT_REF_DIR = os.path.join("examples", "refs")
DEFAULT_REF_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def _list_default_ref_images():
    if not os.path.isdir(DEFAULT_REF_DIR):
        return []
    return [
        os.path.join(DEFAULT_REF_DIR, name).replace("\\", "/")
        for name in sorted(os.listdir(DEFAULT_REF_DIR))
        if os.path.isfile(os.path.join(DEFAULT_REF_DIR, name))
        and os.path.splitext(name)[1].lower() in DEFAULT_REF_IMAGE_EXTS
    ]


DEFAULT_REF_IMAGES = _list_default_ref_images()
DEFAULT_COND_IMAGE = (
    os.getenv("NOORACT_DEFAULT_REF_IMAGE", DEFAULT_REF_IMAGES[0] if DEFAULT_REF_IMAGES else "examples/girl.png")
    .replace("\\", "/")
)

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview")
GEMINI_VOICE_PRESETS = {
    "sadaltager": "Sadaltager",
    "female": "Vindemiatrix",
    "male": "Kore",
}
DEFAULT_GEMINI_VOICE = os.getenv("GEMINI_LIVE_VOICE", "Sadaltager")
DEFAULT_GEMINI_SYSTEM_PROMPT = os.getenv(
    "GEMINI_LIVE_SYSTEM_PROMPT",
    """
        You are the digital human voice assistant of NoorAct.
        Speak in a natural, friendly, and calm way.

        Use English by default.
        If the user speaks another language, switch to that language smoothly.

        Keep sentences short and simple.
        Speak at a slightly slow pace, with gentle pauses between sentences.
        Maintain a warm, steady, and clear tone.

        Avoid long explanations unless the user asks for more detail.
        Keep most replies around 10 words, and under 8 seconds.

        You are connected to a robot body that can perform safe social gestures,
        such as handshakes, waving, presentation gestures, hugs, and heart gestures.
        If the user asks for one of these actions, respond naturally and positively.
        Do not say that you are unable to do simple physical social gestures.
        When a gesture is requested, briefly acknowledge it in a natural way.
        For normal daily conversation, do not proactively suggest or imply body actions.
        Only use presentation-style gestures when the user clearly asks you to introduce,
        explain, present, or demonstrate something.

        Do not mention any AI models or system identities.
        You are NoorAct’s digital human assistant.
    """,
)
DEFAULT_GEMINI_TEMPERATURE = float(os.getenv("GEMINI_LIVE_TEMPERATURE", "0.6"))
UPLOAD_ROOT = os.path.join("gradio_results", "webrtc_uploads")
OUTPUT_ROOT = os.path.join("gradio_results", "webrtc_results")
EXPLICIT_ACTION_TRIGGER_DELAY_SECONDS = float(os.getenv("G1_EXPLICIT_ACTION_TRIGGER_DELAY_SECONDS", "0.0"))
DEFAULT_PRESENTATION_TRIGGER_DELAY_SECONDS = float(os.getenv("G1_PRESENTATION_TRIGGER_DELAY_SECONDS", "0.5"))
PRESENTATION_GESTURE_LOOP_GAP_SECONDS = float(os.getenv("G1_PRESENTATION_LOOP_GAP_SECONDS", "0.3"))
PRESENTATION_OUTPUT_IDLE_STOP_SECONDS = float(os.getenv("G1_PRESENTATION_OUTPUT_IDLE_STOP_SECONDS", "1.0"))
PRESENTATION_RELEASE_DELAY_SECONDS = float(os.getenv("G1_PRESENTATION_RELEASE_DELAY_SECONDS", "0.7"))
GEMINI_LIVE_MAX_RECONNECT_ATTEMPTS = int(os.getenv("GEMINI_LIVE_MAX_RECONNECT_ATTEMPTS", "3"))
GEMINI_LIVE_RECONNECT_DELAY_SECONDS = float(os.getenv("GEMINI_LIVE_RECONNECT_DELAY_SECONDS", "1.5"))

pipeline = None
loaded_ckpt_dir = None
loaded_wav2vec_dir = None
loaded_model_type = None

PIPELINE_LOCK = threading.Lock()
INFERENCE_LOCK = threading.Lock()
SESSIONS: Dict[str, "StreamSession"] = {}
PCS = set()


def _write_frames_to_mp4(frames_list, video_path, fps):
    """将帧列表写入 MP4（仅视频轨）。"""
    os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
    with imageio.get_writer(
        video_path,
        format="mp4",
        mode="I",
        fps=fps,
        codec="h264",
        ffmpeg_params=["-bf", "0"],
    ) as writer:
        for frames in frames_list:
            frames_np = np.asarray(frames, dtype=np.uint8)
            for frame in frames_np:
                writer.append_data(frame)
    return video_path


def save_video_with_audio(frames_list, video_path, audio_path, fps):
    """写入完整视频并混入完整音频。"""
    temp_path = video_path.replace(".mp4", "_temp.mp4")
    _write_frames_to_mp4(frames_list, temp_path, fps)
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            temp_path,
            "-i",
            audio_path,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            video_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return video_path


def _maybe_sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _coerce_temperature(value) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_model_name(model: Optional[str]) -> str:
    text = (model or "").strip() or DEFAULT_GEMINI_MODEL
    if text.startswith("models/"):
        return text.replace("models/", "", 1)
    return text


def _is_retryable_live_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "keepalive ping timeout" in message
        or "abnormal closure" in message
        or "connectionclosederror" in message
        or "1006" in message
        or "1011" in message
    )


def _resolve_gemini_voice_name(value: Optional[str]) -> str:
    text = (value or "").strip()
    if not text:
        return DEFAULT_GEMINI_VOICE

    preset_match = GEMINI_VOICE_PRESETS.get(text.lower())
    if preset_match:
        return preset_match

    for voice_name in GEMINI_VOICE_PRESETS.values():
        if text.lower() == voice_name.lower():
            return voice_name

    return DEFAULT_GEMINI_VOICE


def _build_gemini_voice_options() -> str:
    default_value = _resolve_gemini_voice_name(DEFAULT_GEMINI_VOICE)
    labels = {
        "sadaltager": "Sadaltager",
        "female": "Female",
        "male": "Male",
    }
    options = []
    for key, voice_name in GEMINI_VOICE_PRESETS.items():
        selected = " selected" if voice_name == default_value else ""
        options.append(f'<option value="{voice_name}"{selected}>{labels.get(key, key.title())}</option>')
    return "\n".join(options)


def _build_live_config(
    *,
    system_instruction: str,
    voice_name: str,
    temperature: Optional[float],
):
    if GENAI_IMPORT_ERROR is not None:
        raise RuntimeError(
            "The `google-genai` package is not installed in this environment yet. Please run `pip install google-genai` first."
        )

    config_kwargs = {
        "response_modalities": [types.Modality.AUDIO],
        "input_audio_transcription": types.AudioTranscriptionConfig(),
        "output_audio_transcription": types.AudioTranscriptionConfig(),
        "realtime_input_config": types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_HIGH,
                end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_HIGH,
                prefix_padding_ms=300,
                silence_duration_ms=700,
            ),
            turn_coverage=types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY,
        ),
    }

    if system_instruction:
        config_kwargs["system_instruction"] = types.Content(
            parts=[types.Part(text=system_instruction)]
        )

    if voice_name:
        config_kwargs["speech_config"] = types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice_name,
                )
            )
        )

    if temperature is not None:
        config_kwargs["temperature"] = temperature

    return types.LiveConnectConfig(**config_kwargs)


def _decode_pcm16_bytes(audio_bytes: bytes) -> np.ndarray:
    if not audio_bytes:
        return np.array([], dtype=np.float32)
    return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def _extract_sample_rate_from_mime_type(mime_type: str, default: int = 24000) -> int:
    marker = "rate="
    if not mime_type or marker not in mime_type:
        return default
    try:
        return int(mime_type.split(marker, 1)[1].split(";", 1)[0].strip())
    except (ValueError, TypeError):
        return default


def _save_audio_array_to_wav(audio_array: np.ndarray, wav_path: str, sample_rate: int) -> str:
    os.makedirs(os.path.dirname(wav_path) or ".", exist_ok=True)
    samples = (np.clip(audio_array, -1.0, 1.0) * 32767).astype(np.int16)
    with open(wav_path, "wb") as _:
        pass
    import wave

    with wave.open(wav_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())
    return wav_path


def _save_turn_media_async(
    session: "StreamSession",
    frame_segments,
    audio_array: np.ndarray,
    sample_rate: int,
    fps: int,
    turn_index: int,
):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    wav_path = os.path.join(
        OUTPUT_ROOT,
        f"webrtc_res_{timestamp}_{session.session_id[:8]}_turn_{turn_index:03d}.wav",
    )
    video_path = os.path.join(
        OUTPUT_ROOT,
        f"webrtc_res_{timestamp}_{session.session_id[:8]}_turn_{turn_index:03d}.mp4",
    )

    def worker():
        try:
            _save_audio_array_to_wav(audio_array, wav_path, sample_rate=sample_rate)
            save_video_with_audio(frame_segments, video_path, wav_path, fps=fps)
            session.output_path = os.path.abspath(video_path)
        except Exception as exc:
            logger.warning(
                "Failed to save Gemini-driven MP4 for session {}: {}",
                session.session_id,
                exc,
            )

    thread = threading.Thread(
        target=worker,
        daemon=True,
        name=f"soulx-save-{session.session_id[:8]}-{turn_index:03d}",
    )
    thread.start()

def _output_path_to_url(output_path: str) -> str:
    if not output_path:
        return ""
    abs_output_root = os.path.abspath(OUTPUT_ROOT)
    abs_output_path = os.path.abspath(output_path)
    if not abs_output_path.startswith(abs_output_root):
        return ""
    rel_path = os.path.relpath(abs_output_path, abs_output_root).replace("\\", "/")
    return f"/generated/{rel_path}"


async def _safe_send_json(ws: web.WebSocketResponse, payload: dict) -> None:
    if not ws.closed:
        await ws.send_json(payload)


@dataclass
class AvatarGenerationConfig:
    ckpt_dir: str
    wav2vec_dir: str
    model_type: str
    cond_image: str
    seed: int
    use_face_crop: bool


@dataclass
class GeminiSessionConfig:
    api_key: str
    model: str
    voice_name: str
    system_instruction: str
    temperature: Optional[float]


@dataclass
class SharedClock:
    start_time: Optional[float] = None


class QueueStreamTrack(MediaStreamTrack):
    def __init__(self, kind: str, fps: int, sample_rate: int, clock: SharedClock, queue_maxsize: int):
        super().__init__()
        self.kind = kind
        self.fps = fps
        self.sample_rate = sample_rate
        self.clock = clock
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=queue_maxsize)
        self._timestamp: Optional[int] = None
        self._start: Optional[float] = None
        self._last_video_array: Optional[np.ndarray] = None

    def _timing_params(self) -> Tuple[int, fractions.Fraction, int]:
        if self.kind == "video":
            return (
                int((1 / self.fps) * VIDEO_CLOCK_RATE),
                fractions.Fraction(1, VIDEO_CLOCK_RATE),
                VIDEO_CLOCK_RATE,
            )
        return (
            int(AUDIO_PTIME * self.sample_rate),
            fractions.Fraction(1, self.sample_rate),
            self.sample_rate,
        )

    def _frame_duration_seconds(self) -> float:
        if self.kind == "video":
            return 1.0 / max(float(self.fps), 1.0)
        return AUDIO_PTIME

    async def next_timestamp(self, *, reset_clock: bool = False) -> Tuple[int, fractions.Fraction]:
        if self.readyState != "live":
            raise MediaStreamError

        step, time_base, clock_rate = self._timing_params()

        if self._timestamp is None:
            if self.clock.start_time is None:
                self.clock.start_time = time.time()
            self._start = self.clock.start_time
            self._timestamp = 0
        else:
            if reset_clock:
                # When the source starves for a while, anchor the next media
                # frame to "now" instead of trying to catch back up to an old
                # wall-clock schedule. This keeps audio continuous and in sync
                # with video instead of emitting silence/bursts.
                self._start = time.time() - ((self._timestamp + step) / clock_rate)
            self._timestamp += step
            wait = self._start + (self._timestamp / clock_rate) - time.time()
            if wait > 0:
                await asyncio.sleep(wait)

        return self._timestamp, time_base

    async def recv(self):
        waited_at = time.time()
        frame = await self._queue.get()
        waited_seconds = time.time() - waited_at

        if frame is None:
            self.stop()
            raise MediaStreamError

        if self.kind == "video":
            self._last_video_array = frame.to_ndarray(format="rgb24")

        starvation_gap = waited_seconds > (self._frame_duration_seconds() * 1.5)
        pts, time_base = await self.next_timestamp(reset_clock=starvation_gap)
        frame.pts = pts
        frame.time_base = time_base
        return frame

    def reset_timing(self):
        self._timestamp = None
        self._start = None

    def resync_timing(self):
        if self._timestamp is None:
            self._start = None
            return

        if self.kind == "video":
            clock_rate = VIDEO_CLOCK_RATE
        else:
            clock_rate = self.sample_rate
        self._start = time.time() - (self._timestamp / clock_rate)

    def clear_queue(self):
        self._last_video_array = None
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break


class StreamSession:
    def __init__(self, session_id: str, pc: RTCPeerConnection, loop: asyncio.AbstractEventLoop):
        self.session_id = session_id
        self.pc = pc
        self.loop = loop
        self.clock = SharedClock()
        self.video_track = QueueStreamTrack(
            kind="video",
            fps=DEFAULT_TGT_FPS,
            sample_rate=DEFAULT_SAMPLE_RATE,
            clock=self.clock,
            queue_maxsize=VIDEO_QUEUE_MAXSIZE,
        )
        self.audio_track = QueueStreamTrack(
            kind="audio",
            fps=DEFAULT_TGT_FPS,
            sample_rate=DEFAULT_SAMPLE_RATE,
            clock=self.clock,
            queue_maxsize=AUDIO_QUEUE_MAXSIZE,
        )
        self.status = "idle"
        self.error_message = ""
        self.output_path = ""
        self.selected_audio = ""
        self.avatar_config: Optional[AvatarGenerationConfig] = None
        self.worker: Optional[threading.Thread] = None
        self.stop_event: Optional[threading.Event] = None
        self.request_id = 0
        self.stream_epoch = 0
        self.control_lock = threading.Lock()
        self.closed = False
        self.reply_queue: "queue.Queue[dict]" = queue.Queue()
        self.turn_index = 0
        self.input_transcript = ""
        self.output_transcript = ""
        self.input_transcript_sealed = False
        self.output_transcript_sealed = False
        self.audio_packet_remainder = np.array([], dtype=np.float32)
        self.video_frame_remainder: Optional[np.ndarray] = None
        self.last_audio_publish_at = 0.0

    def prepare_avatar(self, config: AvatarGenerationConfig):
        with self.control_lock:
            if self.closed:
                raise RuntimeError("Session has been closed.")
            self.avatar_config = config
            self.error_message = ""
            self.output_path = ""
            if self.status in {"idle", "completed", "stopped", "error"}:
                self.status = "prepared"

    def start_live_generation(self):
        with self.control_lock:
            if self.closed:
                raise RuntimeError("Session has been closed.")
            if self.avatar_config is None:
                raise RuntimeError("Avatar config has not been prepared.")

            self.request_id += 1
            request_id = self.request_id
            active_worker = self.worker if self.worker and self.worker.is_alive() else None
            active_stop_event = self.stop_event

            self.status = "switching" if active_worker else "starting"
            self.error_message = ""
            self.output_path = ""
            self.selected_audio = "Gemini Live reply"
            self.reply_queue = queue.Queue()
            self.turn_index = 0
            self.input_transcript = ""
            self.output_transcript = ""
            self.input_transcript_sealed = False
            self.output_transcript_sealed = False
            self.audio_packet_remainder = np.array([], dtype=np.float32)
            self.video_frame_remainder = None

            if active_stop_event is not None:
                active_stop_event.set()
                try:
                    self.reply_queue.put_nowait({"kind": "stop"})
                except Exception:
                    pass

            stream_epoch = self._advance_stream_epoch()
            self._reset_tracks()

            if active_worker:
                switch_thread = threading.Thread(
                    target=self._wait_and_start_live_generation,
                    args=(active_worker, request_id, stream_epoch),
                    daemon=True,
                    name=f"soulx-switch-{self.session_id[:8]}",
                )
                switch_thread.start()
                return "switching"

            self._launch_live_generation_worker(request_id, stream_epoch)
            return "starting"

    def _wait_and_start_live_generation(
        self,
        worker: threading.Thread,
        request_id: int,
        stream_epoch: int,
    ):
        worker.join()
        with self.control_lock:
            if self.closed or request_id != self.request_id:
                return
            self._launch_live_generation_worker(request_id, stream_epoch)

    def _launch_live_generation_worker(self, request_id: int, stream_epoch: int):
        self.clock.start_time = None
        stop_event = threading.Event()
        self.stop_event = stop_event
        self.status = "starting"
        self.worker = threading.Thread(
            target=self._run_live_generation,
            args=(request_id, stream_epoch, stop_event),
            daemon=True,
            name=f"soulx-webrtc-{self.session_id[:8]}",
        )
        self.worker.start()

    def _advance_stream_epoch(self) -> int:
        self.stream_epoch += 1
        return self.stream_epoch

    def _enqueue_from_thread(self, track: QueueStreamTrack, frame, stream_epoch: int):
        if self.closed:
            return
        self.loop.call_soon_threadsafe(self._enqueue_frame_nowait, track, frame, stream_epoch)

    async def _enqueue_frame_async(self, track: QueueStreamTrack, frame, stream_epoch: int) -> bool:
        if self.closed or stream_epoch != self.stream_epoch or track.readyState != "live":
            return False
        await track._queue.put(frame)
        return True

    def _enqueue_from_thread_blocking(
        self,
        track: QueueStreamTrack,
        frame,
        stream_epoch: int,
        stop_event: threading.Event,
    ) -> bool:
        if self.closed or stop_event.is_set():
            return False

        future = asyncio.run_coroutine_threadsafe(
            self._enqueue_frame_async(track, frame, stream_epoch),
            self.loop,
        )
        try:
            while True:
                try:
                    return bool(future.result(timeout=0.1))
                except concurrent.futures.TimeoutError:
                    if stop_event.is_set() or self.closed or stream_epoch != self.stream_epoch:
                        future.cancel()
                        return False
        except Exception as exc:
            logger.debug("Blocking enqueue failed for session {}: {}", self.session_id, exc)
            return False

    def _enqueue_frame_nowait(self, track: QueueStreamTrack, frame, stream_epoch: int):
        if stream_epoch != self.stream_epoch:
            return
        if track.readyState != "live":
            return

        if track._queue.full():
            try:
                track._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass

        try:
            track._queue.put_nowait(frame)
        except asyncio.QueueFull:
            pass

    @staticmethod
    def _close_track_queue(track: QueueStreamTrack):
        if track._queue.full():
            try:
                track._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            track._queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

    def _reset_track_state_in_loop(self):
        for track in (self.video_track, self.audio_track):
            track.clear_queue()
            track.resync_timing()

    def _reset_tracks(self):
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is self.loop:
            self._reset_track_state_in_loop()
            return

        done = threading.Event()

        def reset_wrapper():
            self._reset_track_state_in_loop()
            done.set()

        self.loop.call_soon_threadsafe(reset_wrapper)
        if not done.wait(timeout=1.0):
            logger.warning(
                "Track reset is still pending for session {}, continuing asynchronously.",
                self.session_id,
            )

    def _finish_tracks_from_thread(self):
        for track in (self.video_track, self.audio_track):
            try:
                self.loop.call_soon_threadsafe(self._close_track_queue, track)
            except Exception as exc:
                logger.debug(f"Failed to close track for session {self.session_id}: {exc}")

    def stop_generation(self):
        with self.control_lock:
            if self.stop_event is not None:
                self.stop_event.set()
            try:
                self.reply_queue.put_nowait({"kind": "stop"})
            except Exception:
                pass
            self.audio_packet_remainder = np.array([], dtype=np.float32)
            self.video_frame_remainder = None
            self._advance_stream_epoch()
            self._reset_tracks()
            self.status = "stopped"
            self.last_audio_publish_at = 0.0

    def get_public_status(self) -> str:
        video_pending = self.video_track._queue.qsize()
        audio_pending = self.audio_track._queue.qsize()
        if self.status == "completed" and (video_pending > 0 or audio_pending > 0):
            return "draining"
        return self.status

    def enqueue_gemini_audio(self, audio_bytes: bytes, mime_type: str):
        if self.closed or not audio_bytes:
            return
        self.reply_queue.put(
            {
                "kind": "audio",
                "audio_bytes": audio_bytes,
                "mime_type": mime_type,
            }
        )

    def mark_turn_complete(self):
        if not self.closed:
            self.input_transcript_sealed = True
            self.output_transcript_sealed = True
            self.reply_queue.put({"kind": "turn_complete"})

    def mark_interrupted(self):
        if not self.closed:
            self.input_transcript_sealed = True
            self.output_transcript_sealed = True
            self.reply_queue.put({"kind": "interrupted"})

    @staticmethod
    def _merge_streamed_transcript(current: str, incoming: str) -> str:
        current = (current or "").strip()
        incoming = (incoming or "").strip()
        if not incoming:
            return current
        if not current:
            return incoming
        if incoming.startswith(current):
            return incoming
        if current.startswith(incoming):
            return current

        max_overlap = min(len(current), len(incoming))
        for size in range(max_overlap, 0, -1):
            if current.endswith(incoming[:size]):
                return current + incoming[size:]

        return f"{current} {incoming}".strip()

    def update_input_transcript(self, text: str):
        if self.input_transcript_sealed:
            self.input_transcript = (text or "").strip()
            self.input_transcript_sealed = False
            return
        self.input_transcript = self._merge_streamed_transcript(self.input_transcript, text)

    def update_output_transcript(self, text: str):
        if self.output_transcript_sealed:
            self.output_transcript = (text or "").strip()
            self.output_transcript_sealed = False
            return
        self.output_transcript = self._merge_streamed_transcript(self.output_transcript, text)

    @staticmethod
    def _align_frames_to_audio(
        frames: np.ndarray,
        sample_count: int,
        sample_rate: int,
        fps: int,
    ) -> np.ndarray:
        frames_np = np.asarray(frames, dtype=np.uint8)
        if frames_np.ndim == 0 or sample_count <= 0 or sample_rate <= 0 or fps <= 0:
            return frames_np

        expected_frames = max(int(round(sample_count * fps / sample_rate)), 1)
        current_frames = frames_np.shape[0]
        if current_frames > expected_frames:
            return frames_np[:expected_frames]
        if current_frames < expected_frames and current_frames > 0:
            tail = np.repeat(frames_np[-1:], expected_frames - current_frames, axis=0)
            return np.concatenate([frames_np, tail], axis=0)
        return frames_np

    def _align_live_frames_to_emitted_audio(
        self,
        frames: np.ndarray,
        emitted_sample_count: int,
        sample_rate: int,
        fps: int,
    ) -> np.ndarray:
        frames_np = np.asarray(frames, dtype=np.uint8)
        if frames_np.ndim == 0:
            return frames_np

        if self.video_frame_remainder is not None and self.video_frame_remainder.size > 0:
            frames_np = np.concatenate([self.video_frame_remainder, frames_np], axis=0)
            self.video_frame_remainder = None

        if emitted_sample_count <= 0 or sample_rate <= 0 or fps <= 0:
            self.video_frame_remainder = frames_np if frames_np.shape[0] > 0 else None
            return frames_np[:0]

        target_frame_count = max(int(round(emitted_sample_count * fps / sample_rate)), 0)
        current_frame_count = frames_np.shape[0]
        if current_frame_count < target_frame_count and current_frame_count > 0:
            tail = np.repeat(frames_np[-1:], target_frame_count - current_frame_count, axis=0)
            frames_np = np.concatenate([frames_np, tail], axis=0)
            current_frame_count = frames_np.shape[0]

        emit_count = min(target_frame_count, current_frame_count)
        emitted_frames = frames_np[:emit_count]
        remainder = frames_np[emit_count:]
        self.video_frame_remainder = remainder if remainder.shape[0] > 0 else None
        return emitted_frames

    def _packetize_audio_samples(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        flush: bool = False,
    ) -> list[np.ndarray]:
        packet_size = int(sample_rate * AUDIO_PTIME)
        if packet_size <= 0:
            return []

        samples = np.asarray(audio_array, dtype=np.float32)
        if self.audio_packet_remainder.size > 0:
            samples = np.concatenate([self.audio_packet_remainder, samples])

        if samples.size == 0 and not flush:
            return []

        emit_len = (samples.shape[0] // packet_size) * packet_size
        emit_samples = samples[:emit_len]
        self.audio_packet_remainder = samples[emit_len:]
        packets = [
            emit_samples[start:start + packet_size].astype(np.float32, copy=True)
            for start in range(0, emit_samples.shape[0], packet_size)
        ]

        if flush and self.audio_packet_remainder.size > 0:
            chunk = self.audio_packet_remainder
            self.audio_packet_remainder = np.array([], dtype=np.float32)
            if chunk.shape[0] < packet_size:
                chunk = np.concatenate(
                    [chunk, np.zeros(packet_size - chunk.shape[0], dtype=np.float32)]
                )
            packets.append(chunk.astype(np.float32, copy=True))
        return packets

    def _publish_audio_packet(
        self,
        packet: np.ndarray,
        sample_rate: int,
        stop_event: threading.Event,
        stream_epoch: int,
    ) -> bool:
        if stop_event.is_set() or self.closed:
            return False
        chunk = (np.clip(packet, -1.0, 1.0) * 32767).astype(np.int16)
        audio_frame = AudioFrame(format="s16", layout="mono", samples=chunk.shape[0])
        audio_frame.planes[0].update(chunk.tobytes())
        audio_frame.sample_rate = sample_rate
        queued = self._enqueue_from_thread_blocking(
            self.audio_track,
            audio_frame,
            stream_epoch=stream_epoch,
            stop_event=stop_event,
        )
        if queued:
            self.last_audio_publish_at = time.monotonic()
        return queued

    def _publish_video_frame(
        self,
        frame: np.ndarray,
        stop_event: threading.Event,
        stream_epoch: int,
    ) -> bool:
        if stop_event.is_set() or self.closed:
            return False
        video_frame = VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")
        return self._enqueue_from_thread_blocking(
            self.video_track,
            video_frame,
            stream_epoch=stream_epoch,
            stop_event=stop_event,
        )

    def _publish_chunk_media(
        self,
        frames: np.ndarray,
        audio_array: np.ndarray,
        sample_rate: int,
        stop_event: threading.Event,
        stream_epoch: int,
        flush_audio: bool = False,
    ):
        if stop_event.is_set() or self.closed:
            return

        frames_np = np.asarray(frames, dtype=np.uint8)
        audio_packets = self._packetize_audio_samples(
            audio_array,
            sample_rate=sample_rate,
            flush=flush_audio,
        )
        emitted_sample_count = sum(packet.shape[0] for packet in audio_packets)
        frames_np = self._align_live_frames_to_emitted_audio(
            frames_np,
            emitted_sample_count=emitted_sample_count,
            sample_rate=sample_rate,
            fps=max(int(round(self.video_track.fps)), 1),
        )

        audio_index = 0
        video_index = 0
        next_audio_time = AUDIO_PTIME if audio_packets else float("inf")
        frame_interval = 1.0 / max(float(self.video_track.fps), 1.0)
        next_video_time = frame_interval if frames_np.shape[0] > 0 else float("inf")

        while audio_index < len(audio_packets) or video_index < frames_np.shape[0]:
            self._wait_for_live_queue_budget(stop_event)
            if stop_event.is_set() or self.closed:
                return

            send_audio = next_audio_time <= next_video_time
            if video_index >= frames_np.shape[0]:
                send_audio = True
            elif audio_index >= len(audio_packets):
                send_audio = False

            if send_audio:
                if not self._publish_audio_packet(
                    audio_packets[audio_index],
                    sample_rate=sample_rate,
                    stop_event=stop_event,
                    stream_epoch=stream_epoch,
                ):
                    return
                audio_index += 1
                next_audio_time = (audio_index + 1) * AUDIO_PTIME if audio_index < len(audio_packets) else float("inf")
            else:
                if not self._publish_video_frame(
                    frames_np[video_index],
                    stop_event=stop_event,
                    stream_epoch=stream_epoch,
                ):
                    return
                video_index += 1
                next_video_time = (video_index + 1) * frame_interval if video_index < frames_np.shape[0] else float("inf")

    def _queued_media_seconds(self) -> float:
        audio_seconds = self.audio_track._queue.qsize() * AUDIO_PTIME
        fps = max(float(self.video_track.fps), 1.0)
        video_seconds = self.video_track._queue.qsize() / fps
        return max(audio_seconds, video_seconds)

    def is_audio_playback_recent(self, idle_seconds: float) -> bool:
        if self.audio_track._queue.qsize() > 0:
            return True
        if self.last_audio_publish_at <= 0.0:
            return False
        return (time.monotonic() - self.last_audio_publish_at) <= idle_seconds

    def _wait_for_live_queue_budget(
        self,
        stop_event: threading.Event,
        target_seconds: float = TARGET_LIVE_QUEUE_SECONDS,
        max_seconds: float = MAX_LIVE_QUEUE_SECONDS,
    ) -> None:
        if not ENABLE_LIVE_TRACK_PLAYBACK:
            return

        # Keep only a small amount of queued media ahead of wall clock time.
        # If generation is faster than real time, this prevents the WebRTC
        # tracks from accumulating large latency by buffering too far ahead.
        while not stop_event.is_set() and not self.closed:
            queued_seconds = self._queued_media_seconds()
            if queued_seconds <= max_seconds:
                break
            time.sleep(min(0.01, max(queued_seconds - target_seconds, 0.01) * 0.25))

    def _run_live_generation(
        self,
        request_id: int,
        stream_epoch: int,
        stop_event: threading.Event,
    ):
        acquired = INFERENCE_LOCK.acquire(blocking=False)
        if not acquired:
            self.status = "error"
            self.error_message = "Another generation task is already running on this single-GPU server."
            return

        assert self.avatar_config is not None
        config = self.avatar_config
        turn_audio_segments = []
        turn_frame_segments = []
        try:
            self.status = "loading"
            current_pipeline = _ensure_pipeline_loaded(
                ckpt_dir=config.ckpt_dir,
                wav2vec_dir=config.wav2vec_dir,
                model_type=config.model_type,
            )

            self.status = "preparing"
            get_base_data(
                current_pipeline,
                cond_image_path_or_dir=config.cond_image,
                base_seed=config.seed,
                use_face_crop=config.use_face_crop,
            )

            infer_params = get_infer_params()
            sample_rate = infer_params["sample_rate"]
            tgt_fps = infer_params["tgt_fps"]
            cached_audio_duration = infer_params["cached_audio_duration"]
            frame_num = infer_params["frame_num"]
            motion_frames_num = infer_params["motion_frames_num"]
            slice_len = frame_num - motion_frames_num
            speech_slice_len = slice_len * sample_rate // tgt_fps

            self.video_track.fps = tgt_fps
            self.audio_track.sample_rate = sample_rate
            if speech_slice_len <= 0:
                raise RuntimeError("Invalid slice length computed from infer params.")

            cached_audio_length_sum = sample_rate * cached_audio_duration
            audio_end_idx = cached_audio_duration * tgt_fps
            audio_start_idx = audio_end_idx - frame_num
            pending_samples = np.array([], dtype=np.float32)
            audio_dq = deque([0.0] * cached_audio_length_sum, maxlen=cached_audio_length_sum)
            playback_buffer = deque()
            playback_condition = threading.Condition()
            chunk_idx = 0
            turn_active = False
            playback_started = False
            flush_requested = False
            generation_cost_ema = 0.0
            chunk_audio_seconds_ema = 0.0

            def reset_turn_state():
                nonlocal pending_samples, audio_dq, turn_audio_segments, turn_frame_segments, playback_buffer, turn_active, playback_started, flush_requested
                pending_samples = np.array([], dtype=np.float32)
                audio_dq = deque([0.0] * cached_audio_length_sum, maxlen=cached_audio_length_sum)
                turn_audio_segments = []
                turn_frame_segments = []
                with playback_condition:
                    playback_buffer.clear()
                    flush_requested = False
                    playback_started = False
                    playback_condition.notify_all()
                self.audio_packet_remainder = np.array([], dtype=np.float32)
                self.video_frame_remainder = None
                turn_active = False

            def buffered_audio_seconds_locked() -> float:
                if not playback_buffer:
                    return 0.0
                return sum(audio.shape[0] for _, audio, _ in playback_buffer) / float(sample_rate)

            def current_required_buffer_seconds() -> float:
                extra = max(generation_cost_ema - chunk_audio_seconds_ema, 0.0)
                if not playback_started:
                    return min(
                        PLAYBACK_MAX_BUFFER_SECONDS,
                        max(PLAYBACK_PREFILL_SECONDS, generation_cost_ema + 0.15),
                    )
                return min(
                    PLAYBACK_MAX_BUFFER_SECONDS,
                    max(PLAYBACK_MIN_BUFFER_SECONDS, extra + 0.20),
                )

            def wait_for_playback_drain(timeout: float = 3.0):
                deadline = time.time() + timeout
                with playback_condition:
                    while (playback_buffer or flush_requested) and time.time() < deadline:
                        playback_condition.wait(timeout=0.05)

            def playback_worker_loop():
                nonlocal playback_started, flush_requested
                while not stop_event.is_set() and not self.closed:
                    chunk_to_play = None
                    flush_audio = False
                    with playback_condition:
                        while not stop_event.is_set() and not self.closed:
                            buffered_seconds = buffered_audio_seconds_locked()
                            required_buffer = current_required_buffer_seconds()

                            if playback_buffer and (flush_requested or buffered_seconds > required_buffer):
                                buffered_frames, buffered_audio, buffered_stream_epoch = playback_buffer.popleft()
                                chunk_to_play = (buffered_frames, buffered_audio, buffered_stream_epoch)
                                flush_audio = flush_requested and not playback_buffer
                                playback_started = True
                                if flush_audio:
                                    flush_requested = False
                                    playback_started = False
                                break

                            if not playback_buffer and flush_requested:
                                flush_requested = False
                                playback_started = False
                                playback_condition.notify_all()

                            playback_condition.wait(timeout=0.05)

                    if chunk_to_play is None:
                        continue

                    buffered_frames, buffered_audio, buffered_stream_epoch = chunk_to_play
                    self._publish_chunk_media(
                        buffered_frames,
                        buffered_audio,
                        sample_rate=sample_rate,
                        stop_event=stop_event,
                        stream_epoch=buffered_stream_epoch,
                        flush_audio=flush_audio,
                    )

                    with playback_condition:
                        if not playback_buffer and not flush_requested:
                            playback_condition.notify_all()

            playback_thread = None
            if ENABLE_LIVE_TRACK_PLAYBACK:
                playback_thread = threading.Thread(
                    target=playback_worker_loop,
                    daemon=True,
                    name=f"soulx-playback-{self.session_id[:8]}",
                )
                playback_thread.start()

            def process_chunk(speech_chunk: np.ndarray):
                nonlocal chunk_idx, turn_active, generation_cost_ema, chunk_audio_seconds_ema
                if stop_event.is_set() or self.closed:
                    return
                if not turn_active:
                    turn_active = True
                    self.turn_index += 1
                    self.selected_audio = f"gemini_turn_{self.turn_index:03d}"
                audio_dq.extend(speech_chunk.tolist())
                audio_array = np.array(audio_dq, dtype=np.float32)
                audio_embedding = get_audio_embedding(
                    current_pipeline, audio_array, audio_start_idx, audio_end_idx
                )

                _maybe_sync_cuda()
                started_at = time.time()
                video = run_pipeline(current_pipeline, audio_embedding)
                video = video[motion_frames_num:]
                _maybe_sync_cuda()

                frames_np = video.cpu().numpy().astype(np.uint8)
                aligned_frames = self._align_frames_to_audio(
                    frames_np,
                    sample_count=speech_chunk.shape[0],
                    sample_rate=sample_rate,
                    fps=tgt_fps,
                )
                playback_audio = speech_chunk.astype(np.float32, copy=True)
                turn_frame_segments.append(aligned_frames)
                turn_audio_segments.append(playback_audio.copy())
                generation_cost = time.time() - started_at
                chunk_audio_seconds = playback_audio.shape[0] / float(sample_rate)
                generation_cost_ema = (
                    generation_cost if generation_cost_ema <= 0.0
                    else generation_cost_ema * 0.75 + generation_cost * 0.25
                )
                chunk_audio_seconds_ema = (
                    chunk_audio_seconds if chunk_audio_seconds_ema <= 0.0
                    else chunk_audio_seconds_ema * 0.75 + chunk_audio_seconds * 0.25
                )
                if ENABLE_LIVE_TRACK_PLAYBACK:
                    with playback_condition:
                        playback_buffer.append((aligned_frames, playback_audio, stream_epoch))
                        playback_condition.notify_all()
                logger.info(
                    "session={} turn={} chunk={} frames={} chunk_audio_samples={} buffered_seconds={:.2f} required_buffer={:.2f} live_queue={:.2f}s cost={:.2f}s",
                    self.session_id,
                    self.turn_index,
                    chunk_idx,
                    aligned_frames.shape[0],
                    playback_audio.shape[0],
                    buffered_audio_seconds_locked(),
                    current_required_buffer_seconds(),
                    self._queued_media_seconds(),
                    generation_cost,
                )
                chunk_idx += 1

            def finalize_turn():
                nonlocal pending_samples, stream_epoch, flush_requested
                if pending_samples.size > 0:
                    pad_length = speech_slice_len - pending_samples.size
                    final_chunk = np.concatenate(
                        [pending_samples, np.zeros(max(pad_length, 0), dtype=np.float32)]
                    )[:speech_slice_len]
                    pending_samples = np.array([], dtype=np.float32)
                    process_chunk(final_chunk)

                if not turn_frame_segments:
                    if self.status not in {"stopped", "error"}:
                        self.status = "live_ready"
                    reset_turn_state()
                    return

                if ENABLE_LIVE_TRACK_PLAYBACK:
                    with playback_condition:
                        flush_requested = True
                        playback_condition.notify_all()

                    wait_for_playback_drain()

                final_audio = np.concatenate(turn_audio_segments).astype(np.float32)
                turn_index = self.turn_index
                self.status = "saving"
                _save_turn_media_async(
                    self,
                    list(turn_frame_segments),
                    final_audio.copy(),
                    sample_rate=sample_rate,
                    fps=tgt_fps,
                    turn_index=turn_index,
                )
                self.status = "completed"
                reset_turn_state()

            self.status = "live_ready"
            while not stop_event.is_set() and not self.closed and request_id == self.request_id:
                try:
                    item = self.reply_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                kind = item.get("kind")
                if kind == "stop":
                    self.status = "stopped"
                    break
                if kind == "interrupted":
                    stream_epoch = self._advance_stream_epoch()
                    self._reset_tracks()
                    reset_turn_state()
                    self.status = "live_ready"
                    continue
                if kind == "turn_complete":
                    finalize_turn()
                    continue
                if kind != "audio":
                    continue

                mime_type = item.get("mime_type", "audio/pcm;rate=24000")
                reply_sr = _extract_sample_rate_from_mime_type(mime_type, default=24000)
                samples = _decode_pcm16_bytes(item.get("audio_bytes", b""))
                if samples.size == 0:
                    continue
                if reply_sr != sample_rate:
                    samples = librosa.resample(samples, orig_sr=reply_sr, target_sr=sample_rate)

                if self.status not in {"running", "saving"}:
                    self.status = "running"
                pending_samples = np.concatenate([pending_samples, samples.astype(np.float32)])
                while pending_samples.size >= speech_slice_len:
                    chunk = pending_samples[:speech_slice_len]
                    pending_samples = pending_samples[speech_slice_len:]
                    process_chunk(chunk)
        except Exception as exc:
            logger.exception("Gemini-driven WebRTC generation failed for session {}", self.session_id)
            self.status = "error"
            self.error_message = str(exc)
        finally:
            stop_event.set()
            if "playback_condition" in locals():
                with playback_condition:
                    playback_condition.notify_all()
            if "playback_thread" in locals():
                playback_thread.join(timeout=1.0)
            with self.control_lock:
                if self.worker is threading.current_thread():
                    self.worker = None
                if self.stop_event is stop_event:
                    self.stop_event = None
            INFERENCE_LOCK.release()

    async def close(self):
        if self.closed:
            return

        self.closed = True
        if self.stop_event is not None:
            self.stop_event.set()
        for track in (self.video_track, self.audio_track):
            self._close_track_queue(track)

        if self.pc.connectionState != "closed":
            await self.pc.close()


@dataclass
class GeminiLiveBridge:
    browser_ws: web.WebSocketResponse
    stream_session: StreamSession
    session_task: Optional[asyncio.Task] = None
    audio_input_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    text_input_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    model_name: str = DEFAULT_GEMINI_MODEL
    voice_name: str = DEFAULT_GEMINI_VOICE
    system_instruction: str = DEFAULT_GEMINI_SYSTEM_PROMPT
    temperature: Optional[float] = DEFAULT_GEMINI_TEMPERATURE
    suppress_session_closed_event: bool = False
    connected: bool = False
    robot_turn_serial: int = 0
    robot_primary_action_id: str = ""
    robot_requested_action_id: str = ""
    robot_turn_mode: str = "idle"
    robot_presentation_started: bool = False
    robot_turn_complete_received: bool = False
    robot_explicit_action_task: Optional[asyncio.Task] = None
    robot_default_gesture_task: Optional[asyncio.Task] = None
    robot_presentation_loop_task: Optional[asyncio.Task] = None
    robot_turn_cleanup_task: Optional[asyncio.Task] = None
    robot_presentation_variant_cursor: int = 0
    robot_last_output_activity_at: float = 0.0

    async def start_session(self, config: GeminiSessionConfig) -> None:
        await self.close(send_browser_event=False)
        if GENAI_IMPORT_ERROR is not None:
            raise RuntimeError(
                "The `google-genai` package is not installed in this environment yet. Please run `pip install google-genai` first."
            )
        self.model_name = _normalize_model_name(config.model)
        self.voice_name = config.voice_name or DEFAULT_GEMINI_VOICE
        self.system_instruction = config.system_instruction or DEFAULT_GEMINI_SYSTEM_PROMPT
        self.temperature = config.temperature
        self.audio_input_queue = asyncio.Queue()
        self.text_input_queue = asyncio.Queue()
        self.connected = False
        self.suppress_session_closed_event = False
        self._reset_robot_turn_state()
        self.stream_session.start_live_generation()
        self.session_task = asyncio.create_task(self._run_live_session(config.api_key))
        await _safe_send_json(
            self.browser_ws,
            {
                "type": "status",
                "message": f"Connecting to Gemini Live through the Google GenAI SDK (model: {self.model_name}, voice: {self.voice_name})",
            },
        )

    async def send_audio_chunk(self, base64_pcm: str) -> None:
        if not self.session_task:
            raise RuntimeError("The Gemini session has not been established yet.")
        await self.audio_input_queue.put(base64.b64decode(base64_pcm))

    async def send_text(self, text: str) -> None:
        if not self.session_task:
            raise RuntimeError("The Gemini session has not been established yet.")
        await self.text_input_queue.put(text)

    def _cancel_robot_task(self, task: Optional[asyncio.Task]) -> None:
        if task and not task.done():
            task.cancel()

    def _reset_robot_turn_state(self) -> None:
        self.robot_turn_serial += 1
        self.robot_primary_action_id = ""
        self.robot_requested_action_id = ""
        self.robot_turn_mode = "idle"
        self.robot_presentation_started = False
        self.robot_turn_complete_received = False
        self.robot_last_output_activity_at = 0.0
        self._cancel_robot_task(self.robot_turn_cleanup_task)
        self._cancel_robot_task(self.robot_explicit_action_task)
        self._cancel_robot_task(self.robot_default_gesture_task)
        self._cancel_robot_task(self.robot_presentation_loop_task)
        self.robot_turn_cleanup_task = None
        self.robot_explicit_action_task = None
        self.robot_default_gesture_task = None
        self.robot_presentation_loop_task = None

    def _mark_robot_output_activity(self) -> None:
        self.robot_last_output_activity_at = time.monotonic()

    def _robot_output_is_recent(self) -> bool:
        if self.stream_session.is_audio_playback_recent(PRESENTATION_OUTPUT_IDLE_STOP_SECONDS):
            return True
        if self.robot_last_output_activity_at <= 0.0:
            return False
        return (time.monotonic() - self.robot_last_output_activity_at) <= PRESENTATION_OUTPUT_IDLE_STOP_SECONDS

    def _robot_has_pending_turn_task(self) -> bool:
        return any(
            task is not None and not task.done()
            for task in (
                self.robot_explicit_action_task,
                self.robot_default_gesture_task,
                self.robot_presentation_loop_task,
            )
        )

    async def _cleanup_robot_turn_after_playback(self, turn_serial: int) -> None:
        try:
            while turn_serial == self.robot_turn_serial:
                public_status = self.stream_session.get_public_status()
                if (
                    public_status in {"completed", "live_ready", "prepared", "idle", "stopped", "error"}
                    and not self._robot_has_pending_turn_task()
                ):
                    break
                await asyncio.sleep(0.1)

            should_release_presentation = (
                turn_serial == self.robot_turn_serial
                and self.robot_presentation_started
                and not self._robot_has_pending_turn_task()
            )
            if should_release_presentation:
                delay_seconds = max(PRESENTATION_RELEASE_DELAY_SECONDS, 0.0)
                if delay_seconds > 0.0:
                    await asyncio.sleep(delay_seconds)
                if turn_serial == self.robot_turn_serial and not self._robot_has_pending_turn_task():
                    await self._trigger_robot_action(action_id="joint_release_arm_pose")

            if turn_serial == self.robot_turn_serial and not self._robot_has_pending_turn_task():
                self._reset_robot_turn_state()
        except asyncio.CancelledError:
            raise
        finally:
            if self.robot_turn_cleanup_task is asyncio.current_task():
                self.robot_turn_cleanup_task = None

    def _next_presentation_action_id(self) -> str:
        action_id = PRESENTATION_GESTURE_IDS[self.robot_presentation_variant_cursor % len(PRESENTATION_GESTURE_IDS)]
        self.robot_presentation_variant_cursor = (self.robot_presentation_variant_cursor + 1) % len(PRESENTATION_GESTURE_IDS)
        return action_id

    async def _emit_robot_action_event(self, payload: dict) -> None:
        event = dict(payload)
        event["type"] = "robot_action"
        await _safe_send_json(self.browser_ws, event)

    async def _trigger_robot_action(self, *, text: str = "", action_id: str = "") -> dict:
        try:
            payload = await asyncio.to_thread(trigger_robot_action, text, action_id)
        except Exception as exc:  # noqa: BLE001
            payload = {
                "matched": bool(action_id),
                "executed": False,
                "action": action_id,
                "label": action_id,
                "error": str(exc),
                "robot": robot_runtime(),
            }
        await self._emit_robot_action_event(payload)
        return payload

    async def _schedule_default_presentation_gesture(self, text: str, turn_serial: int, delay_seconds: float) -> None:
        try:
            await asyncio.sleep(delay_seconds)
            if (
                turn_serial != self.robot_turn_serial
                or self.robot_primary_action_id
                or self.robot_presentation_started
                or self.robot_turn_complete_received
                or not self._robot_output_is_recent()
            ):
                return
            payload = await self._trigger_robot_action(
                text=text,
                action_id=self._next_presentation_action_id(),
            )
            if payload.get("executed"):
                self.robot_presentation_started = True
                self._ensure_presentation_loop(turn_serial, text)
        except asyncio.CancelledError:
            raise
        finally:
            if self.robot_default_gesture_task is asyncio.current_task():
                self.robot_default_gesture_task = None

    async def _schedule_explicit_robot_action(
        self,
        text: str,
        action_id: str,
        turn_serial: int,
        delay_seconds: float,
    ) -> None:
        try:
            if delay_seconds > 0.0:
                await asyncio.sleep(delay_seconds)
            if turn_serial != self.robot_turn_serial or self.robot_primary_action_id or self.robot_turn_mode == "presentation":
                return
            payload = await self._trigger_robot_action(text=text, action_id=action_id)
            if payload.get("executed"):
                self.robot_turn_mode = "explicit_action"
                self.robot_primary_action_id = action_id
        except asyncio.CancelledError:
            raise
        finally:
            if self.robot_explicit_action_task is asyncio.current_task():
                self.robot_explicit_action_task = None

    def _ensure_presentation_loop(self, turn_serial: int, text: str) -> None:
        if self.robot_presentation_loop_task and not self.robot_presentation_loop_task.done():
            return
        self.robot_presentation_loop_task = asyncio.create_task(
            self._presentation_loop(turn_serial, text)
        )

    async def _presentation_loop(self, turn_serial: int, text: str) -> None:
        try:
            while (
                self.connected
                and turn_serial == self.robot_turn_serial
                and not self.robot_primary_action_id
                and self.robot_presentation_started
            ):
                await asyncio.sleep(PRESENTATION_GESTURE_LOOP_GAP_SECONDS)
                if (
                    turn_serial != self.robot_turn_serial
                    or self.robot_primary_action_id
                    or not self.robot_presentation_started
                    or self.robot_turn_complete_received
                    or not self._robot_output_is_recent()
                ):
                    return
                payload = await self._trigger_robot_action(
                    text=text,
                    action_id=self._next_presentation_action_id(),
                )
                if not payload.get("executed"):
                    return
        except asyncio.CancelledError:
            raise
        finally:
            if self.robot_presentation_loop_task is asyncio.current_task():
                self.robot_presentation_loop_task = None

    async def _handle_robot_input_transcription(self, text: str) -> None:
        runtime = robot_runtime()
        if not runtime.get("enabled"):
            return

        normalized = (text or "").strip()
        if not normalized:
            return

        matched_spec = pick_robot_action(normalized)
        if not matched_spec:
            return

        self.robot_requested_action_id = matched_spec["id"]

        if matched_spec["id"] in PRESENTATION_GESTURE_IDS:
            self.robot_turn_mode = "presentation"
            return

        if self.robot_turn_mode == "presentation":
            return

        self.robot_turn_mode = "explicit_action"
        if self.robot_primary_action_id == matched_spec["id"]:
            return

        if self.robot_explicit_action_task and not self.robot_explicit_action_task.done():
            return

        turn_serial = self.robot_turn_serial
        delay_seconds = max(EXPLICIT_ACTION_TRIGGER_DELAY_SECONDS, 0.0)
        self._cancel_robot_task(self.robot_default_gesture_task)
        self._cancel_robot_task(self.robot_presentation_loop_task)
        self._cancel_robot_task(self.robot_explicit_action_task)
        self.robot_default_gesture_task = None
        self.robot_presentation_loop_task = None
        self.robot_explicit_action_task = asyncio.create_task(
            self._schedule_explicit_robot_action(
                normalized,
                matched_spec["id"],
                turn_serial,
                delay_seconds,
            )
        )

    async def _handle_robot_output_transcription(self, text: str) -> None:
        runtime = robot_runtime()
        if not runtime.get("enabled"):
            return

        normalized = (text or "").strip()
        if not normalized:
            return

        requested_spec = None
        if self.robot_requested_action_id:
            requested_spec = pick_robot_action("", requested_action=self.robot_requested_action_id)
        matched_spec = requested_spec or pick_robot_action(normalized)
        turn_serial = self.robot_turn_serial

        if self.robot_turn_mode == "presentation":
            matched_spec = (
                matched_spec
                if matched_spec and matched_spec["id"] in PRESENTATION_GESTURE_IDS
                else requested_spec
                if requested_spec and requested_spec["id"] in PRESENTATION_GESTURE_IDS
                else pick_robot_action("", requested_action=DEFAULT_PRESENTATION_GESTURE_ID)
            )

        if matched_spec and matched_spec["id"] not in PRESENTATION_GESTURE_IDS:
            if self.robot_turn_mode == "presentation":
                return
            if self.robot_primary_action_id == matched_spec["id"]:
                return
            if self.robot_explicit_action_task and not self.robot_explicit_action_task.done():
                return
            turn_serial = self.robot_turn_serial
            delay_seconds = max(EXPLICIT_ACTION_TRIGGER_DELAY_SECONDS, 0.0)
            self._cancel_robot_task(self.robot_default_gesture_task)
            self._cancel_robot_task(self.robot_presentation_loop_task)
            self._cancel_robot_task(self.robot_explicit_action_task)
            self.robot_default_gesture_task = None
            self.robot_presentation_loop_task = None
            self.robot_explicit_action_task = asyncio.create_task(
                self._schedule_explicit_robot_action(
                    normalized,
                    matched_spec["id"],
                    turn_serial,
                    delay_seconds,
                )
            )
            return

        if self.robot_primary_action_id or self.robot_presentation_started:
            return

        if self.robot_default_gesture_task and not self.robot_default_gesture_task.done():
            return

        # Keep daily conversation still by default.
        # Only presentation-style motion should auto-start when it was explicitly
        # requested by the user, or when Gemini's reply itself clearly matches the
        # presentation gesture intent.
        should_start_presentation = bool(
            matched_spec and matched_spec["id"] in PRESENTATION_GESTURE_IDS
        )
        if not should_start_presentation:
            return

        self.robot_turn_mode = "presentation"

        delay_seconds = (
            DEFAULT_PRESENTATION_TRIGGER_DELAY_SECONDS * 0.7
            if matched_spec and matched_spec["id"] in PRESENTATION_GESTURE_IDS
            else DEFAULT_PRESENTATION_TRIGGER_DELAY_SECONDS
        )
        self.robot_default_gesture_task = asyncio.create_task(
            self._schedule_default_presentation_gesture(normalized, turn_serial, delay_seconds)
        )

    async def end_audio_stream(self) -> None:
        if self.session_task:
            await self.audio_input_queue.put(None)

    async def _send_audio_loop(self, session) -> None:
        while True:
            chunk = await self.audio_input_queue.get()
            if chunk is None:
                try:
                    await session.send_realtime_input(audio_stream_end=True)
                except TypeError:
                    logger.warning("Current google-genai version may not support audio_stream_end; skipping.")
                continue

            await session.send_realtime_input(
                audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000")
            )

    async def _send_text_loop(self, session) -> None:
        while True:
            text = await self.text_input_queue.get()
            if text is None:
                break
            await session.send_realtime_input(text=text)

    async def _receive_loop(self, session) -> None:
        while True:
            async for response in session.receive():
                logger.debug("Received Gemini Live response for session {}: {}", self.stream_session.session_id, response)

                if getattr(response, "go_away", None):
                    await _safe_send_json(
                        self.browser_ws,
                        {
                            "type": "status",
                            "level": "warn",
                            "message": f"Gemini requested this connection to end or reconnect soon: {response.go_away}",
                        },
                    )

                server_content = getattr(response, "server_content", None)
                if server_content:
                    input_transcription = getattr(server_content, "input_transcription", None)
                    if input_transcription and getattr(input_transcription, "text", None):
                        self.stream_session.update_input_transcript(input_transcription.text)
                        await _safe_send_json(
                            self.browser_ws,
                            {
                                "type": "input_transcription",
                                "text": input_transcription.text,
                            },
                        )
                        await self._handle_robot_input_transcription(self.stream_session.input_transcript)

                    output_transcription = getattr(server_content, "output_transcription", None)
                    if output_transcription and getattr(output_transcription, "text", None):
                        self._mark_robot_output_activity()
                        self.stream_session.update_output_transcript(output_transcription.text)
                        await _safe_send_json(
                            self.browser_ws,
                            {
                                "type": "output_transcription",
                                "text": output_transcription.text,
                            },
                        )
                        await self._handle_robot_output_transcription(self.stream_session.output_transcript)

                    model_turn = getattr(server_content, "model_turn", None)
                    if model_turn and getattr(model_turn, "parts", None):
                        for part in model_turn.parts:
                            inline_data = getattr(part, "inline_data", None)
                            if inline_data and getattr(inline_data, "data", None):
                                self._mark_robot_output_activity()
                                self.stream_session.enqueue_gemini_audio(
                                    inline_data.data,
                                    getattr(inline_data, "mime_type", "audio/pcm;rate=24000"),
                                )

                    if getattr(server_content, "interrupted", False):
                        self.stream_session.mark_interrupted()
                        self._reset_robot_turn_state()
                        await _safe_send_json(self.browser_ws, {"type": "interrupted"})

                    if getattr(server_content, "turn_complete", False):
                        turn_serial = self.robot_turn_serial
                        self.robot_turn_complete_received = True
                        self.stream_session.mark_turn_complete()
                        self._cancel_robot_task(self.robot_turn_cleanup_task)
                        self.robot_turn_cleanup_task = asyncio.create_task(
                            self._cleanup_robot_turn_after_playback(turn_serial)
                        )
                        await _safe_send_json(self.browser_ws, {"type": "turn_complete"})

            logger.debug("Gemini receive iterator completed, re-entering receive loop for session {}", self.stream_session.session_id)

    async def _run_live_session(self, api_key: str) -> None:
        if GENAI_IMPORT_ERROR is not None:
            raise RuntimeError(
                "The `google-genai` package is not installed in this environment yet. Please run `pip install google-genai` first."
            )

        client = genai.Client(api_key=api_key)
        config = _build_live_config(
            system_instruction=self.system_instruction,
            voice_name=self.voice_name,
            temperature=self.temperature,
        )
        try:
            attempt = 0
            while True:
                audio_task = None
                text_task = None
                try:
                    logger.info(
                        "Connecting to Gemini Live with model={} attempt={}/{}",
                        self.model_name,
                        attempt + 1,
                        GEMINI_LIVE_MAX_RECONNECT_ATTEMPTS,
                    )
                    async with client.aio.live.connect(model=self.model_name, config=config) as session:
                        self.connected = True
                        await _safe_send_json(
                            self.browser_ws,
                            {
                                "type": "session_ready",
                                "model": self.model_name,
                            },
                        )
                        await _safe_send_json(
                            self.browser_ws,
                            {
                                "type": "status",
                                "level": "ok",
                                "message": "Gemini Live session established. Future reply audio will directly drive avatar generation.",
                            },
                        )

                        audio_task = asyncio.create_task(self._send_audio_loop(session))
                        text_task = asyncio.create_task(self._send_text_loop(session))
                        await self._receive_loop(session)
                    break
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    retryable = _is_retryable_live_error(exc) and attempt + 1 < GEMINI_LIVE_MAX_RECONNECT_ATTEMPTS
                    if retryable:
                        attempt += 1
                        logger.warning(
                            "Gemini Live connection dropped for session {}. Retrying attempt {}/{} after {}s: {}",
                            self.stream_session.session_id,
                            attempt + 1,
                            GEMINI_LIVE_MAX_RECONNECT_ATTEMPTS,
                            GEMINI_LIVE_RECONNECT_DELAY_SECONDS,
                            exc,
                        )
                        self.connected = False
                        self._reset_robot_turn_state()
                        await _safe_send_json(
                            self.browser_ws,
                            {
                                "type": "status",
                                "level": "warn",
                                "message": (
                                    f"Gemini Live connection dropped ({exc}). "
                                    f"Retrying in {GEMINI_LIVE_RECONNECT_DELAY_SECONDS:.1f}s "
                                    f"[attempt {attempt + 1}/{GEMINI_LIVE_MAX_RECONNECT_ATTEMPTS}]"
                                ),
                            },
                        )
                        await asyncio.sleep(GEMINI_LIVE_RECONNECT_DELAY_SECONDS)
                        continue

                    raise
                finally:
                    self.connected = False
                    if audio_task:
                        audio_task.cancel()
                        try:
                            await audio_task
                        except asyncio.CancelledError:
                            pass
                    if text_task:
                        text_task.cancel()
                        try:
                            await text_task
                        except asyncio.CancelledError:
                            pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Gemini live session failed for stream session {}", self.stream_session.session_id)
            self.stream_session.stop_generation()
            await _safe_send_json(
                self.browser_ws,
                {
                    "type": "error",
                    "message": f"Gemini Live session error: {exc}",
                },
            )
        finally:
            self._reset_robot_turn_state()
            if not self.suppress_session_closed_event:
                self.stream_session.stop_generation()
                await _safe_send_json(
                    self.browser_ws,
                    {
                        "type": "session_closed",
                        "message": "Gemini connection ended.",
                    },
                )

    async def close(self, *, send_browser_event: bool = True) -> None:
        if self.session_task:
            self.suppress_session_closed_event = True
            self.session_task.cancel()
            try:
                await self.session_task
            except asyncio.CancelledError:
                pass
            self.session_task = None
        self.connected = False
        self._reset_robot_turn_state()
        await self.audio_input_queue.put(None)
        await self.text_input_queue.put(None)
        self.stream_session.stop_generation()
        if send_browser_event:
            await _safe_send_json(
                self.browser_ws,
                {
                    "type": "session_closed",
                    "message": "Gemini session closed.",
                },
            )


def _ensure_pipeline_loaded(ckpt_dir: str, wav2vec_dir: str, model_type: str):
    global pipeline, loaded_ckpt_dir, loaded_wav2vec_dir, loaded_model_type

    with PIPELINE_LOCK:
        if (
            pipeline is None
            or loaded_ckpt_dir != ckpt_dir
            or loaded_wav2vec_dir != wav2vec_dir
            or loaded_model_type != model_type
        ):
            logger.info(
                "Loading pipeline ckpt_dir={} wav2vec_dir={} model_type={}",
                ckpt_dir,
                wav2vec_dir,
                model_type,
            )
            pipeline = get_pipeline(
                world_size=1,
                ckpt_dir=ckpt_dir,
                model_type=model_type,
                wav2vec_dir=wav2vec_dir,
            )
            loaded_ckpt_dir = ckpt_dir
            loaded_wav2vec_dir = wav2vec_dir
            loaded_model_type = model_type

        return pipeline


def _sanitize_filename(filename: str) -> str:
    keep = [c for c in filename if c.isalnum() or c in {"-", "_", "."}]
    sanitized = "".join(keep).strip("._")
    return sanitized or f"upload_{uuid.uuid4().hex[:8]}"


async def _read_avatar_request(request: web.Request) -> Tuple[str, AvatarGenerationConfig]:
    payload = {}
    content_type = request.content_type or ""

    if content_type.startswith("application/json"):
        payload = await request.json()
    else:
        reader = await request.multipart()
        while True:
            part = await reader.next()
            if part is None:
                break

            if part.filename:
                session_id = payload.get("session_id", "unknown")
                upload_dir = os.path.join(UPLOAD_ROOT, session_id)
                os.makedirs(upload_dir, exist_ok=True)
                filename = _sanitize_filename(part.filename)
                save_path = os.path.join(upload_dir, filename)
                with open(save_path, "wb") as file_obj:
                    while True:
                        chunk = await part.read_chunk()
                        if not chunk:
                            break
                        file_obj.write(chunk)
                payload[part.name] = save_path
            else:
                payload[part.name] = (await part.text()).strip()

    session_id = payload.get("session_id", "").strip()
    if not session_id:
        raise web.HTTPBadRequest(text=json.dumps({"error": "session_id is required"}))

    cond_image = payload.get("cond_image_upload") or payload.get("cond_image_path")
    if not cond_image:
        raise web.HTTPBadRequest(text=json.dumps({"error": "cond_image is required"}))

    config = AvatarGenerationConfig(
        ckpt_dir=payload.get("ckpt_dir") or DEFAULT_CKPT_DIR,
        wav2vec_dir=payload.get("wav2vec_dir") or DEFAULT_WAV2VEC_DIR,
        model_type=payload.get("model_type") or DEFAULT_MODEL_TYPE,
        cond_image=cond_image,
        seed=int(payload.get("seed") or DEFAULT_SEED),
        use_face_crop=_parse_bool(payload.get("use_face_crop")),
    )
    return session_id, config


async def index(_request: web.Request):
    ref_options = "\n".join(
        f'<option value="{path}"{" selected" if path == DEFAULT_COND_IMAGE else ""}>{os.path.basename(path)}</option>'
        for path in DEFAULT_REF_IMAGES
    )
    gemini_voice_options = _build_gemini_voice_options()
    page = (
        HTML_PAGE.replace("__DEFAULT_GEMINI_MODEL__", DEFAULT_GEMINI_MODEL)
        .replace("__GEMINI_VOICE_OPTIONS__", gemini_voice_options)
        .replace("__DEFAULT_GEMINI_PROMPT__", DEFAULT_GEMINI_SYSTEM_PROMPT)
        .replace("__DEFAULT_GEMINI_TEMPERATURE__", str(DEFAULT_GEMINI_TEMPERATURE))
        .replace("__DEFAULT_COND_IMAGE__", DEFAULT_COND_IMAGE)
        .replace("__DEFAULT_COND_IMAGE_PREVIEW__", f"/{DEFAULT_COND_IMAGE}")
        .replace("__REF_PRESET_OPTIONS__", ref_options)
        .replace("__LOGO_ASSET_URL__", "/assets/logo.png")
        .replace("__UAEU_ASSET_URL__", "/assets/uaeu.jpg")
        .replace("__ENABLE_LIVE_TRACK_PLAYBACK__", "true" if ENABLE_LIVE_TRACK_PLAYBACK else "false")
    )
    return web.Response(content_type="text/html", text=page)


async def offer(request: web.Request):
    params = await request.json()
    offer_desc = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    session_id = uuid.uuid4().hex
    session = StreamSession(session_id=session_id, pc=pc, loop=asyncio.get_running_loop())
    disconnect_cleanup_task: Optional[asyncio.Task] = None

    PCS.add(pc)
    SESSIONS[session_id] = session
    if ENABLE_LIVE_TRACK_PLAYBACK:
        pc.addTrack(session.audio_track)
        pc.addTrack(session.video_track)

    async def delayed_disconnect_cleanup():
        try:
            await asyncio.sleep(WEBRTC_DISCONNECT_GRACE_SECONDS)
        except asyncio.CancelledError:
            return

        if pc.connectionState == "disconnected":
            logger.warning(
                "session={} pc_state still disconnected after {:.1f}s, cleaning up",
                session_id,
                WEBRTC_DISCONNECT_GRACE_SECONDS,
            )
            await _cleanup_session(session_id)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        nonlocal disconnect_cleanup_task
        logger.info("session={} pc_state={}", session_id, pc.connectionState)
        if pc.connectionState == "disconnected":
            if disconnect_cleanup_task is None or disconnect_cleanup_task.done():
                disconnect_cleanup_task = asyncio.create_task(delayed_disconnect_cleanup())
            return

        if disconnect_cleanup_task is not None and not disconnect_cleanup_task.done():
            disconnect_cleanup_task.cancel()
            disconnect_cleanup_task = None

        if pc.connectionState in {"failed", "closed"}:
            await _cleanup_session(session_id)

    await pc.setRemoteDescription(offer_desc)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response(
        {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "session_id": session_id,
        }
    )


async def generate(request: web.Request):
    try:
        session_id, config = await _read_avatar_request(request)
        session = SESSIONS.get(session_id)
        if session is None:
            return web.json_response(
                {"error": f"session_id {session_id} not found"},
                status=404,
            )

        if not os.path.exists(config.cond_image):
            return web.json_response(
                {"error": f"Condition image not found: {config.cond_image}"},
                status=400,
            )

        session.prepare_avatar(config)
        return web.json_response(
            {
                "status": session.status,
                "session_id": session_id,
                "selected_audio": session.selected_audio,
            }
        )
    except web.HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to start generation")
        return web.json_response({"error": str(exc)}, status=500)


async def stop(request: web.Request):
    payload = await request.json()
    session_id = payload.get("session_id", "")
    session = SESSIONS.get(session_id)
    if session is None:
        return web.json_response({"error": "session not found"}, status=404)

    session.stop_generation()
    return web.json_response({"status": "stopping", "session_id": session_id})


async def status(request: web.Request):
    session_id = request.match_info["session_id"]
    session = SESSIONS.get(session_id)
    if session is None:
        return web.json_response({"error": "session not found"}, status=404)

    public_status = session.get_public_status()
    return web.json_response(
        {
            "session_id": session_id,
            "status": public_status,
            "error": session.error_message,
            "output_path": session.output_path,
            "output_url": _output_path_to_url(session.output_path),
            "selected_audio": session.selected_audio,
            "pc_state": session.pc.connectionState,
            "input_transcript": session.input_transcript,
            "output_transcript": session.output_transcript,
        }
    )


async def control_ws(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    bridge: Optional[GeminiLiveBridge] = None

    await _safe_send_json(
        ws,
        {
            "type": "status",
            "message": "Browser connected to the Gemini control channel. Waiting for start_session.",
        },
    )

    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue

            payload = json.loads(msg.data)
            msg_type = payload.get("type")

            if msg_type == "start_session":
                session_id = (payload.get("session_id") or "").strip()
                if not session_id:
                    await _safe_send_json(ws, {"type": "error", "message": "session_id is required"})
                    continue

                session = SESSIONS.get(session_id)
                if session is None:
                    await _safe_send_json(ws, {"type": "error", "message": f"session_id {session_id} not found"})
                    continue
                if session.avatar_config is None:
                    await _safe_send_json(
                        ws,
                        {
                            "type": "error",
                            "message": "Please start the conversation first so the backend can prepare the condition image and FlashHead settings.",
                        },
                    )
                    continue

                api_key = (payload.get("api_key") or "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
                if not api_key:
                    await _safe_send_json(ws, {"type": "error", "message": "Missing GEMINI_API_KEY"})
                    continue

                try:
                    bridge = GeminiLiveBridge(browser_ws=ws, stream_session=session)
                    await bridge.start_session(
                        GeminiSessionConfig(
                            api_key=api_key,
                            model=_normalize_model_name(payload.get("model")),
                            voice_name=_resolve_gemini_voice_name(payload.get("voice_name")),
                            system_instruction=(payload.get("system_instruction") or DEFAULT_GEMINI_SYSTEM_PROMPT).strip(),
                            temperature=_coerce_temperature(payload.get("temperature")),
                        )
                    )
                except Exception as exc:
                    logger.exception("Failed to start Gemini session for webrtc stream session {}", session_id)
                    await _safe_send_json(
                        ws,
                        {
                            "type": "error",
                            "message": f"Failed to start the Gemini session: {exc}",
                        },
                    )
                    bridge = None
            elif msg_type == "audio_chunk":
                if bridge is None:
                    await _safe_send_json(ws, {"type": "error", "message": "The Gemini session has not been established yet."})
                    continue
                await bridge.send_audio_chunk(payload.get("data", ""))
            elif msg_type == "audio_stream_end":
                if bridge is not None:
                    await bridge.end_audio_stream()
            elif msg_type == "text":
                if bridge is None:
                    await _safe_send_json(ws, {"type": "error", "message": "The Gemini session has not been established yet."})
                    continue
                text = (payload.get("text") or "").strip()
                if text:
                    await bridge.send_text(text)
            elif msg_type == "close_session":
                if bridge is not None:
                    await bridge.close(send_browser_event=False)
                    bridge.stream_session.stop_generation()
                await _safe_send_json(ws, {"type": "session_closed", "message": "Gemini session closed."})
            elif msg_type == "ping":
                await _safe_send_json(ws, {"type": "pong"})
            else:
                await _safe_send_json(ws, {"type": "error", "message": f"Unsupported message type: {msg_type}"})
    except Exception as exc:
        logger.exception("control_ws crashed")
        await _safe_send_json(
            ws,
            {
                "type": "error",
                "message": f"Gemini control channel exited unexpectedly: {exc}",
            },
        )
    finally:
        if bridge is not None:
            await bridge.close(send_browser_event=False)

    return ws


async def _cleanup_session(session_id: str):
    session = SESSIONS.pop(session_id, None)
    if session is None:
        return

    PCS.discard(session.pc)
    await session.close()


async def on_shutdown(_app: web.Application):
    session_ids = list(SESSIONS.keys())
    for session_id in session_ids:
        await _cleanup_session(session_id)


def build_app() -> web.Application:
    app = web.Application(client_max_size=1024 ** 3)
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_static("/assets", os.path.join(os.path.dirname(__file__), "assets"))
    app.router.add_static("/examples", os.path.join(os.path.dirname(__file__), "examples"))
    app.router.add_static("/generated", os.path.abspath(OUTPUT_ROOT))
    app.router.add_get("/ws", control_ws)
    app.router.add_post("/offer", offer)
    app.router.add_post("/api/generate", generate)
    app.router.add_post("/api/stop", stop)
    app.router.add_get("/api/status/{session_id}", status)
    return app


def main():
    parser = argparse.ArgumentParser(description="SoulX-FlashHead WebRTC streaming server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8080, type=int)
    args = parser.parse_args()

    os.makedirs(UPLOAD_ROOT, exist_ok=True)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    web.run_app(build_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
