/**
 * Model registry: fallback defaults, provider IDs, and display names.
 *
 * Mirrors houyi/llm/models.py in the SDK.
 * In production, model lists should be fetched dynamically from provider
 * APIs (e.g. /v1/models) rather than hardcoded here.
 */

// ---------------------------------------------------------------------------
// Canonical model identifiers
// ---------------------------------------------------------------------------

// DeepSeek (SiliconFlow)
export const DEEPSEEK_V3 = "deepseek-ai/DeepSeek-V3";
export const DEEPSEEK_V31 = "deepseek-ai/DeepSeek-V3.1";
export const DEEPSEEK_R1 = "deepseek-ai/DeepSeek-R1";

// OpenAI
export const GPT_4O = "gpt-4o";
export const GPT_4O_MINI = "gpt-4o-mini";

// Anthropic
export const CLAUDE_35_SONNET = "claude-3-5-sonnet";

// Google
export const GEMINI_25_PRO = "gemini-2.5-pro";

// Qwen
export const QWEN_25_72B = "Qwen/Qwen2.5-72B-Instruct";

// Default model used across the UI when no model is specified
export const DEFAULT_MODEL = DEEPSEEK_V3;

// ---------------------------------------------------------------------------
// Provider identifiers (mirrors houyi/llm/models.py)
//
// Google has two distinct providers:
//   - google_ai: Google AI Studio / Gemini API (API key auth)
//   - vertex:    Google Cloud Vertex AI (GCP project auth)
// ---------------------------------------------------------------------------
export const PROVIDER_SILICONFLOW = "siliconflow";
export const PROVIDER_OPENAI = "openai";
export const PROVIDER_ANTHROPIC = "anthropic";
export const PROVIDER_DEEPSEEK = "deepseek";
export const PROVIDER_GOOGLE_AI = "google_ai";
export const PROVIDER_VERTEX = "vertex";

export const PROVIDER_DISPLAY_NAMES: Record<string, string> = {
  [PROVIDER_SILICONFLOW]: "SiliconFlow",
  [PROVIDER_OPENAI]: "OpenAI",
  [PROVIDER_ANTHROPIC]: "Anthropic",
  [PROVIDER_DEEPSEEK]: "DeepSeek",
  [PROVIDER_GOOGLE_AI]: "Google AI (Gemini)",
  [PROVIDER_VERTEX]: "Vertex AI",
};

// Model options for dropdowns
export const MODEL_OPTIONS = [
  { value: DEEPSEEK_V3, label: "DeepSeek-V3 (Standard)" },
  { value: DEEPSEEK_V31, label: "DeepSeek-V3.1 (Latest)" },
  { value: DEEPSEEK_R1, label: "DeepSeek-R1 (Reasoning)" },
  { value: GPT_4O, label: "GPT-4o" },
  { value: CLAUDE_35_SONNET, label: "Claude 3.5 Sonnet" },
  { value: GEMINI_25_PRO, label: "Gemini 2.5 Pro" },
  { value: QWEN_25_72B, label: "Qwen 2.5 72B" },
] as const;
