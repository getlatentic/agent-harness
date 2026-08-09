//! A **direct-model** harness: it speaks the OpenAI-compatible chat API over
//! HTTP and runs the agent loop in Rust — owning the read/write tool surface
//! — instead of wrapping a CLI. One adapter serves every OpenAI-compatible
//! endpoint (local Ollama, OpenRouter, vLLM, LM Studio, …); the vendor is
//! configuration, not a type, so [`OpenHarness::ollama`] and
//! [`OpenHarness::custom`] are constructors, not separate structs.
//!
//! Auth: a host that already holds the secret passes it as `api_key`, which
//! never touches the environment — an exported variable is inherited by every
//! child this crate spawns, including the `bash` tool, which would put the key
//! within reach of the model. `api_key_env` names a variable to read instead,
//! for CI and headless runs. `None` for both means no auth, the local Ollama
//! case. Edits land on disk directly (`previews_edits: false`), gated
//! only by [`RunMode`] (Ask = read-only) — review stays in the host, exactly
//! as for the CLI adapters.
//!
//! Sessions persist when a session dir is configured (`with_session_dir`): each
//! run writes its transcript and `RunRequest.resume` continues a prior session
//! by id; without one, runs are ephemeral. `with_context_tokens` enables
//! compaction near the context limit. Responses stream — each token fragment
//! arrives as a `RunEvent::Text`.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use serde_json::Value;

// Core agent-harness types this module builds on. It's a submodule of
// agent-harness (the `openai-compatible` feature), so they come from the crate
// root: the `Harness` trait it implements + the request/metadata types it uses.
use crate::{
    CredentialSpec, Harness, HarnessCapabilities, HarnessError, HarnessInfo, HarnessModel, InstallHint,
    HarnessReadiness, InstalledModel, ModelManagement, PullProgressCallback,
    RunCallback, RunHandle, RunRequest,
};

mod instructions;
pub use instructions::InstructionSources;
mod ollama;
mod run;
mod session;
mod skills;
pub use skills::global_skill_roots;
mod tools;
mod wire;

pub use session::SessionRecord;
pub use tools::mcp::{McpPrompt, McpPromptArg, McpServer, McpTransport, PromptMessage};

/// Cap on the context window we ask Ollama to load (`num_ctx`). `/api/show` may
/// report a model's full trained context (often 128k+); loading that much KV
/// cache can exhaust memory, and ~32k is ample for the system prompt + tools + a
/// working file. The documented sweet spot for local tool-calling is 16–32k.
const OLLAMA_CTX_CEILING: u64 = 32_768;
/// Fallback `num_ctx` when a model's context can't be probed — still well above
/// Ollama's 4096 default, which silently truncates our system prompt.
const OLLAMA_CTX_DEFAULT: u64 = 8_192;

/// How a harness instance discovers its model list for [`Harness::list_models`].
enum Discovery {
    /// Query Ollama's `/api/tags` live.
    OllamaTags,
    /// A fixed list declared up front (any other OpenAI-compatible endpoint).
    Static(Vec<HarnessModel>),
    /// The models.dev catalog filtered to a provider id — a cloud endpoint that
    /// proxies a known provider (`"anthropic"`, `"openai"`, …).
    ModelsDev(String),
}

/// A named subagent the `task` tool can spawn (its own role prompt + optional
/// model), registered via [`OpenHarness::with_agent`].
#[derive(Debug, Clone)]
pub struct AgentDef {
    /// One-line description shown to the model when it picks a `subagent_type`.
    pub description: String,
    /// The subagent's system prompt (replaces the default coding-assistant base);
    /// `None` keeps the default.
    pub system_prompt: Option<String>,
    /// Model override for the subagent; `None` uses the run's model.
    pub model: Option<String>,
}

/// Per-token pricing for a model, so a run can attach an estimated cost to
/// [`crate::RunEvent::Usage`]. Register with
/// [`OpenHarness::with_model_cost`]. Rates are USD per million tokens.
#[derive(Debug, Clone, Copy)]
pub struct ModelCost {
    /// USD per million input (prompt) tokens.
    pub input_per_mtok: f64,
    /// USD per million output (completion) tokens.
    pub output_per_mtok: f64,
    /// USD per million cache-read tokens (often ~0.1x input); `None` falls back
    /// to the input rate.
    pub cache_read_per_mtok: Option<f64>,
}

/// Whether a matched [`PermissionRule`] allows or denies the call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Permission {
    Allow,
    Deny,
    /// Defer to the host's permission prompt
    /// ([`OpenHarness::with_permission_prompt`]); treated as `Deny`
    /// when no prompt is set.
    Ask,
}

/// A pre-execution gate on tool calls — the non-interactive subset of OpenCode's
/// permission system. Rules are checked in order before a tool runs; the first
/// whose `tool` and `pattern` match decides (allow or deny); no match → allowed.
/// Use it to deny specific dangerous calls (e.g. `bash` matching `rm -rf`) that
/// `RunMode`'s coarse read-only/edit gate can't express. (OpenCode's interactive
/// "ask" isn't modeled — a library has no channel to prompt mid-run; host review
/// remains the backstop.)
#[derive(Debug, Clone)]
pub struct PermissionRule {
    /// Tool id this applies to (e.g. `"bash"`, `"edit"`); `None` = any tool.
    pub tool: Option<String>,
    /// Substring the call's subject must contain to match (for `bash` the
    /// command, for file tools the path); `None` = any call to the tool.
    pub pattern: Option<String>,
    /// Allow or deny when matched.
    pub effect: Permission,
}

impl PermissionRule {
    /// Deny every call to `tool`.
    pub fn deny(tool: impl Into<String>) -> Self {
        Self { tool: Some(tool.into()), pattern: None, effect: Permission::Deny }
    }

    /// Deny calls to `tool` whose subject contains `pattern`.
    pub fn deny_matching(tool: impl Into<String>, pattern: impl Into<String>) -> Self {
        Self { tool: Some(tool.into()), pattern: Some(pattern.into()), effect: Permission::Deny }
    }

    /// Allow calls to `tool` whose subject contains `pattern` (short-circuits
    /// later deny rules — list specific allows before a broad deny).
    pub fn allow_matching(tool: impl Into<String>, pattern: impl Into<String>) -> Self {
        Self { tool: Some(tool.into()), pattern: Some(pattern.into()), effect: Permission::Allow }
    }

    /// Ask the host's permission prompt for calls to `tool` matching `pattern`.
    pub fn ask_matching(tool: impl Into<String>, pattern: impl Into<String>) -> Self {
        Self { tool: Some(tool.into()), pattern: Some(pattern.into()), effect: Permission::Ask }
    }

    /// Ask the host's permission prompt for every call to `tool`.
    pub fn ask(tool: impl Into<String>) -> Self {
        Self { tool: Some(tool.into()), pattern: None, effect: Permission::Ask }
    }
}

/// What a permission prompt is asked to decide.
#[derive(Debug, Clone)]
pub struct PermissionRequest {
    /// The tool being called (e.g. `"bash"`).
    pub tool: String,
    /// The call's subject — the command for `bash`, the path for file tools —
    /// when the tool exposes one.
    pub subject: Option<String>,
}

/// A host callback deciding whether a [`Permission::Ask`] tool call may proceed
/// (`true` = allow). Invoked synchronously on the run thread, so a host can block
/// on its own confirmation UI — the interactive analogue of OpenCode's ask
/// prompt. Set via [`OpenHarness::with_permission_prompt`].
pub type PermissionPrompt = std::sync::Arc<dyn Fn(&PermissionRequest) -> bool + Send + Sync>;

/// A direct-model harness over an OpenAI-compatible HTTP endpoint.
pub struct OpenHarness {
    id: String,
    display_name: String,
    description: String,
    /// Base URL with no trailing slash; chat is `{base}/v1/chat/completions`.
    base_url: String,
    /// Env var the API key is read from; `None` → no auth (local Ollama).
    api_key_env: Option<String>,
    /// The key itself, when the host passed one. Preferred over the env var,
    /// and never exported — see [`OpenHarnessConfig::api_key`].
    api_key: Option<String>,
    /// Tool ids withheld from this agent — see [`OpenHarnessConfig::disabled_tools`].
    disabled_tools: Vec<String>,
    /// Instruction-file lookup — see [`OpenHarnessConfig::instruction_sources`].
    instruction_sources: InstructionSources,
    /// Extra skill roots — see [`OpenHarnessConfig::global_skill_roots`].
    global_skill_roots: Vec<std::path::PathBuf>,
    /// Whether this endpoint needs a key — see [`OpenHarnessConfig::requires_api_key`].
    requires_api_key: bool,
    discovery: Discovery,
    /// Used when a run doesn't specify a model via `RunTuning.model`.
    default_model: Option<String>,
    /// When set, sessions are persisted here (transcripts + metadata) and runs
    /// are resumable by id; `None` means ephemeral (no disk writes).
    session_dir: Option<PathBuf>,
    /// The model's context-window size in tokens, when known — enables
    /// compaction (summarize old turns near the limit); `None` disables it.
    context_tokens: Option<u64>,
    /// Named subagents the `task` tool can spawn via `subagent_type`.
    agents: Vec<(String, AgentDef)>,
    /// MCP servers to launch over stdio and expose their tools to the model.
    mcp_servers: Vec<McpServer>,
    /// Per-model pricing for cost estimation on `RunEvent::Usage`.
    model_costs: Vec<(String, ModelCost)>,
    /// Pre-execution permission rules gating tool calls.
    permissions: Vec<PermissionRule>,
    /// Host callback consulted for `Permission::Ask` decisions.
    permission_prompt: Option<PermissionPrompt>,
    /// Inline reasoning tag lifted from streamed output into `Thinking`
    /// (default `Some("think")`); `None` disables extraction.
    reasoning_tag: Option<String>,
}

/// Configuration for [`OpenHarness::custom`] — named fields rather than a long
/// positional argument list, so a call site reads unambiguously (which string is
/// the id vs the display name). Derives `Default`, so optional fields can be
/// omitted with `..Default::default()`.
#[derive(Clone, Debug, Default)]
pub struct OpenHarnessConfig {
    /// Stable id used in the registry / picker (e.g. `"openrouter"`).
    pub id: String,
    /// Human-readable name shown in the UI (e.g. `"OpenRouter"`).
    pub display_name: String,
    /// Base URL with no trailing slash; chat is `{base}/v1/chat/completions`.
    pub base_url: String,
    /// Env var the API key is read from; `None` → no auth (a local server).
    ///
    /// Prefer [`api_key`](Self::api_key) when the host already holds the
    /// secret. An environment variable is process-global: every child this
    /// crate spawns inherits it, including the `bash` tool, which puts the key
    /// within reach of the model itself. The env var remains for CI and
    /// headless runs, where a variable is the natural place for it.
    pub api_key_env: Option<String>,
    /// The API key itself, when the host has it — a value passed in, not a
    /// name to look up. Takes precedence over `api_key_env`, and never touches
    /// the environment, so it stays out of child processes, `/proc/PID/environ`
    /// and crash dumps.
    pub api_key: Option<String>,
    /// Whether this endpoint needs an API key at all.
    ///
    /// Separate from where the key comes from. Setting `api_key_env` implies
    /// it, so existing configs are unchanged; a host that passes the secret as
    /// `api_key` sets this instead. Without the split, "needs a key" and "reads
    /// this variable" were the same flag, so a value-only host reported that no
    /// key was required, showed no field to enter one, and claimed to be ready
    /// without one.
    pub requires_api_key: bool,
    /// Tool ids to withhold from this agent. Every tool is offered by default;
    /// name the ones this host does not want.
    ///
    /// A denylist is right here and wrong for environment variables, for the
    /// same reason in reverse: tool ids are a closed set this crate owns, so
    /// naming one cannot miss a case, while environment names are open-ended
    /// and a name-shaped guess always will. See
    /// [`OpenHarness::builtin_tool_names`] for the set.
    ///
    /// Withheld at construction, so a disabled tool never reaches the model —
    /// it costs no schema in the request and cannot be attempted. That is
    /// different from [`PermissionRule::deny`], which advertises the tool and
    /// refuses the call.
    pub disabled_tools: Vec<String>,
    /// Where `AGENTS.md` / `CLAUDE.md` are read from, and how much of them is
    /// kept. Defaults to the working tree only — nothing under `$HOME` is read
    /// until a host asks, via [`InstructionSources::discover_global`] or its
    /// own paths.
    pub instruction_sources: InstructionSources,
    /// Per-user skill directories scanned in addition to the project's. Empty
    /// by default; [`global_skill_roots`] returns the usual ones
    /// for a host that wants them.
    pub global_skill_roots: Vec<std::path::PathBuf>,
    /// Curated models for the picker; may be empty (free-text ids are allowed,
    /// or call [`OpenHarness::with_models_dev`] for catalog discovery).
    pub models: Vec<HarnessModel>,
}

impl OpenHarness {
    /// Every tool this harness can offer, for a host building the choice into
    /// its own settings rather than hardcoding names that drift as tools are
    /// added. Any of these may go in [`OpenHarnessConfig::disabled_tools`].
    pub fn builtin_tool_names() -> Vec<String> {
        tools::ToolSet::builtin_tool_names()
    }

    /// Local Ollama on its default port, with live `/api/tags` discovery and
    /// no auth. Chat hits Ollama's **native** `/api/chat` (not `/v1`) so
    /// `num_ctx` applies, so the model loads the intended context window
    /// instead of Ollama's truncating 4096 default.
    pub fn ollama() -> Self {
        Self::ollama_at("http://localhost:11434")
    }

    /// Ollama served from somewhere other than the default port — a remote box,
    /// a container, a second instance. Identical to [`Self::ollama`] in every
    /// other respect, including the native `/api/chat` path.
    pub fn ollama_at(base_url: impl Into<String>) -> Self {
        Self {
            id: "ollama".to_owned(),
            display_name: "Ollama".to_owned(),
            description: "Local models served by Ollama via its OpenAI-compatible API.".to_owned(),
            base_url: base_url.into(),
            api_key_env: None,
            api_key: None,
            disabled_tools: Vec::new(),
            instruction_sources: InstructionSources::default(),
            global_skill_roots: Vec::new(),
            // Local Ollama takes no key.
            requires_api_key: false,
            discovery: Discovery::OllamaTags,
            default_model: None,
            session_dir: None,
            context_tokens: None,
            agents: Vec::new(),
            mcp_servers: Vec::new(),
            model_costs: Vec::new(),
            permissions: Vec::new(),
            permission_prompt: None,
            reasoning_tag: Some("think".to_owned()),
        }
    }

    /// Any other OpenAI-compatible endpoint (OpenRouter, vLLM, LM Studio, a
    /// self-hosted gateway), configured by an [`OpenHarnessConfig`] so each
    /// argument is named at the call site.
    pub fn custom(config: OpenHarnessConfig) -> Self {
        let OpenHarnessConfig {
            id,
            display_name,
            base_url,
            api_key_env,
            api_key,
            disabled_tools,
            instruction_sources,
            global_skill_roots,
            requires_api_key,
            models,
        } = config;
        // Naming a variable to read is itself a statement that a key is needed,
        // so every existing config keeps working untouched.
        let requires_api_key = requires_api_key || api_key_env.is_some();
        Self {
            id,
            description: format!("{display_name} via its OpenAI-compatible API."),
            display_name,
            base_url,
            api_key_env,
            api_key,
            disabled_tools,
            instruction_sources,
            global_skill_roots,
            requires_api_key,
            default_model: models.first().map(|m| m.value.clone()),
            discovery: Discovery::Static(models),
            session_dir: None,
            context_tokens: None,
            agents: Vec::new(),
            mcp_servers: Vec::new(),
            model_costs: Vec::new(),
            permissions: Vec::new(),
            permission_prompt: None,
            reasoning_tag: Some("think".to_owned()),
        }
    }

    /// Discover models from the [models.dev](https://models.dev) catalog for the
    /// given provider id (`"anthropic"`, `"openai"`, …) instead of a static list
    /// — for a cloud endpoint that proxies a known provider. Needs the
    /// `agent-harness/models-dev` feature (which `openai-compatible` enables); with no
    /// reachable catalog `list_models` falls back to empty (free-text entry).
    pub fn with_models_dev(mut self, provider: impl Into<String>) -> Self {
        self.discovery = Discovery::ModelsDev(provider.into());
        self
    }

    /// Guard the model-management operations: only the Ollama discovery mode
    /// manages models locally. Returns the same "unsupported" error the trait
    /// defaults give, so a non-Ollama instance reports cleanly instead of
    /// hitting a `/api/...` endpoint that isn't there.
    fn require_ollama_management(&self) -> Result<(), HarnessError> {
        match &self.discovery {
            Discovery::OllamaTags => Ok(()),
            Discovery::Static(_) | Discovery::ModelsDev(_) => Err(HarnessError::Other(format!(
                "{} does not support managing models.",
                self.display_name
            ))),
        }
    }

    /// The API key for this instance: the one the host handed over, else the
    /// configured environment variable.
    fn api_key(&self) -> Option<String> {
        self.api_key
            .clone()
            .or_else(|| self.api_key_env.as_ref().and_then(|env| std::env::var(env).ok()))
            .filter(|v| !v.trim().is_empty())
    }

    /// Resolve the run's context settings as `(compaction_limit, ollama_num_ctx)`.
    ///
    /// For Ollama both are the same *effective* window — the explicit
    /// `with_context_tokens` override (uncapped, the host's call), else the
    /// model's probed `/api/show` context capped at [`OLLAMA_CTX_CEILING`], else
    /// [`OLLAMA_CTX_DEFAULT`]. `ollama_num_ctx` is sent in the native `/api/chat`
    /// request so Ollama loads that window instead of its 4096 default (which
    /// would silently truncate the prompt), and compaction targets the same
    /// number so the two never disagree. Other providers self-manage the window:
    /// `ollama_num_ctx` is `None` (they use `/v1`) and the compaction limit is
    /// the explicit override or `None`.
    fn resolve_context(&self, model: &str) -> (Option<u64>, Option<u64>) {
        match &self.discovery {
            Discovery::OllamaTags => {
                let effective = self
                    .context_tokens
                    .or_else(|| ollama::context_length(&self.base_url, model).map(|n| n.min(OLLAMA_CTX_CEILING)))
                    .unwrap_or(OLLAMA_CTX_DEFAULT);
                (Some(effective), Some(effective))
            }
            // Non-Ollama: no per-model context probe here (models.dev carries
            // limits, but that's a separate enrichment).
            Discovery::Static(_) | Discovery::ModelsDev(_) => (self.context_tokens, None),
        }
    }

    /// The registered per-token pricing for `model`, if any.
    fn model_cost_for(&self, model: &str) -> Option<ModelCost> {
        self.model_costs.iter().find(|(m, _)| m == model).map(|(_, c)| *c)
    }

    /// Persist sessions under `dir` so runs are resumable: each run writes its
    /// transcript here and `RunRequest.resume` continues a prior session by id.
    /// Without this, the harness runs ephemerally (no disk writes).
    pub fn with_session_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.session_dir = Some(dir.into());
        self
    }

    /// Tell the runtime the model's context-window size (in tokens), enabling
    /// compaction: as the transcript nears the limit, older turns are summarized
    /// and recent ones kept verbatim. Without it the full transcript is always
    /// replayed (fine for short sessions).
    pub fn with_context_tokens(mut self, tokens: u64) -> Self {
        self.context_tokens = Some(tokens);
        self
    }

    /// Register a named subagent the `task` tool can spawn via `subagent_type`
    /// (e.g. a focused "reviewer" with its own prompt/model). Registration order
    /// is preserved for the catalog shown to the model.
    pub fn with_agent(mut self, name: impl Into<String>, def: AgentDef) -> Self {
        self.agents.push((name.into(), def));
        self
    }

    /// Register an MCP server to launch over stdio; its advertised tools are
    /// offered to the model (namespaced `name_tool`) and dispatched alongside the
    /// built-ins. Connection is best-effort — a server that fails to start or
    /// handshake is skipped at run time (with a status line), never fatal.
    pub fn with_mcp_server(mut self, server: McpServer) -> Self {
        self.mcp_servers.push(server);
        self
    }

    /// Register per-token pricing for a model, so its runs emit an estimated cost
    /// on [`crate::RunEvent::Usage`]. Rates are USD per million tokens.
    pub fn with_model_cost(mut self, model: impl Into<String>, cost: ModelCost) -> Self {
        self.model_costs.push((model.into(), cost));
        self
    }

    /// Add a [`PermissionRule`] gating tool calls before execution (deny specific
    /// dangerous calls, or allow-list specific ones then deny the rest). Rules
    /// apply in the order added, to the main agent and its subagents.
    pub fn with_permission_rule(mut self, rule: PermissionRule) -> Self {
        self.permissions.push(rule);
        self
    }

    /// Set the callback that decides [`Permission::Ask`] tool calls (`true` =
    /// allow). It's invoked synchronously on the run thread, so a host can block
    /// on its own confirmation UI — the interactive permission channel. Without
    /// it, `Ask` rules deny.
    pub fn with_permission_prompt(
        mut self,
        prompt: impl Fn(&PermissionRequest) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.permission_prompt = Some(std::sync::Arc::new(prompt));
        self
    }

    /// Set the inline reasoning tag lifted from streamed output into `Thinking`
    /// — e.g. `"think"` for `<think>…</think>` (DeepSeek-R1, Qwen3), the default.
    /// The convention is model-specific, so set it to match your model.
    pub fn with_reasoning_tag(mut self, tag: impl Into<String>) -> Self {
        self.reasoning_tag = Some(tag.into());
        self
    }

    /// Disable inline reasoning extraction — stream content verbatim. Use for a
    /// non-reasoning model, or one whose reasoning arrives in a dedicated field
    /// (handled separately).
    pub fn without_reasoning_extraction(mut self) -> Self {
        self.reasoning_tag = None;
        self
    }

    /// All persisted sessions for this harness (newest-updated first), or an
    /// empty list when no session dir is configured. Lets a host render a
    /// conversations view without driving a run.
    pub fn sessions(&self) -> Result<Vec<SessionRecord>, HarnessError> {
        match &self.session_dir {
            Some(dir) => session::FileStore::new(dir.clone())
                .list_records()
                .map_err(HarnessError::Other),
            None => Ok(Vec::new()),
        }
    }

    /// List the prompt templates advertised by the configured MCP servers. Each
    /// server is connected, queried, and disconnected, so this spawns the server
    /// processes; a host surfaces the result for the user to pick from, then
    /// resolves one with [`get_mcp_prompt`](Self::get_mcp_prompt) to seed a run.
    pub fn mcp_prompts(&self) -> Vec<McpPrompt> {
        let cwd = std::env::current_dir().unwrap_or_default();
        tools::mcp::list_prompts(&self.mcp_servers, &cwd)
    }

    /// Resolve a prompt template (by server + name, with `arguments`) to its
    /// messages, for a host to seed a run's prompt.
    pub fn get_mcp_prompt(
        &self,
        server: &str,
        name: &str,
        arguments: &[(String, String)],
    ) -> Result<Vec<PromptMessage>, HarnessError> {
        let cwd = std::env::current_dir().unwrap_or_default();
        tools::mcp::get_prompt(&self.mcp_servers, server, name, arguments, &cwd).map_err(HarnessError::Other)
    }
}

impl Harness for OpenHarness {
    fn info(&self) -> HarnessInfo {
        HarnessInfo {
            id: self.id.clone(),
            display_name: self.display_name.clone(),
            description: self.description.clone(),
            // A local server is something the user installs and runs; a hosted
            // endpoint needs nothing. Readiness reports reachability either way.
            install_hint: match self.discovery {
                Discovery::OllamaTags => Some(InstallHint::url("https://ollama.com/download")),
                Discovery::Static(_) | Discovery::ModelsDev(_) => None,
            },
            capabilities: HarnessCapabilities {
                credential_required: self.requires_api_key,
                previews_edits: false,
                // Dynamic discovery surfaces models via list_models(); a
                // static instance lists them here.
                models: match &self.discovery {
                    Discovery::Static(m) => m.clone(),
                    // Dynamic — surfaced live via list_models().
                    Discovery::OllamaTags | Discovery::ModelsDev(_) => Vec::new(),
                },
                allows_custom_model: true,
                supports_effort: false,
                supports_max_turns: true,
                supports_login: false,
                supports_custom_instructions: true,
            },
        }
    }

    fn readiness(&self) -> HarnessReadiness {
        let base = |ready: bool, error: Option<String>| HarnessReadiness {
            harness_id: self.id.clone(),
            ready,
            // A hosted endpoint isn't "installed"; reachability is the signal.
            installed: true,
            version: None,
            auth_configured: ready,
            error,
            details: Value::Null,
        };
        match &self.discovery {
            // Reachability doubles as the readiness probe: if `/api/tags`
            // answers, Ollama is up.
            Discovery::OllamaTags => match ollama::list_tags(&self.base_url) {
                Ok(_) => base(true, None),
                Err(e) => base(
                    false,
                    Some(format!(
                        "Ollama is not reachable at {} — is it running (`ollama serve`)? ({e})",
                        self.base_url
                    )),
                ),
            },
            // A cloud endpoint (static list or models.dev catalog) is ready once
            // its API key (if any) is present.
            Discovery::Static(_) | Discovery::ModelsDev(_) => {
                if self.requires_api_key && self.api_key().is_none() {
                    // Name the variable only when there is one to set; a host
                    // that passes the key as a value has no variable, and
                    // telling its user to export one would be a dead end.
                    let how = match &self.api_key_env {
                        Some(env) => format!("Set {env} to use {}.", self.display_name),
                        None => format!("Add an API key for {}.", self.display_name),
                    };
                    base(false, Some(how))
                } else {
                    base(true, None)
                }
            }
        }
    }

    fn start(&self, request: RunRequest, on_event: RunCallback) -> Result<RunHandle, HarnessError> {
        let RunRequest { run_id, prompt, cwd, mode, tuning, resume, attachments } = request;
        let model = tuning
            .model
            .as_deref()
            .map(str::trim)
            .filter(|m| !m.is_empty())
            .map(str::to_owned)
            .or_else(|| self.default_model.clone())
            .ok_or_else(|| {
                HarnessError::Other(format!(
                    "{}: no model selected and no default — set RunTuning.model",
                    self.id
                ))
            })?;

        let (context_tokens, ollama_num_ctx) = self.resolve_context(&model);
        let model_cost = self.model_cost_for(&model);
        // Inline images become base64 data URIs the wire attaches to the prompt.
        let image_data_uris: Vec<String> =
            attachments.iter().map(|a| wire::image_data_uri(&a.mime_type, &a.data)).collect();
        let cfg = run::LoopConfig {
            run_id,
            base_url: self.base_url.clone(),
            api_key: self.api_key(),
            disabled_tools: self.disabled_tools.clone(),
            instruction_sources: self.instruction_sources.clone(),
            global_skill_roots: self.global_skill_roots.clone(),
            model,
            prompt,
            cwd: cwd.unwrap_or_else(|| std::env::current_dir().unwrap_or_default()),
            mode,
            max_turns: run::LoopConfig::max_turns_or_default(tuning.max_turns),
            resume,
            store: self.session_dir.clone().map(session::FileStore::new),
            context_tokens,
            ollama_num_ctx,
            agents: self.agents.clone(),
            mcp_servers: self.mcp_servers.clone(),
            output_schema: tuning.output_schema,
            model_cost,
            image_data_uris,
            permissions: self.permissions.clone(),
            permission_prompt: self.permission_prompt.clone(),
            reasoning_tag: self.reasoning_tag.clone(),
            extra_instructions: tuning.extra_instructions,
        };

        let cancel = Arc::new(AtomicBool::new(false));
        let thread_cancel = Arc::clone(&cancel);
        std::thread::spawn(move || run::drive(cfg, thread_cancel, on_event));
        Ok(Box::new(run::OpenAiRun::new(cancel)))
    }

    fn credential(&self) -> CredentialSpec {
        match self.requires_api_key {
            // The account name is the env var when there is one, else the id —
            // a host needs a stable slot name either way.
            true => CredentialSpec {
                label: format!("{} API key", self.display_name),
                keychain_service: self.id.clone(),
                keychain_account: self.api_key_env.clone().unwrap_or_else(|| self.id.clone()),
                required: true,
            },
            // Local Ollama needs no key.
            false => CredentialSpec {
                label: format!("{} (no key required)", self.display_name),
                keychain_service: self.id.clone(),
                keychain_account: String::new(),
                required: false,
            },
        }
    }

    fn list_models(&self) -> Result<Vec<HarnessModel>, HarnessError> {
        match &self.discovery {
            Discovery::OllamaTags => ollama::list_tags(&self.base_url).map_err(HarnessError::Other),
            Discovery::Static(_) => Ok(self.info().capabilities.models),
            Discovery::ModelsDev(provider) => Ok(crate::models_dev::provider_models(provider)),
        }
    }

    // Model management is an Ollama-only capability: it installs/removes models
    // on the local server. Other OpenAI-compatible endpoints (OpenRouter,
    // models.dev-backed) host their models remotely, so the trait defaults
    // (unsupported) stand for them.
    fn model_management(&self) -> Option<ModelManagement> {
        match &self.discovery {
            Discovery::OllamaTags => Some(ModelManagement { base_url: self.base_url.clone() }),
            Discovery::Static(_) | Discovery::ModelsDev(_) => None,
        }
    }

    fn list_installed_models(&self) -> Result<Vec<InstalledModel>, HarnessError> {
        self.require_ollama_management()?;
        ollama::list_installed(&self.base_url).map_err(HarnessError::Other)
    }

    fn pull_model(
        &self,
        model: &str,
        cancel: &std::sync::atomic::AtomicBool,
        on_progress: PullProgressCallback<'_>,
    ) -> Result<(), HarnessError> {
        self.require_ollama_management()?;
        ollama::pull(&self.base_url, model, cancel, on_progress).map_err(HarnessError::Other)
    }

    fn delete_model(&self, model: &str) -> Result<(), HarnessError> {
        self.require_ollama_management()?;
        ollama::delete(&self.base_url, model).map_err(HarnessError::Other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ollama_is_keyless_dynamic_and_editing() {
        let h = OpenHarness::ollama();
        let info = h.info();
        assert_eq!(info.id, "ollama");
        // A local server IS something the user installs — the hint is the only
        // way the picker can say where to get it now that nothing self-installs.
        assert!(info.install_hint.is_some_and(|h| h.url.contains("ollama.com")));
        assert!(!info.capabilities.credential_required);
        assert!(!info.capabilities.previews_edits);
        assert!(info.capabilities.allows_custom_model);
        // Dynamic discovery → no static models in info(); list_models() fills it.
        assert!(info.capabilities.models.is_empty());
        assert!(!h.credential().required);
    }

    #[test]
    fn only_ollama_exposes_model_management() {
        let ollama = OpenHarness::ollama();
        let mgmt = ollama.model_management().expect("Ollama manages models");
        assert_eq!(mgmt.base_url, "http://localhost:11434");

        // A remote OpenAI-compatible endpoint hosts its models, so management is
        // unsupported — and the operations report that rather than calling out.
        let remote = OpenHarness::custom(OpenHarnessConfig {
            id: "openrouter".to_owned(),
            display_name: "OpenRouter".to_owned(),
            base_url: "https://openrouter.ai/api".to_owned(),
            api_key_env: Some("OPENROUTER_API_KEY".to_owned()),
            ..Default::default()
        });
        assert!(remote.model_management().is_none());
        assert!(remote.list_installed_models().is_err());
        assert!(remote.delete_model("whatever").is_err());
        let cancel = std::sync::atomic::AtomicBool::new(false);
        assert!(remote.pull_model("whatever", &cancel, &mut |_| {}).is_err());
    }

    #[test]
    fn a_key_passed_as_a_value_needs_no_environment_variable() {
        // The whole point: a host holding the secret hands it over directly.
        // Nothing is exported, and no variable name is involved.
        let h = OpenHarness::custom(OpenHarnessConfig {
            id: "openrouter".to_owned(),
            display_name: "OpenRouter".to_owned(),
            base_url: "https://openrouter.ai/api".to_owned(),
            api_key: Some("sk-or-v1-example".to_owned()),
            requires_api_key: true,
            ..Default::default()
        });

        // Declares that it needs a key, so a host shows the field for it.
        assert!(h.info().capabilities.credential_required);
        // Has one, so it is ready — no variable was ever set.
        assert!(h.readiness().ready);
        // And the credential slot is real, so a host can store into it.
        let spec = h.credential();
        assert!(spec.required && !spec.keychain_account.is_empty());
    }

    #[test]
    fn a_value_only_provider_without_a_key_says_so_without_naming_a_variable() {
        // Telling someone to export a variable that does not exist is a dead
        // end — this is the message a host with its own key field wants.
        let h = OpenHarness::custom(OpenHarnessConfig {
            id: "acme".to_owned(),
            display_name: "Acme".to_owned(),
            base_url: "https://acme.test".to_owned(),
            requires_api_key: true,
            ..Default::default()
        });
        let readiness = h.readiness();
        assert!(!readiness.ready);
        let error = readiness.error.unwrap_or_default();
        assert!(error.contains("Add an API key"), "{error}");
        assert!(!error.contains("Set "), "{error}");
    }

    #[test]
    fn naming_a_variable_still_implies_a_key_is_needed() {
        // Every config written before the split keeps working untouched.
        let h = OpenHarness::custom(OpenHarnessConfig {
            id: "openrouter".to_owned(),
            display_name: "OpenRouter".to_owned(),
            base_url: "https://openrouter.ai/api".to_owned(),
            api_key_env: Some("OPENROUTER_API_KEY".to_owned()),
            ..Default::default()
        });
        assert!(h.info().capabilities.credential_required);
        assert!(h.credential().required);
    }

    #[test]
    fn a_value_wins_over_the_environment() {
        // A host that passes a key should not be silently overridden by a stale
        // variable in the user's shell.
        std::env::set_var("ACME_KEY", "from-the-environment");
        let h = OpenHarness::custom(OpenHarnessConfig {
            id: "acme".to_owned(),
            display_name: "Acme".to_owned(),
            base_url: "https://acme.test".to_owned(),
            api_key_env: Some("ACME_KEY".to_owned()),
            api_key: Some("from-the-host".to_owned()),
            ..Default::default()
        });
        assert_eq!(h.api_key().as_deref(), Some("from-the-host"));
        std::env::remove_var("ACME_KEY");
    }

    #[test]
    fn custom_requires_its_key_and_lists_static_models() {
        let h = OpenHarness::custom(OpenHarnessConfig {
            id: "openrouter".to_owned(),
            display_name: "OpenRouter".to_owned(),
            base_url: "https://openrouter.ai/api".to_owned(),
            api_key_env: Some("OPENROUTER_API_KEY".to_owned()),
            models: vec![HarnessModel { value: "x-ai/grok".to_owned(), label: "Grok".to_owned() }],
            ..Default::default()
        });
        assert!(h.info().capabilities.credential_required);
        assert!(h.credential().required);
        assert_eq!(h.credential().keychain_account, "OPENROUTER_API_KEY");
        // Static discovery → list_models() returns the curated list.
        assert_eq!(h.list_models().unwrap().len(), 1);
        // The first static model is the default when a run omits one.
        assert_eq!(h.default_model.as_deref(), Some("x-ai/grok"));
    }
}
