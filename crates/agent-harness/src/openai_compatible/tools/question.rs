//! `question` — ask the user one or more multiple-choice questions. Ported from
//! OpenCode's question tool (MIT), adapted to our event model: it emits a
//! [`RunEvent::AskQuestion`] and **ends the run** (`stop`). The host renders the
//! options as chips; the user's pick returns as the next prompt, which resumes
//! the session — so the model continues with the answer in hand (this is why
//! statefulness had to land first). Read-only → offered in both modes.

use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use crate::{Question, QuestionOption, RunEvent, ToolKind};

use super::{parse_args, schema_for, Tool, ToolCtx, ToolOutcome};

#[derive(Deserialize, JsonSchema)]
struct QuestionArgs {
    /// The questions to ask the user.
    questions: Vec<QuestionPrompt>,
}

#[derive(Deserialize, JsonSchema)]
struct QuestionPrompt {
    /// A short label/heading for the question.
    header: Option<String>,
    /// The question text shown to the user.
    question: String,
    /// The selectable options.
    options: Vec<QuestionOpt>,
    /// Allow selecting more than one option (default false).
    multiple: Option<bool>,
}

#[derive(Deserialize, JsonSchema)]
struct QuestionOpt {
    /// The option label.
    label: String,
    /// What the option means (optional).
    description: Option<String>,
}

// Named `QuestionTool` (not `Question`) to avoid clashing with the neutral
// `crate::Question` payload type it builds.
pub(super) struct QuestionTool;
impl Tool for QuestionTool {
    fn id(&self) -> &str {
        "question"
    }
    fn description(&self) -> &str {
        "Ask the user one or more multiple-choice questions to clarify intent or \
         choose between approaches. The user's answer arrives as the next \
         message — pose the question, then wait for it. Put a recommended option \
         first; a free-text answer is always available, so don't add an 'Other'."
    }
    fn parameters(&self) -> Value {
        schema_for::<QuestionArgs>()
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Other
    }
    fn mutating(&self) -> bool {
        false
    }
    fn in_subagent(&self) -> bool {
        false // a subagent has no user to answer
    }
    fn execute(&self, args: &Value, ctx: &ToolCtx) -> ToolOutcome {
        let a: QuestionArgs = match parse_args(args) {
            Ok(a) => a,
            Err(o) => return o,
        };
        if a.questions.is_empty() {
            return ToolOutcome::err("question: provide at least one question");
        }
        let questions: Vec<Question> = a
            .questions
            .into_iter()
            .map(|q| Question {
                header: q.header,
                prompt: q.question,
                options: q
                    .options
                    .into_iter()
                    .map(|o| QuestionOption { label: o.label, description: o.description })
                    .collect(),
                multi_select: q.multiple.unwrap_or(false),
                // OpenCode auto-adds a "type your own answer" option by default.
                allow_free_text: true,
            })
            .collect();
        let n = questions.len();
        let ask = RunEvent::AskQuestion {
            run_id: ctx.run_id.to_owned(),
            request_id: ctx.call_id.to_owned(),
            questions,
        };
        ToolOutcome::stop(format!(
            "Posed {n} question{} to the user; their answer will arrive as the next message — wait for it before continuing.",
            if n == 1 { "" } else { "s" }
        ))
        .with_events(vec![ask])
    }
}
