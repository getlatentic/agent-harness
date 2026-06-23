//! `skill` — load a discovered skill's full instructions by name. The model
//! sees a catalog of skills in its system prompt (built in `crate::openai_compatible::skills`) and
//! calls this to pull a skill's body into context on demand (OpenCode's lazy
//! design — MIT). Read-only → offered in both modes.

use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use crate::ToolKind;

use super::{parse_args, schema_for, Tool, ToolCtx, ToolOutcome};

#[derive(Deserialize, JsonSchema)]
struct SkillArgs {
    /// The name of the skill to load (one of those listed in the system prompt).
    name: String,
}

pub(super) struct LoadSkill;
impl Tool for LoadSkill {
    fn id(&self) -> &str {
        "skill"
    }
    fn description(&self) -> &str {
        "Load a skill's full instructions by name. Call it when a task matches a \
         skill listed under 'Available skills' in your system prompt; its \
         content is returned for you to follow."
    }
    fn parameters(&self) -> Value {
        schema_for::<SkillArgs>()
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Other
    }
    fn mutating(&self) -> bool {
        false
    }
    fn truncates_output(&self) -> bool {
        false // a skill's body is instructions the model must see in full
    }
    fn execute(&self, args: &Value, ctx: &ToolCtx) -> ToolOutcome {
        let a: SkillArgs = match parse_args(args) {
            Ok(a) => a,
            Err(o) => return o,
        };
        match ctx.skills.iter().find(|s| s.name == a.name) {
            Some(skill) => ToolOutcome::ok(format!("<skill name=\"{}\">\n{}\n</skill>", skill.name, skill.body)),
            None => {
                let available: Vec<&str> = ctx.skills.iter().map(|s| s.name.as_str()).collect();
                if available.is_empty() {
                    ToolOutcome::err(format!("skill: no skill named `{}` — none are available", a.name))
                } else {
                    ToolOutcome::err(format!("skill: no skill named `{}` — available: {}", a.name, available.join(", ")))
                }
            }
        }
    }
}
