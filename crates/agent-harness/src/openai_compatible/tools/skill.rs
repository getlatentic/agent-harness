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
    /// The skill to load. Omit it to list what is available.
    #[serde(default)]
    name: Option<String>,
}

pub(super) struct LoadSkill;
impl Tool for LoadSkill {
    fn id(&self) -> &str {
        "skill"
    }
    fn description(&self) -> &str {
        "Load a skill's full instructions by name, or call it with no name to \
         list the skills available and what each is for. Use a skill when the \
         task matches its description; its content is returned for you to follow."
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
        // No name is a request for the catalog. It is in the system prompt when
        // it fits the budget; when it does not, this is how the model finds out
        // what exists — the catalog moves from every request to the one that
        // asks for it.
        let Some(name) = a.name else {
            return match super::super::skills::catalog(ctx.skills) {
                Some(catalog) => ToolOutcome::ok(catalog),
                None => ToolOutcome::ok("No skills are available.".to_owned()),
            };
        };
        match ctx.skills.iter().find(|s| s.name == name) {
            Some(skill) => ToolOutcome::ok(format!("<skill name=\"{}\">\n{}\n</skill>", skill.name, skill.body)),
            None => {
                let available: Vec<&str> = ctx.skills.iter().map(|s| s.name.as_str()).collect();
                if available.is_empty() {
                    ToolOutcome::err(format!("skill: no skill named `{}` — none are available", name))
                } else {
                    ToolOutcome::err(format!("skill: no skill named `{}` — available: {}", name, available.join(", ")))
                }
            }
        }
    }
}
