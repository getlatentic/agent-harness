//! Ranked lookup over tools the model was not shown.
//!
//! Every tool costs its schema in every request. Built-ins are a fixed handful,
//! but MCP servers are open-ended: connect three with twenty tools each and the
//! prompt carries sixty schemas whether or not the task touches any of them.
//!
//! So MCP tools can be *deferred* — registered and callable, absent from the
//! initial list — and found through a search tool that returns the schemas for
//! what matches. Codex does this with an in-process BM25 index; Claude Code
//! does it through the Anthropic API's `tool_reference` blocks, which is not
//! reachable from an OpenAI-compatible endpoint, so the local index is the
//! portable design.
//!
//! BM25 is implemented here rather than pulled in. It is thirty lines against a
//! corpus of tool descriptions, and a dependency for that is surface we would
//! carry into every consumer's build and audit.

/// BM25 term-frequency saturation. 1.2 is the standard choice: raising a term's
/// count past a few occurrences stops helping, which is what keeps a
/// description that repeats a word from crowding out one that is simply about
/// it.
const K1: f64 = 1.2;

/// BM25 length normalisation. 0.75 discounts long documents without
/// overwhelming them — a tool with a thorough description should not be
/// penalised into invisibility for being thorough.
const B: f64 = 0.75;

/// One searchable entry: an id and the text that describes it.
pub(crate) struct Entry {
    pub id: String,
    /// Name and description joined — what a query is matched against.
    pub text: String,
}

/// A BM25 index over tool descriptions, built once per run.
pub(crate) struct Index {
    documents: Vec<Document>,
    average_length: f64,
    total: usize,
}

struct Document {
    id: String,
    terms: Vec<String>,
    length: f64,
}

impl Index {
    pub(crate) fn build(entries: Vec<Entry>) -> Self {
        let documents: Vec<Document> = entries
            .into_iter()
            .map(|entry| {
                let terms = tokenize(&entry.text);
                let length = terms.len() as f64;
                Document { id: entry.id, terms, length }
            })
            .collect();
        let total = documents.len();
        let average_length = if total == 0 {
            0.0
        } else {
            documents.iter().map(|d| d.length).sum::<f64>() / total as f64
        };
        Self { documents, average_length, total }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.total == 0
    }

    /// The `limit` best matches for `query`, best first. Entries scoring zero
    /// are dropped: a tool sharing no term with the query is not a weak match,
    /// it is not a match, and padding the result with noise would spend the
    /// tokens the deferral was meant to save.
    pub(crate) fn search(&self, query: &str, limit: usize) -> Vec<&str> {
        if self.is_empty() {
            return Vec::new();
        }
        let terms = tokenize(query);
        let mut scored: Vec<(f64, &str)> = self
            .documents
            .iter()
            .map(|doc| (self.score(doc, &terms), doc.id.as_str()))
            .filter(|(score, _)| *score > 0.0)
            .collect();
        // Ties broken by id so the same query gives the same answer twice.
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(b.1)));
        scored.into_iter().take(limit).map(|(_, id)| id).collect()
    }

    fn score(&self, doc: &Document, terms: &[String]) -> f64 {
        terms.iter().map(|term| self.term_score(doc, term)).sum()
    }

    fn term_score(&self, doc: &Document, term: &str) -> f64 {
        let frequency = doc.terms.iter().filter(|t| *t == term).count() as f64;
        if frequency == 0.0 {
            return 0.0;
        }
        let containing = self.documents.iter().filter(|d| d.terms.iter().any(|t| t == term)).count() as f64;
        // Standard BM25 IDF with the +0.5 smoothing, floored at zero so a term
        // in every document contributes nothing rather than going negative and
        // penalising a document for containing it.
        let idf = (((self.total as f64 - containing + 0.5) / (containing + 0.5)) + 1.0).ln().max(0.0);
        let normalised = doc.length / self.average_length.max(1.0);
        idf * (frequency * (K1 + 1.0)) / (frequency + K1 * (1.0 - B + B * normalised))
    }
}

/// Lowercase alphanumeric runs. Deliberately plain: tool names and descriptions
/// are short technical English, where stemming mostly turns distinct terms into
/// the same one. Underscores and hyphens split, so `read_file` matches "read".
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index() -> Index {
        Index::build(vec![
            Entry { id: "github_create_issue".into(), text: "github create issue open a new issue on a repository".into() },
            Entry { id: "github_list_prs".into(), text: "github list pull requests for a repository".into() },
            Entry { id: "slack_post".into(), text: "slack post a message to a channel".into() },
            Entry { id: "db_query".into(), text: "run a read only sql query against the database".into() },
        ])
    }

    #[test]
    fn a_query_finds_the_tool_it_describes() {
        let index = index();
        assert_eq!(index.search("file a bug on github", 3).first().copied(), Some("github_create_issue"));
        assert_eq!(index.search("send a slack message", 3).first().copied(), Some("slack_post"));
        assert_eq!(index.search("sql", 3), vec!["db_query"]);
    }

    #[test]
    fn a_query_matching_nothing_returns_nothing() {
        // Not "the least bad option": returning noise would spend the tokens
        // deferral exists to save, and invite a call to an unrelated tool.
        let index = index();
        assert!(index.search("photosynthesis", 5).is_empty());
        assert!(Index::build(Vec::new()).search("anything", 5).is_empty());
    }

    #[test]
    fn a_term_common_to_everything_does_not_decide_the_ranking() {
        // "github" is in two of four documents; on its own it should return
        // both, and adding a discriminating term must pick one.
        let index = index();
        let both = index.search("github", 5);
        assert_eq!(both.len(), 2, "got {both:?}");
        assert_eq!(index.search("github pull requests", 5).first().copied(), Some("github_list_prs"));
    }

    #[test]
    fn results_are_capped_and_ordered_the_same_way_twice() {
        let index = index();
        assert_eq!(index.search("github repository issue", 1).len(), 1, "the limit is honoured");
        assert_eq!(index.search("github repository", 5), index.search("github repository", 5));
    }

    #[test]
    fn identifiers_split_so_a_plain_word_matches() {
        let index = Index::build(vec![Entry { id: "x".into(), text: "read_file fetches a path".into() }]);
        assert_eq!(index.search("read", 3), vec!["x"]);
        assert_eq!(index.search("file", 3), vec!["x"]);
    }
}
