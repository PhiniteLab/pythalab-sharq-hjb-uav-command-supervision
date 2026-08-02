# ownerDocs — owner documentation workspace

The owner's working area. Structure is standard across the collection
(skill: owner-docs-scaffold); content is repo-specific.

| Area | Who writes | Content |
|---|---|---|
| myGeneralFiles/ | owner | identity, role, standing context |
| myResearch/ | owner | raw research inputs: reports, prompt outputs, ideas |
| myLiterature/pdfs/ | Claude (manifest) | corpus PDFs (gitignored) + manifest.json |
| myLiterature/review/ | Claude, owner evaluates | consolidated literature reviews |
| myManuscriptRules/ | owner (binding) | academic-writing ruleset — binds all manuscript work |
| myMethods/ | Claude, owner evaluates | method specs, installation & build plans |
| myProgression/ | Claude at WU closure | cumulative scientific-progression ledger |
| CLAUDE-REVIEW/ | Claude | adversarial reviews, specs, verification scripts |
| CODEX-REVIEW/ | Codex | independent reviews — READ-ONLY for Claude |

Empty folders are placeholders, not obligations. The owner owns everything in this
tree except `CLAUDE-REVIEW/` (Claude writes, owner reads); `CODEX-REVIEW/` is
always read-only for Claude.
