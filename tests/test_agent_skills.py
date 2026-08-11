from __future__ import annotations

import re
from pathlib import Path

import yaml

from scripts.agent import sync_skills

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = REPO_ROOT / ".agents" / "skills"
EXPECTED_SKILLS = {
    "alphabrain-codebase-nav",
    "alphabrain-data-resources",
    "alphabrain-deployment",
    "alphabrain-env-troubleshoot",
    "alphabrain-evaluation",
    "alphabrain-setup",
    "alphabrain-training",
}
SCAFFOLD_MARKERS = (
    "[todo",
    "todo:",
    "structuring this skill",
    "replace with the first main section",
    "complete and informative explanation",
    "not every skill requires all three types",
)


def read_skill(path: Path) -> tuple[dict[str, object], str, str]:
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    assert lines and lines[0] == "---", f"{path} must start with YAML frontmatter"
    try:
        closing = lines.index("---", 1)
    except ValueError as exc:
        raise AssertionError(f"{path} has no closing YAML frontmatter delimiter") from exc
    metadata = yaml.safe_load("\n".join(lines[1:closing]))
    assert isinstance(metadata, dict), f"{path} frontmatter must be a mapping"
    return metadata, "\n".join(lines[closing + 1 :]), raw


def test_canonical_skill_contracts() -> None:
    skill_directories = {path.name for path in CANONICAL_ROOT.iterdir() if path.is_dir()}
    assert skill_directories == EXPECTED_SKILLS
    assert set(sync_skills.skill_names()) == EXPECTED_SKILLS

    for skill_name in sorted(EXPECTED_SKILLS):
        skill_root = CANONICAL_ROOT / skill_name
        assert not any(path.is_symlink() for path in skill_root.rglob("*"))
        skill_path = skill_root / "SKILL.md"
        metadata, body, raw = read_skill(skill_path)

        assert set(metadata) == {"name", "description"}
        assert metadata["name"] == skill_name
        description = metadata["description"]
        assert isinstance(description, str) and description.strip()
        assert re.search(r"\b(?:use|trigger)\b", description, flags=re.IGNORECASE), (
            f"{skill_name} needs an English invocation trigger"
        )
        assert re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", description), (
            f"{skill_name} needs a Chinese invocation trigger"
        )
        assert body.strip(), f"{skill_name} needs a non-empty instruction body"
        assert len(raw.splitlines()) < 500, f"{skill_name}/SKILL.md must stay under 500 lines"
        lowered = raw.casefold()
        assert not any(marker in lowered for marker in SCAFFOLD_MARKERS), (
            f"{skill_name}/SKILL.md still contains generated scaffolding"
        )

        openai_path = skill_root / "agents" / "openai.yaml"
        openai_metadata = yaml.safe_load(openai_path.read_text(encoding="utf-8"))
        assert isinstance(openai_metadata, dict)
        interface = openai_metadata.get("interface")
        assert isinstance(interface, dict)
        assert {"display_name", "short_description", "default_prompt"} <= set(interface)
        assert isinstance(interface["display_name"], str) and interface["display_name"].strip()
        assert isinstance(interface["short_description"], str)
        assert 25 <= len(interface["short_description"]) <= 64
        assert isinstance(interface["default_prompt"], str)
        assert f"${skill_name}" in interface["default_prompt"]


def test_claude_mirror_matches_canonical_skills() -> None:
    assert sync_skills.check() == []


def test_sync_detects_drift_and_preserves_unrelated_claude_skills(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / ".agents" / "skills"
    target_root = tmp_path / ".claude" / "skills"
    canonical = source_root / "canonical-skill"
    stale_mirror = target_root / "canonical-skill"
    unrelated = target_root / "local-only-skill"

    (canonical / "agents").mkdir(parents=True)
    canonical_skill = "---\nname: canonical-skill\ndescription: current\n---\n\n# Current\n"
    canonical_openai = "interface:\n  display_name: \"Canonical\"\n"
    (canonical / "SKILL.md").write_text(canonical_skill, encoding="utf-8")
    (canonical / "agents" / "openai.yaml").write_text(canonical_openai, encoding="utf-8")

    stale_mirror.mkdir(parents=True)
    (stale_mirror / "SKILL.md").write_text("stale\n", encoding="utf-8")
    unrelated.mkdir(parents=True)
    unrelated_skill = "local skill must survive\n"
    (unrelated / "SKILL.md").write_text(unrelated_skill, encoding="utf-8")

    monkeypatch.setattr(sync_skills, "SOURCE_ROOT", source_root)
    monkeypatch.setattr(sync_skills, "TARGET_ROOT", target_root)

    drift = sync_skills.check()
    assert "content differs: canonical-skill/SKILL.md" in drift

    sync_skills.sync()

    assert sync_skills.check() == []
    assert (stale_mirror / "SKILL.md").read_text(encoding="utf-8") == canonical_skill
    assert (stale_mirror / "agents" / "openai.yaml").read_text(encoding="utf-8") == canonical_openai
    assert (unrelated / "SKILL.md").read_text(encoding="utf-8") == unrelated_skill

    (stale_mirror / "SKILL.md").unlink()
    (stale_mirror / "SKILL.md").symlink_to(canonical / "SKILL.md")
    assert "mirror must be a copy, not a symlink: canonical-skill/SKILL.md" in sync_skills.check()

    sync_skills.sync()
    assert sync_skills.check() == []
    assert not (stale_mirror / "SKILL.md").is_symlink()
