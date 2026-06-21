"""Context-aware agent that prompts for context before patch submission."""

import concurrent.futures
import json
import re
from pathlib import Path

from pydantic import BaseModel
from rich.console import Console
from rich.rule import Rule

from minisweagent.agents.default import (
    DefaultAgent,
    NonTerminatingException,
    Submitted,
    TerminatingException,
)

console = Console(highlight=False)

def _extract_explore_context_block(text: str) -> str | None:
    m = re.search(r"<EXPLORE_CONTEXT>(.*?)</EXPLORE_CONTEXT>", text, re.DOTALL)
    return None if m is None else m.group(1).strip()

def _validate_explore_context_format(context: str) -> bool:
    if not context or not context.strip():
        return False
    lines = [l.strip() for l in context.strip().split("\n")]
    has_file = False
    has_lines = False
    for line in lines:
        if not line:
            continue
        if line.startswith("File:"):
            has_file = True
            continue
        if line.startswith("Lines:"):
            try:
                range_part = line.split(":", 1)[1].strip()
                start_s, end_s = range_part.split("-", 1)
                start = int(start_s.strip())
                end = int(end_s.strip())
                if start <= 0 or end <= 0 or start > end:
                    return False
                has_lines = True
            except (ValueError, IndexError):
                return False
            continue
        # Any other content is forbidden (keeps it machine-parseable)
        return False
    return has_file and has_lines


def _split_bash_segments(cmd: str) -> list[str]:
    if not (cmd and cmd.strip()):
        return []
    out: list[str] = []
    for part in re.split(r"\s*(?:&&|\|\|)\s*", cmd):
        for seg in re.split(r"\s*;\s*", part):
            s = seg.strip()
            if s:
                out.append(s)
    return out


def _bash_simple_command_reads_file(p: str) -> bool:
    """Heuristic: this shell command fragment prints file / line content (needs EXPLORE_CONTEXT per system prompt)."""
    p = p.strip()
    if not p:
        return False
    if re.match(
        r"^(ls|cd|export|find|echo|mkdir|rm|touch|which|true|false|printf|tee|mktemp)\s",
        p,
    ):
        return False
    if re.match(r"^(?:mv|cp|chmod|chown|install)\s", p):
        return False
    if re.match(r"^(?:grep|rg|egrep|fgrep)\b", p):
        return False
    if re.match(r"^python3?(?:\d+\.\d+)?\s", p) and re.search(
        r"\.py(\s+|$)", p
    ) and " -c" not in p:  # running a script, not a one-liner
        return False
    if re.search(r"\bcat\s*<<", p):
        return False
    if re.search(r"\bcat(?:\s+[>>]?>|\s*>)", p):
        return False
    if re.search(r"\bsed\b", p):
        if re.search(r"(?:^|\s)-i[\d=.,\w-]*\s", p) or re.search(
            r"(?:^|\s)-i$", p
        ) or "--in-place" in p:
            return False
        return True
    if re.search(r"\bcat\b", p):
        return True
    if re.search(r"\b(nl|head|tail|less|more|awk)\b", p):
        return True
    if re.search(r"\b(od|xxd|hexdump|strings)\b", p):
        return True
    if re.search(r"^git\s", p) and re.search(
        r"^git\s+(show|diff|cat-file)\b", p
    ):
        return True
    return False


def _bash_segment_looks_like_file_reading(seg: str) -> bool:
    for pipe in seg.split("|"):
        if _bash_simple_command_reads_file(pipe):
            return True
    return False


def _bash_prints_file_content(bash: str) -> bool:
    for seg in _split_bash_segments(bash):
        if _bash_segment_looks_like_file_reading(seg):
            return True
    return False


def _is_final_submission_command(bash: str) -> bool:
    """Final submit runs `git diff --cached`; that is patch output, not explore-context reading."""
    return bool(
        re.search(
            r"\b(?:COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT|MINI_SWE_AGENT_FINAL_OUTPUT)\b",
            bash,
        )
    )


def _explore_violation_user_message(violation: int, limit: int) -> str:
    return (
        "ERROR: This action appears to print source code content, but EXPLORE_CONTEXT is "
        "missing or malformed.\n"
        "Do NOT run this read action until you provide a valid EXPLORE_CONTEXT block.\n\n"
        "Use this exact response skeleton (fill all required fields):\n\n"
        "<EXPLORE_CONTEXT>\n"
        "File: /absolute/path/to/file.ext\n"
        "Lines: <start>-<end>\n"
        "</EXPLORE_CONTEXT>\n\n"
        "```bash\n"
        "<ONE_COMMAND_THAT_PRINTS_THE_DECLARED_LINES>\n"
        "```\n\n"
        "Required fields:\n"
        "- File: absolute path only\n"
        "- Lines: positive integer range where start <= end\n"
        "- Command: must print the declared file/line content to stdout\n\n"
        f"Consecutive violations: {violation}/{limit}."
    )


class ContextAwareAgentConfig(BaseModel):
    """Config for context-aware agent with additional templates."""

    system_template: str
    instance_template: str
    timeout_template: str
    format_error_template: str
    action_observation_template: str
    context_request_template: str
    context_confirmation_template: str
    action_regex: str = r"```bash\s*\n(.*?)\n```"
    context_regex: str = r"<PATCH_CONTEXT>(.*?)</PATCH_CONTEXT>"
    step_limit: int = 0
    cost_limit: float = 3.0
    save_context_to_file: bool = True
    step_response_timeout: float = 0.0
    """Seconds for one LM step; 0 disables per-step timeout."""
    explore_context_retry_limit: int = 3
    """Consecutive read commands without valid EXPLORE_CONTEXT before ExploreContextEnforcementExceeded; 0 disables."""


class ExploreContextEnforcementExceeded(TerminatingException):
    """Too many file-reading commands without a valid EXPLORE_CONTEXT block."""


class ContextRequested(Exception):
    """Raised when agent wants to submit but needs to provide context first."""


class StepResponseTimeout(Exception):
    """Raised when a single agent step exceeds response timeout."""


class ContextAwareAgent(DefaultAgent):
    """Agent that requires context output before final submission."""

    def __init__(self, *args, config_class=ContextAwareAgentConfig, **kwargs):
        super().__init__(*args, config_class=config_class, **kwargs)
        self.patch_context: str | None = None
        self.context_requested: bool = False
        self._explore_context_violations: int = 0

    def add_message(self, role: str, content: str, **kwargs):
        """Extend supermethod to print messages."""
        super().add_message(role, content, **kwargs)
        if role == "assistant":
            console.print(
                f"\n[red][bold]mini-swe-agent[/bold] (step [bold]{self.model.n_calls}[/bold], [bold]${self.model.cost:.2f}[/bold]):[/red]\n",
                end="",
                highlight=False,
            )
        else:
            console.print(f"\n[bold green]{role.capitalize()}[/bold green]:\n", end="", highlight=False)
        console.print(content, highlight=False, markup=False)

    def has_finished(self, output: dict[str, str]):
        """Check if agent wants to finish. Request context if not provided yet."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if not lines:
            return

        first_line = lines[0].strip()
        if first_line not in ["MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]:
            return

        # Agent wants to submit
        if not self.context_requested:
            # First time: request context
            self.context_requested = True
            raise ContextRequested(
                self.render_template(
                    self.config.context_request_template, submission="".join(lines[1:])  # type: ignore
                )
            )

        # Second time: context should have been provided
        if self.patch_context is None:
            raise Submitted("No context provided. Submitting without context.")

        # Save context if configured
        if self.config.save_context_to_file:  # type: ignore
            self._save_context()

        raise Submitted("".join(lines[1:]))

    def query(self) -> dict:
        """Extend supermethod to show waiting status."""
        with console.status("Waiting for the LM to respond..."):
            return super().query()

    def step(self) -> dict:
        """Override step to extract context from assistant messages."""
        console.print(Rule())
        response = self.query()

        # Try to extract context from the response
        if self.context_requested and self.patch_context is None:
            context_match = re.search(
                self.config.context_regex, response["content"], re.DOTALL  # type: ignore
            )
            if context_match:
                raw_context = context_match.group(1).strip()
                # Validate context format
                if self._validate_context_format(raw_context):
                    self.patch_context = raw_context
                    # Add confirmation message
                    confirmation = self.render_template(
                        self.config.context_confirmation_template,  # type: ignore
                        context_length=len(self.patch_context),
                    )
                    self.add_message("user", confirmation)
                else:
                    # Request properly formatted context
                    error_msg = (
                        "ERROR: The context format is incorrect. "
                        "Please provide ONLY file paths and line ranges in the format:\n"
                        "File: /absolute/path/to/file.ext\n"
                        "Lines: start_line-end_line\n\n"
                        "Do NOT include code snippets, explanations, or any other content. "
                        "Then re-issue the submission command."
                    )
                    self.add_message("user", error_msg)

        return self.get_observation(response)

    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the observation, optionally surfacing explore-context blocks."""
        action = self.parse_action(response)
        content = response.get("content", "")
        explore_context = _extract_explore_context_block(content)
        ec_valid = (
            explore_context is not None
            and _validate_explore_context_format(explore_context)
        )
        limit = int(getattr(self.config, "explore_context_retry_limit", 3) or 0)
        cmd = action.get("action", "")
        requires = (
            limit > 0
            and not _is_final_submission_command(cmd)
            and _bash_prints_file_content(cmd)
        )

        if not requires or ec_valid:
            self._explore_context_violations = 0
        if requires and not ec_valid:
            self._explore_context_violations += 1
            if self._explore_context_violations > limit:
                raise ExploreContextEnforcementExceeded(
                    f"Repeated read-file actions without valid EXPLORE_CONTEXT exceeded the retry limit ({limit}). "
                    "Terminating this instance to avoid an infinite correction loop."
                )
            self.add_message(
                "user",
                _explore_violation_user_message(
                    self._explore_context_violations, limit
                ),
            )
            return {
                "output": "",
                "returncode": -1,
                "action": action.get("action", ""),
            }

        output = self.execute_action(action)
        if explore_context is not None and not _validate_explore_context_format(
            explore_context
        ):
            explore_context = None
        observation = self.render_template(
            self.config.action_observation_template,  # type: ignore[arg-type]
            output=output,
            explore_context=explore_context,
        )
        self.add_message("user", observation)
        return output

    def _validate_context_format(self, context: str) -> bool:
        """Validate that context contains file paths and line ranges in correct format."""
        if not context or len(context.strip()) == 0:
            return False
        
        lines = context.strip().split('\n')
        # Check for at least one File/Lines pair
        has_file = False
        has_lines = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith('File:'):
                has_file = True
            elif line.startswith('Lines:'):
                # Validate line range format: number-number
                try:
                    range_part = line.split(':', 1)[1].strip()
                    if '-' in range_part:
                        start, end = range_part.split('-', 1)
                        int(start.strip())
                        int(end.strip())
                        has_lines = True
                except (ValueError, IndexError):
                    return False
        
        return has_file and has_lines

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run with context request handling."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.patch_context = None
        self.context_requested = False
        self._explore_context_violations = 0
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template))

        while True:
            try:
                self._step_with_timeout_guard()
            except ContextRequested as e:
                self.add_message("user", str(e))
            except StepResponseTimeout as e:
                self.add_message("user", str(e))
                return type(e).__name__, str(e)
            except TerminatingException as e:
                self.add_message("user", str(e))
                return type(e).__name__, str(e)
            except NonTerminatingException as e:
                self.add_message("user", str(e))
            except Exception:
                raise

    def _step_with_timeout_guard(self) -> dict:
        """Execute one step with optional timeout guard for model responses."""
        timeout_s = float(getattr(self.config, "step_response_timeout", 0.0) or 0.0)
        if timeout_s <= 0:
            return self.step()

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self.step)
        try:
            return future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            future.cancel()
            timeout_message = (
                f"Step response timed out after {timeout_s:.1f}s. "
                "Terminating this instance and saving trajectory."
            )
            raise StepResponseTimeout(timeout_message)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _save_context(self):
        """Save extracted context to a JSON file alongside the trajectory."""
        try:
            context_data = {
                "patch_context": self.patch_context,
                "context_length": len(self.patch_context) if self.patch_context else 0,
                "total_steps": self.model.n_calls,
                "total_cost": self.model.cost,
                "messages_count": len(self.messages),
            }
            # Save to extra field for trajectory saving
            if not hasattr(self, "_context_data"):
                self._context_data = context_data
        except Exception:
            pass  # Don't fail if context saving fails

    def get_context_data(self) -> dict:
        """Get the extracted context data for external saving."""
        return {
            "patch_context": self.patch_context,
            "context_length": len(self.patch_context) if self.patch_context else 0,
            "total_steps": self.model.n_calls,
            "total_cost": self.model.cost,
        }

