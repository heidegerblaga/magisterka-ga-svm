#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

try:
    from codemie_sdk import CodeMieClient  # type: ignore
    from codemie_sdk.models.workflow import (  # type: ignore
        WorkflowCreateRequest,
        WorkflowMode,
        WorkflowUpdateRequest,
    )
except ImportError:
    try:
        from codemie_sdk_python import CodeMieClient  # type: ignore
        from codemie_sdk_python.models.workflow import (  # type: ignore
            WorkflowCreateRequest,
            WorkflowMode,
            WorkflowUpdateRequest,
        )
    except ImportError:
        CodeMieClient = None  # type: ignore[assignment]
        WorkflowCreateRequest = None  # type: ignore[assignment]
        WorkflowMode = None  # type: ignore[assignment]
        WorkflowUpdateRequest = None  # type: ignore[assignment]


AUTH_SERVER_URL = "https://auth.codemie.lab.epam.com"
AUTH_CLIENT_ID = "codemie-sdk"
AUTH_REALM_NAME = "codemie-prod"
CODEMIE_API_DOMAIN = "https://codemie.lab.epam.com/code-assistant-api"
DEFAULT_WORKFLOW_FILE = "configs/codemie_ga_svm_workflow.yaml"
AUTO_MODEL_SENTINELS = {"", "__AUTO__", "__AUTO_MODEL__", "AUTO", "auto", None}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or update a CodeMie workflow from a local YAML file."
    )
    parser.add_argument(
        "--project",
        default=os.getenv("CODEMIE_PROJECT", "").strip(),
        help="CodeMie project name. Falls back to CODEMIE_PROJECT or the repo directory name.",
    )
    parser.add_argument(
        "--name",
        default=os.getenv("CODEMIE_WORKFLOW_NAME", "").strip(),
        help="Workflow name. Falls back to the YAML top-level name.",
    )
    parser.add_argument(
        "--description",
        default=os.getenv("CODEMIE_WORKFLOW_DESCRIPTION", "").strip(),
        help="Workflow description. Falls back to the YAML top-level description.",
    )
    parser.add_argument(
        "--yaml-file",
        default=DEFAULT_WORKFLOW_FILE,
        help=f"Workflow YAML file. Default: {DEFAULT_WORKFLOW_FILE}",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("CODEMIE_MODEL", "").strip(),
        help="Model name for __AUTO__ placeholders. Falls back to the default enabled CodeMie model.",
    )
    parser.add_argument(
        "--mode",
        default=os.getenv("CODEMIE_WORKFLOW_MODE", "Sequential"),
        choices=["Sequential", "Autonomous", "sequential", "autonomous"],
        help="Workflow mode.",
    )
    parser.add_argument(
        "--shared",
        action=argparse.BooleanOptionalAction,
        default=_parse_bool(os.getenv("CODEMIE_WORKFLOW_SHARED", "false")),
        help="Whether the workflow should be shared.",
    )
    parser.add_argument("--icon-url", default=os.getenv("CODEMIE_WORKFLOW_ICON_URL", "").strip())
    parser.add_argument(
        "--update-if-exists",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Update an existing workflow with the same project and name instead of creating a duplicate.",
    )
    parser.add_argument(
        "--run-after-create",
        action="store_true",
        help="Run the workflow immediately after create/update.",
    )
    parser.add_argument(
        "--user-input",
        default="Analyze the latest GA-SVM results and propose the next best runs.",
        help="Execution input used only with --run-after-create.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render the final request payload without calling the CodeMie API.",
    )
    return parser.parse_args()


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def build_client() -> Any:
    if CodeMieClient is None:
        raise RuntimeError(
            "CodeMie SDK import failed. Install dependencies from requirements.txt first."
        )

    username = os.getenv("CODEMIE_USERNAME", "").strip()
    password = os.getenv("CODEMIE_PASSWORD", "").strip()
    if not username or not password:
        raise RuntimeError(
            "CodeMie credentials are missing. Fill CODEMIE_USERNAME and CODEMIE_PASSWORD in .env."
        )

    return CodeMieClient(
        username=username,
        password=password,
        auth_server_url=AUTH_SERVER_URL,
        auth_client_id=AUTH_CLIENT_ID,
        auth_realm_name=AUTH_REALM_NAME,
        codemie_api_domain=CODEMIE_API_DOMAIN,
    )


def load_workflow_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Workflow YAML file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Workflow YAML is invalid: {exc}") from exc

    if not isinstance(data, dict):
        raise RuntimeError("Workflow YAML must contain a top-level mapping.")
    if not isinstance(data.get("tasks"), list) or not data["tasks"]:
        raise RuntimeError("Workflow YAML must contain a non-empty top-level tasks list.")
    return data


def normalize_mode(mode: str) -> WorkflowMode:
    normalized = mode.strip().lower()
    if normalized == "sequential":
        return WorkflowMode.SEQUENTIAL
    if normalized == "autonomous":
        return WorkflowMode.AUTONOMOUS
    raise RuntimeError(f"Unsupported workflow mode: {mode}")


def get_field(obj: Any, field: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(field)
    return getattr(obj, field, None)


def pick_model_name(client: Any, requested_model: str) -> str:
    if requested_model.strip():
        return requested_model.strip()

    models = list(client.llms.list())
    if not models:
        raise RuntimeError("CodeMie returned no models, so __AUTO__ cannot be resolved.")

    enabled = [model for model in models if bool(get_field(model, "enabled"))]
    default_models = [model for model in enabled if bool(get_field(model, "default"))]
    candidates = default_models or enabled or models
    chosen = candidates[0]
    return (
        get_field(chosen, "deployment_name")
        or get_field(chosen, "base_name")
        or get_field(chosen, "label")
        or ""
    )


def replace_auto_models(node: Any, model_name: str) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "model" and value in AUTO_MODEL_SENTINELS:
                node[key] = model_name
            else:
                replace_auto_models(value, model_name)
        return

    if isinstance(node, list):
        for item in node:
            replace_auto_models(item, model_name)


def resolve_request_payload(
    args: argparse.Namespace,
    workflow_data: dict[str, Any],
    client: Any | None,
) -> dict[str, Any]:
    project = args.project.strip() or Path.cwd().name
    name = args.name.strip() or str(workflow_data.get("name", "")).strip() or Path(args.yaml_file).stem
    description = args.description.strip() or str(workflow_data.get("description", "")).strip()

    if not project:
        raise RuntimeError("Workflow project is empty.")
    if not name:
        raise RuntimeError("Workflow name is empty.")

    model_name = args.model.strip()
    if _workflow_uses_auto_model(workflow_data):
        if client is None and not model_name:
            model_name = "REPLACE_WITH_CODEMIE_MODEL"
        elif not model_name:
            model_name = pick_model_name(client, model_name)
        replace_auto_models(workflow_data, model_name)

    workflow_data["name"] = name
    if description:
        workflow_data["description"] = description

    return {
        "project": project,
        "name": name,
        "description": description or None,
        "yaml_config": yaml.safe_dump(workflow_data, sort_keys=False, allow_unicode=False),
        "mode": normalize_mode(args.mode),
        "shared": bool(args.shared),
        "icon_url": args.icon_url.strip() or None,
        "resolved_model": model_name or None,
    }


def _workflow_uses_auto_model(node: Any) -> bool:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "model" and value in AUTO_MODEL_SENTINELS:
                return True
            if _workflow_uses_auto_model(value):
                return True
        return False

    if isinstance(node, list):
        return any(_workflow_uses_auto_model(item) for item in node)

    return False


def list_project_workflows(client: Any, project: str) -> list[Any]:
    page = 0
    per_page = 100
    workflows: list[Any] = []

    while True:
        batch = list(client.workflows.list(page=page, per_page=per_page, projects=[project]))
        if not batch:
            break
        workflows.extend(batch)
        if len(batch) < per_page:
            break
        page += 1

    return workflows


def find_existing_workflow(client: Any, project: str, name: str) -> Any | None:
    for workflow in list_project_workflows(client, project):
        if str(get_field(workflow, "name") or "").strip() == name and str(
            get_field(workflow, "project") or ""
        ).strip() == project:
            return workflow
    return None


def extract_workflow_id(payload: Any) -> str | None:
    if isinstance(payload, dict):
        value = payload.get("id") or payload.get("_id")
        return str(value) if value else None
    value = get_field(payload, "id")
    return str(value) if value else None


def main() -> int:
    load_dotenv()
    args = parse_args()
    workflow_data = load_workflow_yaml(Path(args.yaml_file))

    client = None if args.dry_run else build_client()
    payload = resolve_request_payload(args, workflow_data, client)

    printable_payload = {
        "project": payload["project"],
        "name": payload["name"],
        "description": payload["description"],
        "mode": payload["mode"].value,
        "shared": payload["shared"],
        "icon_url": payload["icon_url"],
        "resolved_model": payload["resolved_model"],
        "yaml_config": payload["yaml_config"],
    }

    if args.dry_run:
        print(json.dumps({"action": "dry-run", "request": printable_payload}, indent=2))
        return 0

    existing = None
    if args.update_if_exists:
        existing = find_existing_workflow(client, payload["project"], payload["name"])

    request_kwargs = {
        "project": payload["project"],
        "name": payload["name"],
        "description": payload["description"],
        "yaml_config": payload["yaml_config"],
        "mode": payload["mode"],
        "shared": payload["shared"],
        "icon_url": payload["icon_url"],
    }

    if existing is not None:
        workflow_id = str(get_field(existing, "id"))
        response = client.workflows.update(workflow_id, WorkflowUpdateRequest(**request_kwargs))
        action = "updated"
    else:
        response = client.workflows.create_workflow(WorkflowCreateRequest(**request_kwargs))
        workflow_id = extract_workflow_id(response)
        action = "created"

    execution = None
    if args.run_after_create:
        if not workflow_id:
            raise RuntimeError("Workflow ID is missing, so the workflow cannot be executed.")
        execution = client.workflows.run(workflow_id, user_input=args.user_input)

    output = {
        "action": action,
        "workflow_id": workflow_id,
        "project": payload["project"],
        "name": payload["name"],
        "resolved_model": payload["resolved_model"],
        "response": response,
    }
    if execution is not None:
        output["execution"] = execution

    print(json.dumps(output, indent=2, default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
