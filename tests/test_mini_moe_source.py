import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_mini_moe_modules_are_syntax_valid_without_importing_torch():
    for relative_path in (
        "nanovllm/layers/mini_moe.py",
        "nanovllm/layers/triton_moe.py",
        "nanovllm/models/qwen3_mini_moe.py",
        "scripts/convert_qwen3_to_mini_moe.py",
    ):
        ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))


def test_mini_moe_is_built_in_without_eager_optional_imports():
    model_tree = ast.parse(
        (ROOT / "nanovllm/models/registry.py").read_text(encoding="utf-8")
    )
    backend_tree = ast.parse(
        (ROOT / "nanovllm/layers/backends.py").read_text(encoding="utf-8")
    )

    def importing_functions(tree, module):
        return [
            function
            for function in ast.walk(tree)
            if isinstance(function, ast.FunctionDef)
            and any(
                isinstance(node, ast.ImportFrom) and node.module == module
                for node in ast.walk(function)
            )
        ]

    assert len(importing_functions(
        model_tree, "nanovllm.models.qwen3_mini_moe"
    )) == 1
    assert len(importing_functions(
        backend_tree, "nanovllm.layers.triton_attention_backend"
    )) == 1


def test_loader_keeps_dense_checkpoint_loading_and_adds_name_expansion():
    source = (ROOT / "nanovllm/utils/loader.py").read_text(encoding="utf-8")

    assert "checkpoint_parameter_names" in source
    assert "lambda name: (name,)" in source
    assert "for param_name in parameter_names" in source
    assert "validate_loaded_parameters" in source
    assert "if missing:" not in source


def test_mini_moe_owns_strict_cloned_expert_validation():
    source = (ROOT / "nanovllm/models/qwen3_mini_moe.py").read_text(
        encoding="utf-8"
    )

    assert "def validate_loaded_parameters" in source
    assert "mlp.router.weight" in source
    assert "checkpoint did not initialise Mini-MoE parameters" in source


def test_mini_moe_checkpoint_mapping_preserves_native_sparse_parameters():
    tree = ast.parse(
        (ROOT / "nanovllm/models/qwen3_mini_moe.py").read_text(
            encoding="utf-8"
        )
    )
    model_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "Qwen3MiniMoEForCausalLM"
    )
    method = next(
        node
        for node in model_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "checkpoint_parameter_names"
    )
    source = ast.unparse(method)

    assert "experts." in source
    assert "router." in source


def test_mini_moe_checkpoint_mapping_clones_only_dense_mlp_parameters():
    source = (ROOT / "nanovllm/models/qwen3_mini_moe.py").read_text(
        encoding="utf-8"
    )
    # Importing this model needs CUDA/PyTorch, so the structural test above is
    # intentionally the portable check used by the macOS core environment.
    # This assertion pins the early-return guard that makes native checkpoints
    # reloadable without constructing the model.
    assert 'suffix.startswith(("experts.", "router."))' in source
