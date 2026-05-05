from __future__ import annotations

import inspect

import modal


APP_NAME = "localwispr-vllm-cohere-probe"


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv")
    .run_commands('uv pip install --system -U "vllm[audio]==0.20.1"')
    .run_commands('uv pip install --system "transformers"')
)

app = modal.App(APP_NAME, image=image)


@app.function(timeout=900)
def probe() -> dict[str, str]:
    import vllm
    import vllm.envs as vllm_envs
    import vllm.renderers.base as renderer_base
    import vllm.renderers.inputs.preprocess as input_preprocess
    import vllm.multimodal.processing.processor as mm_processor
    from vllm.model_executor.models import cohere_asr
    from vllm.transformers_utils.processors import cohere_asr as cohere_processor

    print(f"vllm={vllm.__version__}")
    env_names = sorted(name for name in dir(vllm_envs) if "V1" in name or "ENGINE" in name)
    print("env_names=" + ", ".join(env_names))
    print(f"model_source={cohere_asr.__file__}")
    print(f"processor_source={cohere_processor.__file__}")
    names = sorted(name for name in dir(cohere_asr) if "Cohere" in name or "cohere" in name)
    print("cohere_asr_names=" + ", ".join(names))
    processor_names = sorted(
        name for name in dir(cohere_processor) if "Cohere" in name or "cohere" in name
    )
    print("cohere_processor_names=" + ", ".join(processor_names))

    snippets: dict[str, str] = {}
    for module, names_to_try in (
        (
            cohere_asr,
            [
                "CohereASRProcessingInfo",
                "CohereASRMultiModalProcessor",
                "CohereAsrProcessingInfo",
                "CohereAsrMultiModalProcessor",
                "CohereAsrForConditionalGeneration",
            ],
        ),
        (
            input_preprocess,
            [
                "parse_model_prompt",
                "parse_enc_dec_prompt",
                "_parse_enc_prompt",
                "_parse_dec_prompt",
            ],
        ),
        (
            renderer_base,
            [
                "BaseRenderer",
            ],
        ),
        (
            mm_processor,
            [
                "BaseMultiModalProcessor",
                "EncDecMultiModalProcessor",
            ],
        ),
        (
            cohere_processor,
            [
                "CohereASRProcessor",
                "CohereASRFeatureExtractor",
            ],
        ),
    ):
        for name in names_to_try:
            value = getattr(module, name, None)
            if value is None:
                continue
            try:
                source = inspect.getsource(value)
            except Exception as exc:
                source = f"<inspect failed: {type(exc).__name__}: {exc}>"
            snippets[f"{module.__name__}.{name}"] = source[:12000]
            print(f"\n===== {module.__name__}.{name} =====")
            print(source[:12000])
    return snippets


@app.local_entrypoint()
def main() -> None:
    probe.remote()
