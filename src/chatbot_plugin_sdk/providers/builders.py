"""Factory functions for instantiating embedding providers from config dicts."""
from __future__ import annotations

from chatbot_plugin_sdk.providers.endpoint import EndpointProvider
from chatbot_plugin_sdk.providers.fastembed import FastEmbedDenseProvider, FastEmbedSparseProvider
from chatbot_plugin_sdk.providers.gemini import GeminiDenseProvider
from chatbot_plugin_sdk.providers.huggingface import HuggingFaceDenseProvider


def build_dense_provider(config: dict):
    """Instantiate a dense embedding provider from a config dict.

    Config keys:
        provider_type: ``"local"`` (fastembed), ``"gemini"``, ``"huggingface"``, or ``"endpoint"``.
        model: Model name / repo ID (required for ``local``, ``gemini``, and ``huggingface``).
        dimension: Vector dimension (required for all provider types).
        api_key: API key / token (for ``gemini`` and ``huggingface``; Bearer token for ``endpoint``).
        endpoint_url: Service URL (for ``endpoint``).
        rpm / tpm / rpd: Rate-limit parameters (all three required to enable limiting).

    Returns ``None`` when ``provider_type`` is empty or unrecognised.
    """
    provider_type = config.get("provider_type", "")
    model = config.get("model", "")
    dimension = config.get("dimension", 768)
    rpm, tpm, rpd = config.get("rpm"), config.get("tpm"), config.get("rpd")

    if provider_type == "local":
        return FastEmbedDenseProvider(model=model, dimension=dimension)

    if provider_type == "gemini":
        rate_limit = None
        if all(v is not None for v in (rpm, tpm, rpd)):
            from chatbot_plugin_sdk.rate_limit import SlidingWindowStrategy
            rate_limit = SlidingWindowStrategy(rpm=rpm, tpm=tpm, rpd=rpd)
        return GeminiDenseProvider(
            api_key=config.get("api_key", ""),
            model=model,
            dimension=dimension,
            rate_limit=rate_limit,
        )

    if provider_type == "huggingface":
        rate_limit = None
        if all(v is not None for v in (rpm, tpm, rpd)):
            from chatbot_plugin_sdk.rate_limit import SlidingWindowStrategy
            rate_limit = SlidingWindowStrategy(rpm=rpm, tpm=tpm, rpd=rpd)
        return HuggingFaceDenseProvider(
            api_token=config.get("api_key", ""),
            model=model,
            dimension=dimension,
            rate_limit=rate_limit,
        )

    if provider_type == "endpoint":
        rate_limit = None
        if all(v is not None for v in (rpm, tpm, rpd)):
            from chatbot_plugin_sdk.rate_limit import SlidingWindowStrategy
            rate_limit = SlidingWindowStrategy(rpm=rpm, tpm=tpm, rpd=rpd)
        return EndpointProvider(
            url=config["endpoint_url"],
            response_key="dense",
            api_key=config.get("api_key"),
            dimension=dimension,
            rate_limit=rate_limit,
        )

    return None


def build_sparse_provider(config: dict):
    """Instantiate a sparse embedding provider from a config dict.

    Config keys:
        provider_type: ``"local"`` (fastembed) or ``"endpoint"``.
        model: Model name (required for ``local``).
        dimension: Vocabulary size (required for ``local``; used by ``endpoint``
                   to verify the DB column matches).
        endpoint_url: Service URL (for ``endpoint``).
        api_key: Bearer token (for ``endpoint``).
        rpm / tpm / rpd: Rate-limit parameters (all three required to enable limiting).

    Returns ``None`` when ``provider_type`` is empty or unrecognised.
    """
    provider_type = config.get("provider_type", "")
    model = config.get("model", "")
    dimension = config.get("dimension", 30522)
    rpm, tpm, rpd = config.get("rpm"), config.get("tpm"), config.get("rpd")

    if provider_type == "local":
        return FastEmbedSparseProvider(model=model, dimension=dimension)

    if provider_type == "endpoint":
        rate_limit = None
        if all(v is not None for v in (rpm, tpm, rpd)):
            from chatbot_plugin_sdk.rate_limit import SlidingWindowStrategy
            rate_limit = SlidingWindowStrategy(rpm=rpm, tpm=tpm, rpd=rpd)
        return EndpointProvider(
            url=config["endpoint_url"],
            response_key="sparse",
            api_key=config.get("api_key"),
            dimension=dimension,
            timeout=config.get("timeout", 60.0),
            rate_limit=rate_limit,
        )

    return None
