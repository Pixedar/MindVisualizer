"""Analyze probe trajectory through brain regions and explain transitions via GPT."""

import os
from pathlib import Path

import numpy as np


# Brain region knowledge base for RAG
REGION_KNOWLEDGE = {
    "Retrosplenial Area (RSP)": (
        "The retrosplenial cortex is involved in spatial memory, navigation, "
        "and contextual memory. It integrates information from the hippocampus "
        "and visual cortex to form spatial representations."
    ),
    "Anterior Cingulate Area (ACA)": (
        "The anterior cingulate cortex plays roles in attention, error monitoring, "
        "conflict detection, emotional regulation, and decision-making. It connects "
        "limbic and prefrontal regions."
    ),
    "Prelimbic Area (PL)": (
        "The prelimbic cortex is part of the medial prefrontal cortex, involved in "
        "working memory, goal-directed behavior, fear expression, and emotional regulation."
    ),
    "Infralimbic Area (ILA)": (
        "The infralimbic cortex is involved in fear extinction, autonomic regulation, "
        "and emotional processing. It has reciprocal connections with the amygdala."
    ),
    "Orbital Area (ORB)": (
        "The orbitofrontal cortex is critical for reward-based decision making, "
        "value encoding, and behavioral flexibility. It integrates sensory and "
        "emotional information."
    ),
    "Primary Somatosensory Area (SSp)": (
        "The primary somatosensory cortex processes tactile information from the body "
        "including touch, pressure, temperature, and pain."
    ),
    "Supplemental Somatosensory Area (SSs)": (
        "The supplemental somatosensory area processes higher-order tactile information "
        "and is involved in texture discrimination and object recognition by touch."
    ),
    "Primary Motor Area (MOp)": (
        "The primary motor cortex directly controls voluntary movements by sending "
        "signals to muscles via the corticospinal tract."
    ),
    "Secondary Motor Area (MOs)": (
        "The secondary motor area (premotor cortex) is involved in movement planning, "
        "preparation, and the selection of appropriate motor programs."
    ),
    "Primary Visual Area (VISp)": (
        "The primary visual cortex (V1) processes basic visual features including "
        "orientation, spatial frequency, and color."
    ),
    "Anterolateral Visual Area (VISal)": (
        "The anterolateral visual area processes complex visual features and is part "
        "of the ventral visual stream involved in object recognition."
    ),
    "Anteromedial Visual Area (VISam)": (
        "The anteromedial visual area is part of the dorsal visual stream, involved "
        "in spatial processing and visuospatial attention."
    ),
    "Posterior Parietal Association Area (PTLp)": (
        "The posterior parietal cortex integrates sensory information for spatial "
        "awareness, attention, and sensorimotor transformations for reaching and grasping."
    ),
    "Primary Auditory Area (AUDp)": (
        "The primary auditory cortex processes basic acoustic features including "
        "frequency, intensity, and temporal patterns of sound."
    ),
    "Dorsal Auditory Area (AUDd)": (
        "The dorsal auditory area is part of the auditory 'where' pathway, "
        "processing spatial aspects of sound localization."
    ),
    "Ventral Auditory Area (AUDv)": (
        "The ventral auditory area is part of the auditory 'what' pathway, "
        "processing sound identity and complex spectral features."
    ),
    "Temporal Association Area (TEa)": (
        "The temporal association area integrates auditory and visual information "
        "for multisensory processing and is involved in social cognition."
    ),
    "Perirhinal Area (PERI)": (
        "The perirhinal cortex is critical for object recognition memory and serves "
        "as a gateway between neocortical areas and the hippocampus."
    ),
    "Ectorhinal Area (ECT)": (
        "The ectorhinal cortex is involved in the processing of contextual information "
        "and forms part of the parahippocampal region."
    ),
    "Hippocampal Formation (HPF)": (
        "The hippocampal formation is essential for episodic memory formation, "
        "spatial navigation, and memory consolidation. It includes the hippocampus "
        "proper, dentate gyrus, and subiculum."
    ),
    "Entorhinal Area (ENT)": (
        "The entorhinal cortex is the main interface between the hippocampus and "
        "neocortex. It contains grid cells for spatial navigation and is critical "
        "for memory encoding."
    ),
    "Thalamus (TH)": (
        "The thalamus is a major relay station for sensory and motor information. "
        "It routes signals between subcortical areas and the cortex and plays roles "
        "in consciousness, sleep, and alertness."
    ),
    "Hypothalamus (HY)": (
        "The hypothalamus regulates homeostatic functions including body temperature, "
        "hunger, thirst, circadian rhythms, and the autonomic nervous system. "
        "It controls the pituitary gland."
    ),
    "Cerebellum (CB)": (
        "The cerebellum coordinates voluntary movements, balance, and motor learning. "
        "It also contributes to cognitive functions including attention and language."
    ),
}



def _call_direct_responses_api(question: str, model: str) -> str:
    """Direct OpenAI Responses API call for no-RAG mode.

    Avoids the LangChain empty-content failure path seen with GPT-5.4 in the
    background thread.
    """
    _ensure_ssl()

    from dotenv import load_dotenv
    load_dotenv()

    from openai import OpenAI

    client = OpenAI(timeout=60.0)

    instructions = (
        "You are a neuroscience expert interpreting information flow in a "
        "resting-state brain.\n\n"
        "Do NOT repeat the region list.\n"
        "Do explain:\n"
        "1. what specific information is likely flowing,\n"
        "2. how the signal is transformed across regions,\n"
        "3. what spontaneous resting-state process could produce this path,\n"
        "4. the functional purpose of this exact trajectory.\n"
        "Be concrete, not generic."
    )

    resp = client.responses.create(
        model=model,
        instructions=instructions,
        input=question,
        # This is not a model downgrade. It just keeps enough budget for the visible answer.
        reasoning={"effort": "low"},
        max_output_tokens=2500,
        # Optional, if your SDK/version accepts it and you want longer answers:
        # text={"verbosity": "high"},
    )

    text = (resp.output_text or "").strip()
    if text:
        return text

    if getattr(resp, "status", None) == "incomplete" and getattr(resp, "incomplete_details", None):
        reason = getattr(resp.incomplete_details, "reason", "unknown")
        return f"[GPT ERROR] Responses API incomplete: {reason}"

    return "[GPT ERROR] Responses API returned no visible text."

def _get_mesh_volume(mesh_overlay, key: str) -> float:
    """Estimate mesh volume from bounding box (for nesting detection)."""
    bounds = mesh_overlay.get_mesh_bounds(key)
    if bounds is None:
        return float('inf')
    return float((bounds[1] - bounds[0]) * (bounds[3] - bounds[2]) * (bounds[5] - bounds[4]))


def analyze_probe_path(path_positions: np.ndarray, mesh_overlay,
                       sample_every: int = 10,
                       field_mags: np.ndarray | None = None,
                       entropy_sampler=None,
                       entropy_field=None) -> list[dict]:
    """Analyze which brain regions the probe traveled through.

    Handles nested regions: when the probe is inside multiple overlapping
    regions (e.g., a subregion inside a larger area), the smaller region
    is treated as the primary region and the larger one as context.

    Args:
        path_positions: (N,3) probe path
        mesh_overlay: FlowMeshOverlay instance
        sample_every: check every N steps
        field_mags: raw field magnitudes (independent of speed scale)
        entropy_sampler: TriLinearSampler for entropy
        entropy_field: entropy grid (G,G,G) numpy array

    Returns a list of dicts with keys: region_key, region_name, entry_idx,
    exit_idx, relative_position, avg_flow_strength, avg_entropy,
    enclosing_regions.
    """
    if len(path_positions) == 0:
        return []

    region_keys = mesh_overlay.get_all_region_keys()
    # Pre-compute volumes for nesting detection
    volumes = {k: _get_mesh_volume(mesh_overlay, k) for k in region_keys}

    transitions = []
    prev_regions = set()

    for i in range(0, len(path_positions), sample_every):
        pt = path_positions[i]
        in_regions = set()

        for key in region_keys:
            if mesh_overlay.point_in_mesh(pt, key):
                in_regions.add(key)

        # Sample entropy at this point
        ent_val = None
        if entropy_sampler is not None and entropy_field is not None:
            try:
                ent_val = float(entropy_sampler.sample_scalar(entropy_field, pt[None, :])[0])
            except Exception:
                pass

        # Detect entries
        entered = in_regions - prev_regions
        for key in entered:
            center = mesh_overlay.get_mesh_center(key)
            rel_pos = _relative_position(pt, center, mesh_overlay.get_mesh_bounds(key))

            # Find enclosing (larger) regions at this point
            enclosing = []
            for other_key in in_regions:
                if other_key != key and volumes.get(other_key, 0) > volumes.get(key, 0):
                    enclosing.append(mesh_overlay.get_region_name(other_key))

            # Flow strength (independent of speed scale)
            flow_str = None
            if field_mags is not None and i < len(field_mags):
                flow_str = float(field_mags[i])

            # Hemisphere info (if mesh_overlay supports it)
            hemisphere = None
            if hasattr(mesh_overlay, 'get_hemisphere_label'):
                try:
                    hemisphere = mesh_overlay.get_hemisphere_label(key, pt)
                except Exception:
                    pass

            transition_dict = {
                "region_key": key,
                "region_name": mesh_overlay.get_region_name(key),
                "entry_idx": i,
                "exit_idx": None,
                "relative_position": rel_pos,
                "entry_point": pt.copy(),
                "enclosing_regions": enclosing,
                "avg_flow_strength": flow_str,
                "avg_entropy": ent_val,
                "_flow_samples": [flow_str] if flow_str is not None else [],
                "_ent_samples": [ent_val] if ent_val is not None else [],
            }
            if hemisphere is not None:
                transition_dict["hemisphere"] = hemisphere
            transitions.append(transition_dict)

        # Accumulate samples for active transitions
        for t in transitions:
            if t["exit_idx"] is None:
                if field_mags is not None and i < len(field_mags):
                    t["_flow_samples"].append(float(field_mags[i]))
                if ent_val is not None:
                    t["_ent_samples"].append(ent_val)

        # Detect exits
        exited = prev_regions - in_regions
        for key in exited:
            for t in reversed(transitions):
                if t["region_key"] == key and t["exit_idx"] is None:
                    t["exit_idx"] = i
                    if t["_flow_samples"]:
                        t["avg_flow_strength"] = float(np.mean(t["_flow_samples"]))
                    if t["_ent_samples"]:
                        t["avg_entropy"] = float(np.mean(t["_ent_samples"]))
                    break

        prev_regions = in_regions

    # Close any still-open regions
    for t in transitions:
        if t["exit_idx"] is None:
            t["exit_idx"] = len(path_positions) - 1
            if t["_flow_samples"]:
                t["avg_flow_strength"] = float(np.mean(t["_flow_samples"]))
            if t["_ent_samples"]:
                t["avg_entropy"] = float(np.mean(t["_ent_samples"]))

    # Clean up internal fields
    for t in transitions:
        t.pop("_flow_samples", None)
        t.pop("_ent_samples", None)

    # Sort by volume (smaller = more specific) so primary regions come first at each transition
    transitions.sort(key=lambda t: (t["entry_idx"], volumes.get(t["region_key"], float('inf'))))

    return transitions


def _relative_position(point: np.ndarray, center: np.ndarray | None,
                       bounds: np.ndarray | None) -> str:
    """Describe position relative to region center (e.g., 'upper-left')."""
    if center is None or bounds is None:
        return "unknown"

    diff = point - center
    extent = np.array([
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4],
    ])
    extent = np.maximum(extent, 1e-6)
    rel = diff / (extent * 0.5)  # normalized to [-1, 1]

    parts = []
    if abs(rel[2]) > 0.3:
        parts.append("upper" if rel[2] > 0 else "lower")
    if abs(rel[0]) > 0.3:
        parts.append("right" if rel[0] > 0 else "left")
    if abs(rel[1]) > 0.3:
        parts.append("anterior" if rel[1] > 0 else "posterior")

    if not parts:
        return "center"
    return "-".join(parts)


def format_transitions_text(transitions: list[dict],
                            include_branches: list[dict] | None = None) -> str:
    """Format transitions into a human-readable description for GPT.

    Handles nested regions: smaller (more specific) regions are primary,
    with larger enclosing regions noted as context.
    """
    if not transitions:
        return "The probe did not pass through any identified brain regions."

    lines = ["The probe traveled through the following brain regions in order:\n"]
    for i, t in enumerate(transitions, 1):
        duration = (t["exit_idx"] or 0) - (t["entry_idx"] or 0)

        # Include hemisphere in region name if available
        region_label = t['region_name']
        hemisphere = t.get("hemisphere")
        if hemisphere:
            region_label = f"{region_label} ({hemisphere})"

        extras = []
        if t.get("avg_flow_strength") is not None:
            extras.append(f"flow_strength={t['avg_flow_strength']:.4f}")
        if t.get("avg_entropy") is not None:
            extras.append(f"model_entropy={t['avg_entropy']:.3f} "
                          f"({'high uncertainty' if t['avg_entropy'] > 1.5 else 'confident'})")
        extras_str = f", {', '.join(extras)}" if extras else ""

        enclosing = t.get("enclosing_regions", [])
        ctx_str = ""
        if enclosing:
            ctx_str = f" [within larger region: {', '.join(enclosing)}]"

        lines.append(
            f"{i}. {region_label} - entered at position "
            f"({t['relative_position']}), "
            f"traversed for ~{duration} steps{extras_str}{ctx_str}"
        )

    if include_branches:
        lines.append("\nBranching events (alternative flow paths detected):")
        for b in include_branches:
            lines.append(f"  - Branch at step ~{b.get('entry_idx', '?')}: "
                         f"flow split toward {b.get('region_name', 'unknown')}")

    return "\n".join(lines)


def build_rag_documents():
    """Build LangChain documents from the region knowledge base."""
    from langchain_core.documents import Document
    docs = []
    for name, desc in REGION_KNOWLEDGE.items():
        docs.append(Document(
            page_content=f"Brain Region: {name}\nFunction: {desc}",
            metadata={"region_name": name}
        ))
    return docs


def _ensure_ssl():
    """Fix SSL cert path for environments where Git overrides it."""
    try:
        import certifi
        cert_file = certifi.where()
        # Force override if current path is missing (Git installer on Windows
        # often sets SSL_CERT_FILE to a non-existent path)
        cur = os.environ.get("SSL_CERT_FILE", "")
        if not cur or not os.path.isfile(cur):
            os.environ["SSL_CERT_FILE"] = cert_file
        cur2 = os.environ.get("REQUESTS_CA_BUNDLE", "")
        if not cur2 or not os.path.isfile(cur2):
            os.environ["REQUESTS_CA_BUNDLE"] = cert_file
    except ImportError:
        pass


def create_rag_chain(model: str = "gpt-5.4-mini"):
    """Create a LangChain RAG chain with region knowledge."""
    _ensure_ssl()

    from dotenv import load_dotenv
    load_dotenv()

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    docs = build_rag_documents()

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    prompt = ChatPromptTemplate.from_template(
        """You are a neuroscience expert interpreting information flow in a resting-state brain.

Reference knowledge about relevant brain regions:
{context}

{question}

CRITICAL INSTRUCTIONS — read carefully before answering:
1. Do NOT list which regions the probe passed through — the user already knows that.
2. Do NOT give a generic answer like "this flow is associated with the default mode network" — that is too shallow.
3. DO explain what SPECIFIC INFORMATION is likely flowing along this exact path. What is being computed, processed, or communicated? How does the information CHANGE as it passes through each region?
4. DO describe the functional MEANING of this particular flow trajectory: why would information flow in this exact direction, through these specific regions, in this order? What is the computational purpose?
5. DO explain how each region transforms the signal before passing it on — not just what each region "does" in general, but what it does to THIS specific incoming signal.
6. DO consider laterality: if the flow is in the left or right hemisphere, explain what that implies about the type of processing (e.g., left-lateralized language processing, right-lateralized spatial processing).
7. This is a RESTING-STATE brain — describe what spontaneous cognitive process would produce this exact flow. Be concrete: "the brain is likely consolidating a recent spatial memory" is better than "this involves memory processing."
"""
    )

    llm = ChatOpenAI(model=model, temperature=0.3, max_tokens=1500)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def create_direct_chain(model: str = "gpt-5.4-mini"):
    """Create a direct GPT chain without RAG (uses only GPT's own knowledge)."""
    _ensure_ssl()

    from dotenv import load_dotenv
    load_dotenv()

    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_template(
        """You are a neuroscience expert interpreting information flow in a resting-state brain.

Using your deep knowledge of neuroanatomy, functional connectivity, and resting-state \
networks, answer the following question:

{question}

CRITICAL INSTRUCTIONS — read carefully before answering:
1. Do NOT list which regions the probe passed through — the user already knows that.
2. Do NOT give a generic answer like "this flow is associated with the default mode network" — that is too shallow.
3. DO explain what SPECIFIC INFORMATION is likely flowing along this exact path. What is being computed, processed, or communicated? How does the information CHANGE as it passes through each region?
4. DO describe the functional MEANING of this particular flow trajectory: why would information flow in this exact direction, through these specific regions, in this order? What is the computational purpose?
5. DO explain how each region transforms the signal before passing it on — not just what each region "does" in general, but what it does to THIS specific incoming signal.
6. DO consider laterality: if the flow is in the left or right hemisphere, explain what that implies about the type of processing (e.g., left-lateralized language processing, right-lateralized spatial processing).
7. This is a RESTING-STATE brain — describe what spontaneous cognitive process would produce this exact flow. Be concrete: "the brain is likely consolidating a recent spatial memory" is better than "this involves memory processing."
"""
    )

    llm = ChatOpenAI(model=model, temperature=0.3, max_tokens=1500)

    chain = (
        {"question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


_rag_chain = None
_direct_chain = None
_cached_model = None


def analyze_with_gpt(transitions: list[dict], use_rag: bool = True,
                     model: str = "gpt-5.4-mini") -> str:
    global _rag_chain, _direct_chain, _cached_model

    if not transitions:
        return "No brain regions were traversed. Place the probe inside the brain volume."

    description = format_transitions_text(transitions)
    question = (
        f"A probe was placed in a resting-state brain and followed the intrinsic information "
        f"flow field through these regions:\n\n{description}\n\n"
        f"Analyze this SPECIFIC flow pathway in depth:\n"
        f"1. What concrete information is likely being carried along this path?\n"
        f"2. How does the information CHANGE and get TRANSFORMED as it passes through each region?\n"
        f"3. What spontaneous resting-state cognitive process would produce this EXACT flow?\n"
        f"4. What is the functional PURPOSE of information flowing in this exact direction and order?\n"
        f"Do NOT repeat the region list or give a generic network label. Give deep, specific insight."
    )

    # IMPORTANT: bypass LangChain entirely for no-RAG mode
    if not use_rag:
        try:
            print(f"[GPT] Using Responses API directly (model={model}, no RAG)...")
            return _call_direct_responses_api(question, model)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"[GPT ERROR] Direct Responses API failed: {e}"

    # keep existing RAG path for now
    if _cached_model != model:
        _rag_chain = None
        _direct_chain = None
        _cached_model = model

    if _rag_chain is None:
        print(f"[GPT] Initializing RAG chain (model={model})...")
        try:
            _rag_chain = create_rag_chain(model=model)
        except Exception as e:
            return f"[GPT ERROR] Failed to initialize RAG: {e}"

    try:
        result = _rag_chain.invoke(question)
        text = result if isinstance(result, str) else str(result)
        return text.strip() if text and text.strip() else "[GPT ERROR] Empty RAG response."
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"[GPT ERROR] {e}"
