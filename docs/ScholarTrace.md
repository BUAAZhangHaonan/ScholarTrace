# Recommendation report

## Executive recommendation

I would build this as a **layered local stack**, not as a single paper-search product. The strongest core I found is:

- **Metadata spine:** OpenAlex + DBLP + arXiv + OpenReview. OpenAlex gives you a fully open, snapshotable scholarly graph; DBLP gives you much better CS venue and author normalization; arXiv remains mandatory for AI preprints and source/PDF access; OpenReview is the only source that cleanly exposes the modern ML conference submission/review world. ([OpenAlex Developers](https://developers.openalex.org/api-reference/introduction))
- **Open-access and citation enrichment:** Crossref + Unpaywall + CORE + OpenCitations. Crossref is the DOI and publisher-metadata workhorse; Unpaywall and CORE are practical open-access resolvers; OpenCitations is the cleanest open citation edge layer. ([www.crossref.org](https://www.crossref.org/documentation/retrieve-metadata/rest-api/))
- **Parsing:** GROBID + Docling as the default pair, with Marker only when GPLv3 is acceptable. GROBID is still the safest scholarly-structure extractor; Docling is the best general local document conversion layer I found. ([GROBID](https://grobid.readthedocs.io/en/latest/Introduction/))
- **Search:** OpenSearch + Qdrant by default. Use Vespa only if ranking itself is the core product and you are willing to accept more schema and ops complexity. ([GitHub](https://github.com/opensearch-project/opensearch))
- **Local AI services:** vLLM for generation on NVIDIA boxes, TEI for embeddings and rerankers, Ollama or llama.cpp when simplicity or Mac/CPU deployment matters more than peak throughput. ([vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/))
- **Human-facing library:** Zotero + Better BibTeX. JabRef is the right alternative only if your center of gravity is already `.bib`-native. ([Zotero](https://www.zotero.org/support/zotero_data))

This stack is the best balance of **fully local deployment, open licensing, modularity, CS/AI coverage, and maintenance health** that I found. The key reason is simple: the core data sources all support local-friendly harvesting, and the core runtime pieces already expose normal interfaces such as REST, OpenAPI, OpenAI-compatible APIs, or JSON-RPC/MCP surfaces. ([OpenAlex Developers](https://developers.openalex.org/download/download-to-machine))

------

## Landscape overview

A good local academic retrieval system has seven separate layers: **source harvesting**, **artifact storage**, **document parsing**, **search indexes**, **graph/index analytics**, **LLM-facing interaction**, and **human library management**. The open ecosystem is strongest in the first four layers; it is weaker in polished local citation-graph UX and in legally clean, high-coverage full-text acquisition across closed publishers. That is why the right architecture is modular: sources and licenses change at different speeds, and your parsing, indexing, and agent layers should not be tightly coupled to one vendor or one dataset. ([OpenAlex Developers](https://developers.openalex.org/download/download-to-machine))

Also plan storage early. OpenAlex’s own download guide says a full snapshot currently takes about **330 GB**, and Crossref’s public data file documentation says the **2025** file is about **200 GB**. That alone is enough reason to design for incremental sync, append-only raw storage, and reproducible snapshot imports instead of periodic full rebuilds. ([OpenAlex Developers](https://developers.openalex.org/download/download-to-machine))

------

## 1. Paper discovery and metadata sources

### The sources I would build around

**OpenAlex** should be your primary open metadata backbone. Its API covers works, authors, sources, institutions, topics, and more; the snapshot lives in S3 as gzip-compressed JSON Lines; the full snapshot is free to download anonymously; and the AWS registry lists the dataset as **CC0**. For a local system, that combination is unusually strong: breadth, legal clarity, and a full offline copy. The downside is that for CS-specific venue/person curation it is not as sharp as DBLP, and it is not itself your full-text solution. ([OpenAlex Developers](https://developers.openalex.org/api-reference/introduction))

**DBLP** is still the mandatory CS supplement. It exposes separate publication/author/venue search APIs, publishes the full XML dump, and now also provides daily RDF dumps and a SPARQL service over the dblp knowledge graph enhanced with open citation data. That makes DBLP the best source for CS venue identity, author disambiguation, and proceedings-centric normalization. Its weakness is scope: it is much narrower than OpenAlex and not a full-text provider. ([DBLP](https://dblp.org/faq/How%2Bto%2Buse%2Bthe%2Bdblp%2Bsearch%2BAPI))

**arXiv** remains non-negotiable for AI. arXiv exposes a public API, an OAI-PMH interface with nightly metadata updates, and bulk full-text access paths, including processed PDFs and source files via S3. It is also one of the few places where preserving version history is practical: the OAI docs explicitly distinguish the latest-version metadata from the `arXivRaw` history-oriented format. Its weakness is obvious: it covers preprints, not the whole literature. ([arXiv Info](https://info.arxiv.org/help/api/index.html))

**OpenReview** is the missing piece in most general scholarly systems, and it matters disproportionately in AI. The docs and official Python client make it clear that you can retrieve venue notes, submissions, replies, reviews, rebuttals, decisions, and attachments through the API. For ICLR/NeurIPS-style workflows, no other source captures the review-side graph this cleanly. The downside is source heterogeneity: the data model is invitation- and venue-driven, so you should expect venue-specific adapters. ([OpenReview Documentation](https://docs.openreview.net/))

### The enrichers I would add immediately

**Crossref** is the DOI normalizer and publisher-metadata backbone you still need. Its REST API requires no signup, exposes rich metadata around works and journals, and its public data file provides a bulk path when you need local mirrors. It is excellent for DOI reconciliation, venue metadata, funder IDs, ORCID/ROR enrichment, and general publisher-side cleanup. It is not a blanket full-text solution. ([www.crossref.org](https://www.crossref.org/documentation/retrieve-metadata/rest-api/))

**Unpaywall** and **CORE** are the most practical open-access acquisition layers to add on top. Unpaywall is the canonical OA lookup layer around DOI-based works, and CORE aggregates and enriches metadata and full text from a large network of repositories and OA journals via API access. In practice, I would use Unpaywall first for OA resolution and CORE as a second OA/full-text route when you need broader repository harvesting. ([Unpaywall](https://unpaywall.org/products/api?utm_source=chatgpt.com))

**OpenCitations** is the cleanest open citation dataset I found. Its site states that the data is released under **CC0**, it exposes APIs, and the January 2026 dumps include large-scale bibliographic and citation metadata. I would ingest it even if OpenAlex is already present, because open citation edges are valuable enough to justify an explicit local layer. ([OpenCitations](https://opencitations.net/))

### Useful, but not where I would anchor the system

**Semantic Scholar** is useful but should not be the backbone of a local-first stack. The API is good, the datasets are real, and it is operationally polished. The problem is policy: the official API page says the introductory API-key rate is **1 request per second**, and the dataset license limits default use to **internal, non-commercial research and educational purposes** unless you negotiate broader terms. That is fine for enrichment, bad for a base layer you want to expose freely through your own local services. ([Semantic Scholar](https://www.semanticscholar.org/product/api))

For expansion beyond core CS/AI, the best additions are **Lens** for scholarly-plus-patent linkage, **OpenAIRE** for EU/open-repository graph coverage, and **NCBI/PubMed/PMC** when you drift into bio/biomed adjacent work. Lens is the cleanest patent crossover source in this set because its API explicitly exposes both scholarly and patent request/response schemas. ([Lens API Documentation](https://docs.api.lens.org/))

**Verdict for this layer:** build around **OpenAlex + DBLP + arXiv + OpenReview**, add **Crossref + Unpaywall + CORE + OpenCitations**, and treat **Semantic Scholar** as an optional enrichment source rather than a dependency you cannot live without. ([OpenAlex Developers](https://developers.openalex.org/api-reference/introduction))

------

## 2. Full-text retrieval, parsing, and local storage

The right full-text policy is: **store raw artifacts first, parse second, never trust one parser as ground truth**. Keep the original PDF/HTML/LaTeX/source tarball, then build one or more parsed representations beside it. That preserves reparseability when parser quality or licensing changes, which it will. This matters because scholarly PDFs are still messy, and the ecosystem does not offer a single perfect parser. ([GROBID](https://grobid.readthedocs.io/en/latest/Introduction/))

**GROBID** is still the default scholarly parser I would trust most. Its docs describe it as a machine-learning library for extracting and restructuring raw documents into structured XML/TEI with a particular focus on technical and scientific publications, and the project has been steadily maintained since becoming open source in 2011. That combination of TEI output, bibliography parsing, and scholarly focus still makes it the safest base parser for papers. ([GROBID](https://grobid.readthedocs.io/en/latest/Introduction/))

**Docling** is the best companion parser in this stack. The project is very broad in document coverage, handles PDF layout, reading order, tables, formulas, code, images, and more, and uses a unified internal representation. Its MIT license and very active 2026 release cadence make it especially attractive for a local infrastructure project that may evolve into shared lab tooling. ([GitHub](https://github.com/docling-project/docling))

**Marker** is excellent when your downstream system wants markdown or chunked JSON fast. The repository emphasizes markdown/JSON/chunk/HTML output, support for tables, equations, references, images, and operation across GPU, CPU, or MPS. I still would not make it the universal default in a shared stack, because the project is GPLv3, and that license choice becomes a real design constraint the moment you start distributing integrated services. ([GitHub](https://github.com/datalab-to/marker))

**PyMuPDF** is a strong utility layer for fast text extraction, page geometry, splitting, and preprocessing, but the project is AGPL/commercial. I would use it as a sharp tool where needed, not as the central parsing contract for a lab platform unless that license is already acceptable. **Nougat** is worth keeping as a fallback path for OCR-like academic documents and difficult scans, not as your main parser. ([GitHub](https://github.com/pymupdf/PyMuPDF))

I would **not** anchor a new system on **CERMINE** or **science-parse**. CERMINE is AGPL and its issue tracker shows long-standing low-activity and build/dependency problems. science-parse exists, but it is not where I would place a fresh core dependency in 2026. ([GitHub](https://github.com/CeON/CERMINE))

For storage, I would keep four physical layers:

```text
raw_artifacts/     PDFs, source tarballs, HTML, LaTeX, supplementary files
parsed_docs/       TEI, markdown, JSON chunks, bbox/layout output
postgres/          canonical works, authors, venues, IDs, artifact manifests
parquet_lake/      snapshot imports, analytics tables, graph edges, audit logs
```

Use a simple filesystem first; switch to MinIO only when you need multi-machine object semantics. Put **canonical IDs and alias mapping** at the center: DOI, arXiv ID, OpenAlex ID, dblp key, OpenReview forum ID, and any local hash should all map to one internal work identity.

------

## 3. Local indexing and search

My default recommendation is **OpenSearch + Qdrant**, not an all-in-one vector database. OpenSearch gives you mature lexical retrieval, filtering, aggregations, dashboards, REST APIs, vector fields, and explicit hybrid-search pipelines. Qdrant gives you a clean vector service with simple local modes, easy self-hosting, hybrid-query support, and a very low-friction developer experience for dense retrieval. That split is easier to reason about, easier to swap, and easier to debug than a single opaque “AI search” box. ([GitHub](https://github.com/opensearch-project/opensearch))

The concrete search layout I would use is straightforward: put **title/abstract/venue/author/year/topic/full text** into OpenSearch for BM25, filters, and facets; put **document and chunk embeddings** into Qdrant; then rerank the union with a local cross-encoder or LLM ranker. Keep both a **document-level index** and a **chunk-level index**. Papers need both: document retrieval finds the right work, chunk retrieval finds the right claim, paragraph, theorem, or experimental detail.

If your lab cares deeply about ranking research and you want one serving system for lexical, vector, tensor, and learned-ranking logic, **Vespa** is the higher-ceiling alternative. Its docs and README show first-class BM25+embedding hybrid retrieval, self-hosting, active releases, and a design centered on low-latency ranking at scale. The trade-off is complexity: Vespa is powerful, but it is not the cheapest place to learn by doing. ([Vespa Documentation](https://docs.vespa.ai/en/learn/tutorials/hybrid-search.html))

The secondary options are all real, but I would rank them lower for this particular job:

- **pgvector** is good when you want vectors in the same database as the rest of your application data. It supports exact and ANN search, sparse vectors, and normal Postgres joins. I still would not use it as the main search tier for a serious literature system once you want rich BM25, faceting, reranking, and retrieval experimentation. ([GitHub](https://github.com/pgvector/pgvector))
- **Weaviate** is capable and self-hostable, with semantic and hybrid search, but it brings a broader platform surface than you need if you are already comfortable composing search services yourself. ([Weaviate Documentation](https://docs.weaviate.io/weaviate))
- **Milvus** is strong at scale and supports standalone deployment, but it is heavier and more cloud-native in posture than I would choose for a first local scholarly system. ([GitHub](https://github.com/milvus-io/milvus))
- **LanceDB** is attractive for embedded local prototyping and single-user experiments. For a durable lab service with rich filters and nontrivial search UX, I would still rather stand up OpenSearch or Qdrant+OpenSearch. ([LanceDB](https://lancedb.github.io/lancedb/python/python/))

**Verdict for this layer:** start with **OpenSearch + Qdrant**, and move to **Vespa** only when search quality engineering becomes the main product, not just a subsystem. ([GitHub](https://github.com/opensearch-project/opensearch))

------

## 4. AI-assisted analysis and interaction

### Local model serving

For local generation, **vLLM** is the best default on serious NVIDIA hardware. It exposes an OpenAI-compatible server and is one of the most active open-source serving projects in the space. In a lab setting, that matters because once other tools speak “OpenAI-compatible,” your orchestration layer stays simple. ([vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/))

For simpler workstation deployment, **Ollama** is hard to beat. Its API runs locally by default on `localhost:11434`, and the operational model is trivial compared with heavier GPU-serving stacks. I would choose it for a personal box, a small Mac setup, or fast iteration—not for peak throughput. ([Ollama](https://docs.ollama.com/api/introduction))

For GGUF-heavy, Mac-friendly, or CPU-first deployment, **llama.cpp** and **llama-cpp-python** are the right tools. Both expose OpenAI-compatible servers, and llama.cpp’s server now covers chat, responses, embeddings, and reranking routes. This is the cleanest way to keep a local endpoint alive on commodity hardware without turning model serving into a mini-platform team. ([Llama CPP Python](https://llama-cpp-python.readthedocs.io/en/latest/server/))

For embeddings and reranking, use **Text Embeddings Inference (TEI)** as a dedicated service. The project explicitly supports text embeddings, rerankers, sparse SPLADE pooling, OpenAPI docs, air-gapped deployment, and even Metal support for local Macs. That is exactly what you want in a literature stack: a separate, swappable embedding/reranker service rather than a hidden helper inside the vector DB. ([GitHub](https://github.com/huggingface/text-embeddings-inference))

### Paper-facing workflows

**PaperQA2** is the most paper-native open system I found. Its repo describes it as high-accuracy RAG for scientific documents with citations, focused on PDFs and other document types. If you want a local “ask questions across papers and get cited answers” layer, this is the first thing I would try before writing my own paper agent. ([GitHub](https://github.com/future-house/paper-qa))

**Haystack** is the best general orchestration framework in this landscape. Its public positioning is clear: modular, production-ready pipelines with explicit control over retrieval, routing, memory, generation, and agents. That makes it a better systems foundation than most paper-chat demos. ([GitHub](https://github.com/deepset-ai/haystack))

**ASReview** is the right tool when the task is screening, triage, or systematic-review style prioritization rather than free-form conversation. The project is explicit about active learning, human-in-the-loop screening, transparency, and privacy-first local use. That fills an important gap most paper agents do not handle well. ([GitHub](https://github.com/asreview/asreview))

**LlamaIndex** is viable, but I would not make it the foundation of a strict local-first scholarly stack. The OSS project is real, but its current messaging is tightly coupled to **LlamaParse** as an enterprise parsing/OCR platform. If you are already choosing local parsers such as GROBID and Docling, Haystack or direct service composition is cleaner. ([GitHub](https://github.com/run-llama/llama_index))

**Verdict for this layer:** make “paper interaction” a thin layer above your own retrieval APIs. Use **vLLM/Ollama/llama.cpp + TEI** for serving, **PaperQA2** for paper-grounded QA, **Haystack** for orchestration, and **ASReview** when the task is screening rather than chatting. ([vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/))

------

## 5. Citation and knowledge graph navigation

The raw ingredients for a strong local citation graph now exist, but the polished product layer is still weak. OpenAlex exposes citation-related fields such as `referenced_works`, `cited_by_api_url`, and `related_works`; OpenCitations provides open CC0 citation data and dumps; DBLP’s knowledge graph and SPARQL service already integrate open citation data. That is enough to build your own local citation and author graph with very good CS coverage. ([OpenAlex Developers](https://developers.openalex.org/guides/recipes))

I would **not** start with a graph database. I would start with **Parquet/DuckDB + NetworkX** for offline graph construction, neighbor expansion, coauthor statistics, venue overlap, citation trails, and batch analytics. DuckDB is extremely well suited to file-backed analytics over Parquet and other local files, and NetworkX gives you a huge surface of graph algorithms without operational baggage. ([DuckDB](https://duckdb.org/))

Move to **Neo4j** only when graph traversal or graph-backed interaction becomes a first-class product feature. Neo4j’s docs and first-party **GraphRAG for Python** package make it the best graph-DB choice once you truly need an interactive graph service, graph UX, or graph-grounded retrieval. Before that point, it is usually more machinery than value. ([Graph Database & Analytics](https://neo4j.com/docs/neo4j-graphrag-python/current/))

------

## 6. Personal library management

For the human-facing layer, **Zotero** is still the best answer. It stores data locally by default, sync is optional, unlimited local files are allowed even without paid storage, and the platform exposes both a Web API and a local JavaScript API. That gives you the rare combination of good end-user UX and enough programmatic surface to integrate into a custom local stack. ([Zotero](https://www.zotero.org/support/zotero_data))

**Better BibTeX** is the obvious companion if you write in LaTeX, Markdown, or any text-first workflow. It is MIT-licensed, active, and current releases are compatible with Zotero 8 and Zotero 9 beta. In practice, this is the simplest way to keep your human library and your build system aligned without constant citation-key churn. ([GitHub](https://github.com/retorquere/zotero-better-bibtex))

**JabRef** is the right alternative when `.bib` is already your source of truth. It is active, open source, supports cite-as-you-write, and recent releases added OpenAlex/OpenCitations integration plus more REST and HTTP-server functionality. I would still choose Zotero for the broader plugin and annotation ecosystem unless your workflow is already deeply BibTeX-native. ([GitHub](https://github.com/JabRef/jabref))

One hard rule here: **do not treat `zotero.sqlite` as your integration contract**. Zotero’s own local API docs explicitly say that direct SQLite access is much more fragile than using the local JavaScript API. Use Zotero as the human UI and sync/export bridge, not as the computational ground truth for your retrieval system. ([Zotero](https://www.zotero.org/support/dev/client_coding/javascript_api))

------

## 7. Integration blueprint

### Blueprint A — the one I would actually build first

This is the best default for a serious local research system:

```text
Harvesters
  OpenAlex + DBLP + arXiv + OpenReview + Crossref + Unpaywall + CORE + OpenCitations
      ↓
Canonical metadata store
  Postgres (works/authors/venues/IDs/artifacts/edges)
      ↓
Raw artifact store
  filesystem or MinIO (PDF, source, HTML, supplements)
      ↓
Parser layer
  GROBID + Docling (+ Marker optional)
      ↓
Search layer
  OpenSearch (lexical/filter/facet) + Qdrant (dense/chunk vectors) + TEI reranker
      ↓
Interaction layer
  FastAPI + MCP server + Typer CLI + vLLM/Ollama/llama.cpp + PaperQA2/Haystack
      ↓
Human layer
  Zotero + Better BibTeX
```

This composition is concrete, locally runnable, and cleanly separable. Every core component here either exposes a documented API or can be wrapped behind one: OpenSearch is RESTful, Qdrant has local and server modes, vLLM is OpenAI-compatible, Ollama is localhost HTTP by default, llama-cpp-python exposes an OpenAI-compatible server, FastAPI gives you your own API front door, and MCP gives you a standard tool surface for LLM clients. ([GitHub](https://github.com/opensearch-project/opensearch))

The one schema detail that matters more than people expect is **canonical identity**. I would define at least these internal entities:

```text
Work(id, doi, arxiv_id, openalex_id, dblp_key, openreview_forum_id, ...)
Artifact(id, work_id, kind, source_url, license, sha256, local_path, ...)
Chunk(id, artifact_id, section_title, text, citation_spans, bbox, embedding_id)
Edge(src_work_id, dst_work_id, type)   # cites, version_of, same_as, reviewed_at, author_of
SourceEvent(source, cursor, fetched_at, checksum)
```

That alias layer is what lets you deduplicate “the same paper” across OpenAlex, DBLP, arXiv, OpenReview, Crossref, and your local files without turning every query into a fuzzy join problem.

### Blueprint B — the higher-ceiling unified serving stack

Keep the same harvesting, storage, and parser layers, but replace OpenSearch + Qdrant with **Vespa**. This is the stack to choose when you care about custom ranking expressions, one-engine hybrid retrieval, large-scale serving, and search quality engineering as a first-class research problem. It is a strong choice for a shared lab service. It is not the cheap path to a first usable prototype. ([Vespa Documentation](https://docs.vespa.ai/en/learn/tutorials/hybrid-search.html))

### Blueprint C — a single-user workstation prototype

If you want a smaller first system, do this:

- Zotero + Better BibTeX for the human library. ([Zotero](https://www.zotero.org/support/sync))
- OpenAlex + DBLP + arXiv + OpenReview harvesters only. ([OpenAlex Developers](https://developers.openalex.org/api-reference/introduction))
- Docling first, GROBID second. ([GitHub](https://github.com/docling-project/docling))
- Qdrant in local-path mode or LanceDB for vectors. ([GitHub](https://github.com/qdrant/qdrant))
- Ollama or llama.cpp for local serving. ([Ollama](https://docs.ollama.com/api/introduction))

This is a good prototype. It is not where I would stop.

------

## 8. Maintenance and community snapshot as of April 2026

The healthy part of this ecosystem is the infrastructure, not the paper-chat gloss. OpenReview’s Python client was updated on **April 15, 2026**; Docling released **v2.88.0** on **April 13, 2026**; ASReview released **v3.0.4** on **April 14, 2026**; Zotero 9.0 changes are dated **April 10, 2026**; OpenSearch shows **3.6.0** on **April 7, 2026**; Vespa shows a latest release on **April 15, 2026**; and PaperQA2 released **v2026.03.18** on **March 18, 2026**. Those are healthy signs. ([GitHub](https://github.com/orgs/openreview/repositories))

The older but still trustworthy projects are **GROBID**, **DBLP**, **arXiv**, **Zotero**, and **OpenAlex**. They are not shiny, but they are infrastructure-quality and documentation-rich. For this project, that is a virtue. ([GROBID](https://grobid.readthedocs.io/en/latest/Introduction/))

The projects with real caveats are **Marker** and **PyMuPDF** because of licensing, and **Semantic Scholar** because of policy and rate limits rather than technical quality. ([GitHub](https://github.com/datalab-to/marker/blob/master/LICENSE))

The things I would **not** bet a new core system on are **CERMINE**, **science-parse**, and **Papers with Code as a live dependency**. CERMINE shows long-standing low-activity complaints and build issues, science-parse is not where the frontier is anymore, and the Papers with Code data repo has an open issue noting that the site now redirects to Hugging Face while the public archive is explicitly framed as the last publicly available snapshot. That is archival enrichment, not a stable live substrate. ([GitHub](https://github.com/CeON/CERMINE/issues))

------

## 9. Gaps and risks

- **There is still no polished open local equivalent of Connected Papers or ResearchRabbit that I would trust as a core dependency.** The data is there through OpenAlex, OpenCitations, and DBLP; the polished local graph UX mostly is not. Expect to build that layer yourself if it matters. ([OpenAlex Developers](https://developers.openalex.org/guides/recipes))
- **Full-text rights remain fragmented.** Crossref gives metadata, not blanket redistribution rights, and arXiv’s bulk data page explicitly notes that for most full-text uses you must link back rather than redistribute unless the license permits it. Local deployment helps privacy and latency; it does not dissolve licensing. ([www.crossref.org](https://www.crossref.org/documentation/retrieve-metadata/rest-api/))
- **OpenReview will force source-specific engineering.** Its note and invitation model is powerful, but venue and year conventions vary enough that “one generic parser” is the wrong mental model. ([OpenReview Documentation](https://docs.openreview.net/how-to-guides/data-retrieval-and-modification/how-to-get-all-notes-for-submissions-reviews-rebuttals-etc))
- **Parser quality is still probabilistic.** GROBID, Docling, and Marker are all strong, but complex layouts, bad scans, supplements, and tables will still produce edge cases. Build parser ensembles and confidence flags from the beginning. ([GROBID](https://grobid.readthedocs.io/en/latest/Introduction/))
- **Benchmark/code tracking is weaker than it used to be.** If you care about reproducing model–dataset–metric links, you should treat Papers with Code as an archival enrichment source and be ready to build your own benchmark/code linkage over time. ([GitHub](https://github.com/paperswithcode/paperswithcode-data/issues/116))

------

## 10. Prioritized action list

1. **Build the metadata spine first:** OpenAlex, DBLP, arXiv, OpenReview. Get canonical IDs and alias mapping correct before doing anything clever with agents. ([OpenAlex Developers](https://developers.openalex.org/api-reference/introduction))
2. **Add OA/full-text enrichment next:** Crossref, Unpaywall, CORE, OpenCitations. That immediately improves coverage without forcing you into closed-index dependencies. ([www.crossref.org](https://www.crossref.org/documentation/retrieve-metadata/rest-api/))
3. **Stand up GROBID + Docling and keep raw artifacts.** Do not wait until later to preserve raw PDFs and parser outputs separately. ([GROBID](https://grobid.readthedocs.io/en/latest/Introduction/))
4. **Deploy OpenSearch + Qdrant and implement true hybrid retrieval.** Do document-level and chunk-level indexing from day one. ([OpenSearch Documentation](https://docs.opensearch.org/latest/vector-search/ai-search/hybrid-search/index/))
5. **Expose the system through FastAPI + CLI + MCP.** Standard interfaces matter more than a fancy UI early on. ([Model Context Protocol](https://modelcontextprotocol.io/specification/draft/schema))
6. **Add local model services only after retrieval is trustworthy.** vLLM/Ollama/llama.cpp plus TEI are the right primitives; PaperQA2 and Haystack come after the search layer is real. ([vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/))
7. **Integrate Zotero once the data plane is stable.** Let Zotero serve the human workflow; do not let it define the backend architecture. ([Zotero](https://www.zotero.org/support/sync))
8. **Add Neo4j only when graph interaction becomes a real product feature.** Until then, use DuckDB/Parquet + NetworkX and stay lean. ([DuckDB](https://duckdb.org/))

The first milestone worth chasing is deliberately boring: **one canonical work table, one raw artifact store, one parser ensemble, and one hybrid retrieval API**. Once that exists, everything else—paper chat, citation trails, topic alerting, benchmark dashboards, review mining—becomes a tractable extension rather than a fragile demo layered on top of quotas and scraped pages.