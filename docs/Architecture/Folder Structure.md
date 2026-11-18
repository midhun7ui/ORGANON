
ORGANON/
│
├── organon/                         # Core Python package
│   │
│   ├── __init__.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   ├── pipeline.py
│   │   ├── registry.py
│   │   └── exceptions.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loaders/
│   │   ├── preprocessors/
│   │   ├── chunkers/
│   │   └── indexing.py
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── providers/
│   │
│   ├── vectorstores/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── providers/
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── retriever.py
│   │   ├── hybrid.py
│   │   ├── rerankers/
│   │   └── context.py
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── providers/
│   │
│   ├── security/
│   ├── evaluation/
│   ├── observability/
│   ├── cache/
│   ├── api/
│   └── config/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
│
├── examples/
│
├── docs/
│   │
│   ├── assets/
│   │   └── Visual Architecture.jpeg
│   │
│   ├── architecture/
│   │   ├── overview.md
│   │   ├── system-design.md
│   │   └── data-flow.md
│   │
│   ├── getting-started/
│   │   ├── installation.md
│   │   ├── quickstart.md
│   │   └── configuration.md
│   │
│   ├── concepts/
│   │   ├── ingestion.md
│   │   ├── embeddings.md
│   │   ├── vector-stores.md
│   │   ├── retrieval.md
│   │   ├── generation.md
│   │   └── providers.md
│   │
│   ├── security/
│   │   ├── overview.md
│   │   └── multi-tenancy.md
│   │
│   ├── deployment/
│   │   ├── docker.md
│   │   └── production.md
│   │
│   ├── development/
│   │   ├── contributing.md
│   │   ├── testing.md
│   │   └── provider-development.md
│   │
│   └── api-reference/
│       └── overview.md
│
├── scripts/
├── docker/
│
├── .env.example
├── .gitignore
├── config.yaml
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── LICENSE
├── CONTRIBUTING.md
├── CHANGELOG.md
└── README.md
