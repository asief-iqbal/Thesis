# CASRAP Diagram Set

This file stores the synchronized Mermaid diagrams for the current repository architecture. These diagrams match the updated methodology and README.

## 1. End-to-End System Diagram

```mermaid
flowchart LR
    A[Benchmark mixture<br/>GSM8K MBPP WikiText-2 MMLU BoolQ] --> B[Audit and cleaning<br/>JSON audit reports]
    B --> C[Oracle labeling<br/>dense vs sparse loss gap]
    C --> D[LCR training<br/>BERT-mini router]
    D --> E[Runtime controller]

    subgraph Runtime [Runtime adaptive pruning loop]
        F[Prompt] --> G[Dense Llama-3.2-1B baseline]
        F --> H[LCR scorer<br/>BERT-mini + auxiliary + attention stats]
        G --> I[Early-Llama signals]
        J[Hardware telemetry] --> K[10D DDQN state]
        H --> K
        I --> K
        K --> L[DDQN action selection<br/>15 actions]
        L --> M[Pruning engine]
        M --> N[GQA-safe head pruning<br/>or reversible layer skipping]
        N --> O[Pruned inference]
        G --> P[Reward computation<br/>throughput and continuation PPL]
        O --> P
        P --> Q[Replay update + model restoration]
    end

    E --> Runtime
```

## 2. Runtime Controller Diagram

```mermaid
flowchart TD
    A[State inputs] --> B[Hardware telemetry<br/>CPU RAM battery GPU VRAM util]
    A --> C[LCR score]
    A --> D[Early-Llama features<br/>hidden norm entropy concentration]
    B --> E[10-dimensional state vector]
    C --> E
    D --> E
    E --> F[Policy network<br/>10 -> 128 -> 128 -> 15]
    F --> G{epsilon-greedy}
    G -->|explore| H[Random action]
    G -->|exploit| I[argmax Q action]
    H --> J[Apply pruning action]
    I --> J
    J --> K[Benchmark pruned run]
    K --> L[Compute reward]
    L --> M[Store transition]
    M --> N[Replay training]
    N --> O[Target sync every 200 steps]
```

## 3. LCR Architecture Diagram

```mermaid
flowchart TD
    A[Input prompt] --> B[BERT-mini encoder<br/>4 layers 256 hidden]
    B --> C[ScalarMix over embedding and hidden states]
    C --> D[Mean pooling<br/>256-d representation]

    A --> E[Auxiliary text features<br/>9 prompt statistics]
    B --> F[Attention statistics<br/>48 features]
    E --> G[Concatenate aux + attention stats<br/>57-d vector]
    F --> G
    G --> H[AuxProjector<br/>57 -> 48]
    D --> I[Fusion<br/>256 + 48 = 304]
    H --> I
    I --> J[Regressor head<br/>304 -> 202 -> 101 -> 1]
    J --> K[Sigmoid sensitivity score<br/>0 to 1]
```
