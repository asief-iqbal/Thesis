flowchart TD
    %% Foundational Setup
    A[Dataset Curation & Complexity Scoring] --> B[Base Model: Llama‑3.2‑1B Setup]
    B --> C{Methodology}

    %% Static Baseline
    C -->|Static Baseline| D[Measure Unpruned Performance]

    %% Dynamic Framework
    C -->|Dynamic Framework| E[Initialize DDQN Agent]
    E --> F[State Vector: Hardware + Complexity]
    F --> G[Training Loop]

    %% Core RL Loop
    G --> H[Action Selection (epsilon‑greedy)]
    H --> I[Apply Pruning: Heads 5–15% / Layers 5–10%]
    I --> J[Measure Performance: Speed & PPL]
    J --> K[Reward: alpha*speed − beta*PPL]
    K --> L[Experience Replay Training]
    L --> M[Model Restoration]
    M -->|Continue Training| G
    M -->|Training Complete| N[Test Evaluation: 300 Episodes]

    %% Final Analysis
    D --> O[Baseline vs Pruned Comparison]
    N --> O
    O --> P[Results Analysis & Reporting]

    %% Styling
    classDef setup fill:#87CEEB,stroke:#333,stroke-width:2px
    classDef baseline fill:#DDA0DD,stroke:#333,stroke-width:2px
    classDef rl fill:#98FB98,stroke:#333,stroke-width:2px
    classDef analysis fill:#FFB6C1,stroke:#333,stroke-width:2px

    class A,B setup
    class D baseline
    class E,F,G,H,I,J,K,L,M,N rl
    class O,P analysis

