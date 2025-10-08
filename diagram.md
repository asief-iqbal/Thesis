graph TD
    %% Foundational Setup
    A[Dataset Curation<br/>& Complexity Scoring] --> B[Base Model<br/>Llama-3.2-1B Setup]
    
    %% Dual Path
    B --> C{Methodology}
    
    %% Static Baseline
    C -->|Static Baseline| D[Measure Unpruned<br/>Performance]
    
    %% Dynamic Framework
    C -->|Dynamic Framework| E[Initialize DDQN<br/>Agent]
    E --> F[State Vector:<br/>Hardware + Complexity]
    F --> G[Training Loop]
    
    %% Core RL Loop
    G --> H[Action Selection<br/>ε-greedy]
    H --> I[Apply Pruning<br/>Heads/Layers 5-15%]
    I --> J[Measure Performance<br/>Speed vs Quality]
    J --> K[Reward Calculation<br/>α×Speed - β×PPL]
    K --> L[Experience Replay<br/>DDQN Training]
    L --> M[Model Restoration]
    M -->|Continue Training| G
    M -->|Training Complete| N[Test Evaluation<br/>300 Episodes]
    
    %% Final Analysis
    D --> O[Performance<br/>Comparison]
    N --> O
    O --> P[Results Analysis<br/>& Publication]
    
    %% Styling
    classDef setup fill:#87CEEB,stroke:#333,stroke-width:2px
    classDef baseline fill:#DDA0DD,stroke:#333,stroke-width:2px
    classDef rl fill:#98FB98,stroke:#333,stroke-width:2px
    classDef analysis fill:#FFB6C1,stroke:#333,stroke-width:2px
    
    class A,B setup
    class D baseline
    class E,F,G,H,I,J,K,L,M,N rl
    class O,P analysis
