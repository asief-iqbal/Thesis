flowchart TD
    %% Styling
    classDef foundational fill:#87CEEB,stroke:#333,stroke-width:2px,color:#000
    classDef baseline fill:#DDA0DD,stroke:#333,stroke-width:2px,color:#000
    classDef dynamic fill:#98FB98,stroke:#333,stroke-width:2px,color:#000
    classDef decision fill:#FFE4B5,stroke:#333,stroke-width:3px,color:#000
    classDef process fill:#F0E68C,stroke:#333,stroke-width:2px,color:#000
    
    %% Foundational Setup
    A[Custom Prompt Dataset<br/>Curation] --> B[Prompt Complexity<br/>Scoring from Dataset]
    B --> C[Hardware State<br/>Monitoring Setup]
    C --> D[Base Model Selection<br/>Llama-3.2-1B]
    D --> E[Initial Model<br/>Warmup & Validation]
    
    %% Decision Point
    E --> F{Evaluation Path}
    
    %% Path A: Static Baseline
    F -->|Path A: Static Baseline| G[Load Unpruned<br/>Base Model]
    G --> H[Measure Baseline Metrics<br/>Time, PPL, Token Speed]
    H --> I[Store Baseline<br/>Performance Data]
    
    %% Path B: Dynamic Framework Setup
    F -->|Path B: Dynamic Framework| J[Initialize DDQN Agent<br/>Policy + Target Networks]
    J --> K[State Vector Construction<br/>7D: Hardware + Complexity]
    K --> L[Experience Replay<br/>Buffer Setup]
    
    %% Training Loop
    L --> M[RL Training Loop]
    M --> N[Hardware State<br/>Monitoring]
    N --> O[Action Selection<br/>ε-greedy Strategy]
    O --> P{Pruning Action}
    
    %% Action Types
    P -->|5-15%| Q[Attention Head<br/>Structured Pruning]
    P -->|5-10%| R[Transformer Layer<br/>Skipping]
    P -->|Baseline| S[No Pruning<br/>Full Model]
    
    %% Performance Measurement
    Q --> T[Apply Dynamic Pruning<br/>& GPU Synchronization]
    R --> T
    S --> T
    T --> U[Measure Pruned<br/>Performance]
    U --> V[Calculate Reward Function<br/>α×Speed - β×Quality]
    
    %% Penalty Logic
    V --> W{Performance Check}
    W -->|>10% Slowdown| X[Apply -20 Penalty<br/>Early Termination]
    W -->|Acceptable| Y[Store Experience<br/>in Replay Buffer]
    
    %% Model Management
    X --> Z[Model Restoration<br/>& Memory Cleanup]
    Y --> AA[DDQN Training<br/>Experience Replay]
    AA --> BB[Target Network<br/>Update Every 200 Steps]
    BB --> Z
    
    %% Loop Control
    Z --> CC{Training Complete?}
    CC -->|No| M
    CC -->|Yes| DD[Final Model<br/>Evaluation]
    
    %% Final Evaluation
    I --> EE[Performance Comparison<br/>Analysis]
    DD --> EE
    EE --> FF[Test Set Evaluation<br/>300 Episodes]
    FF --> GG[Results Analysis<br/>& Convergence Study]
    
    %% Apply Styles
    class A,B,C,D,E foundational
    class G,H,I baseline
    class J,K,L,M,N,O,Q,R,S,T,U,V,Y,AA,BB,DD,FF,GG dynamic
    class F,P,W,CC decision
    class X,Z process
