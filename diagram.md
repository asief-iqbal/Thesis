# RL Controller State Vector and Double DQN Diagram

This diagram illustrates the flow of the 7-dimensional state vector through the RL controller's Double DQN system in CASRAP.

```mermaid
graph TD
    A[State Vector Input\n1. CPU Utilization (0-1)\n2. Memory Available GB (/16)\n3. Battery % (0-1)\n4. GPU Available (0/1)\n5. GPU Memory Free GB (/24)\n6. GPU Utilization (0-1)\n7. Complexity Score (0-1)] --> B[Policy Net (DQN)\nSelects Action via ε-greedy]
    B --> C{ε < random()}
    C -->|Exploration| D[Random Action\nfrom Action Space]
    C -->|Exploitation| E[Best Action\nargmax(Q-values)]
    D --> F[Pruning Action\n(Target, Intensity)]
    E --> F
    F --> G[Apply Pruning\nto Model Engine]
    G --> H[Benchmark\nTokens/sec & PPL]
    H --> I[Calculate Reward\nα * Speed Bonus + β * PPL Penalty]
    I --> J[Store Transition\n(State, Action, Reward, Next State)]
    J --> K[Train Step\n(if buffer full)]
    K --> L[Policy Net\nCompute Current Q(s,a)]
    L --> M[Target Net\nCompute Target Q(s',a')\nwhere a' = Policy Net argmax]
    M --> N[Compute Loss\nMSE(Q_current, Target)]
    N --> O[Backprop & Update\nPolicy Net Weights]
    O --> P[Periodic Update\n(every 200 steps)]
    P --> Q[Copy Policy Net\nto Target Net]
    Q --> R[Stable Targets\nfor Next Train Step]
    R --> K
```

## Explanation
- **State Vector**: 7 normalized features from hardware telemetry (CPU, memory, battery, GPU) and prompt complexity (token length + perplexity).
- **Policy Net**: Learns and selects actions; updated on every training step.
- **Target Net**: Provides stable Q-value estimates for Bellman targets; updated periodically to match Policy Net.
- **Double DQN**: Policy Net chooses the action for next state, Target Net evaluates it, reducing overestimation bias.
