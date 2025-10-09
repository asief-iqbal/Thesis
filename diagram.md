# RL Controller State Vector and Double DQN Diagram

This diagram illustrates the flow of the 7-dimensional state vector through the RL controller's Double DQN system in CASRAP.

```mermaid
graph TD
    A[State Vector Input<br/>1. CPU Utilization (0-1)<br/>2. Memory Available GB (/16)<br/>3. Battery % (0-1)<br/>4. GPU Available (0/1)<br/>5. GPU Memory Free GB (/24)<br/>6. GPU Utilization (0-1)<br/>7. Complexity Score (0-1)] --> B[Policy Net (DQN)<br/>Selects Action via ε-greedy]
    B --> C{ε < random()}
    C -->|Exploration| D[Random Action<br/>from Action Space]
    C -->|Exploitation| E[Best Action<br/>argmax(Q-values)]
    D --> F[Pruning Action<br/>(Target, Intensity)]
    E --> F
    F --> G[Apply Pruning<br/>to Model Engine]
    G --> H[Benchmark<br/>Tokens/sec & PPL]
    H --> I[Calculate Reward<br/>α * Speed Bonus + β * PPL Penalty]
    I --> J[Store Transition<br/>(State, Action, Reward, Next State)]
    J --> K[Train Step<br/>(if buffer full)]
    K --> L[Policy Net<br/>Compute Current Q(s,a)]
    L --> M[Target Net<br/>Compute Target Q(s',a')<br/>where a' = Policy Net argmax]
    M --> N[Compute Loss<br/>MSE(Q_current, Target)]
    N --> O[Backprop & Update<br/>Policy Net Weights]
    O --> P[Periodic Update<br/>(every 200 steps)]
    P --> Q[Copy Policy Net<br/>to Target Net]
    Q --> R[Stable Targets<br/>for Next Train Step]
    R --> K
```

## Explanation
- **State Vector**: 7 normalized features from hardware telemetry (CPU, memory, battery, GPU) and prompt complexity (token length + perplexity).
- **Policy Net**: Learns and selects actions; updated on every training step.
- **Target Net**: Provides stable Q-value estimates for Bellman targets; updated periodically to match Policy Net.
- **Double DQN**: Policy Net chooses the action for next state, Target Net evaluates it, reducing overestimation bias.

