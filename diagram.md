%% RL Controller State Vector and Double DQN Diagram
graph TD
  A["State Vector Input<br/>1. CPU Utilization (0-1)<br/>2. Memory Available GB (/16)<br/>3. Battery (0-1)<br/>4. GPU Available (0/1)<br/>5. GPU Memory Free GB (/24)<br/>6. GPU Utilization (0-1)<br/>7. Complexity Score (0-1)"] --> B["Policy Net (DQN)<br/>Selects Action via ε-greedy"]
  B --> C{ε &lt; random()}
  C -->|Exploration| D["Random Action<br/>from Action Space"]
  C -->|Exploitation| E["Best Action<br/>argmax(Q-values)"]
  D --> F["Pruning Action<br/>(Target, Intensity)"]
  E --> F
  F --> G["Apply Pruning<br/>to Model Engine"]
  G --> H["Benchmark<br/>Tokens/sec &amp; PPL"]
  H --> I["Calculate Reward<br/>α * Speed Bonus + β * PPL Penalty"]
  I --> J["Store Transition<br/>(State, Action, Reward, Next State)"]
  J --> K["Train Step<br/>(if buffer full)"]
  K --> L["Policy Net<br/>Compute Current Q(s,a)"]
  L --> M["Target Net<br/>Compute Target Q(s',a')<br/>where a' = Policy Net argmax"]
  M --> N["Compute Loss<br/>MSE(Q_current, Target)"]
  N --> O["Backprop & Update<br/>Policy Net Weights"]
  O --> P["Periodic Update<br/>(every 200 steps)"]
  P --> Q["Copy Policy Net<br/>to Target Net"]
  Q --> R["Stable Targets<br/>for Next Train Step"]
  R --> K

