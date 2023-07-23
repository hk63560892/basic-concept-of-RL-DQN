# Reinforcement Learning (RL) Notes

Reinforcement Learning is a subset of machine learning where an agent learns to behave in an environment, by performing certain actions and observing the results/rewards of those actions.

## Key Concepts of Reinforcement Learning

1. **Agent**: An agent takes actions based on the state it is in.
2. **Environment**: The world through which the agent moves. The environment takes the agent's current state and action as input, and returns the agent's reward and next state.
3. **Action (A)**: What the agent can do. The set of all actions available to the agent is called action space.
4. **State (S)**: Current situation returned by the environment.
5. **Reward (R)**: A feedback from the environment. The goal of a reinforcement learning agent is to maximize the total reward it receives over the long run.
6. **Policy (π)**: The strategy that the agent uses to determine next action based on the current state. It can be deterministic or a probability distribution.
7. **Value (V or Q)**: Expected long-term return with discount, as a function of state or state-action pair.
8. **Q-Table**: A simple lookup table where we calculate the maximum expected future rewards for action at each state.

## Reinforcement Learning Process

The process starts with the agent performing an action on the environment. The environment then returns the next state and reward for the agent. This cycle continues until the agent reaches the goal.

## Types of RL Algorithms

There are multiple types of RL algorithms, including but not limited to:

- Q-Learning
- Deep Q Learning (DQN)
- State-Action-Reward-State-Action (SARSA)
- Deep Deterministic Policy Gradient (DDPG)
- Proximal Policy Optimization (PPO)

## Markov Decision Processes in Reinforcement Learning

### Definition

A Markov Decision Process (MDP) is a mathematical model used to describe an environment for reinforcement learning. An MDP is a discrete time stochastic control process and provides a framework where an agent can take actions that influence the states of the environment, and receive rewards in return.

### Components

An MDP is defined by the following components:

- **States (S)**: A finite set of states that the agent can be in.
- **Actions (A)**: A finite set of actions that the agent can take.
- **Transition Probability (P)**: A function `P(s'|s, a)` that gives the probability of moving to state `s'` when action `a` is taken in state `s`.
- **Reward Function (R)**: A function `R(s, a, s')` that gives the expected immediate reward received after transitioning to state `s'`, from state `s`, due to action `a`.

### Policy (π)

A policy is a strategy that the agent employs to determine the next action based on the current state. It's often denoted as `π(a|s)`, which is the probability of taking action `a` while in state `s`.

### Objective

In MDP, the objective is to find an optimal policy `π*` that maximizes the expected cumulative reward over a trajectory (sequence of states visited by the agent), typically with some form of discounting applied to future rewards.

### Markov Property

An essential property of MDPs is the Markov property, which says that the transition probabilities are only dependent on the current state and action, and are independent of the history of prior states.

### Solving MDPs

MDPs can be solved using dynamic programming methods such as Value Iteration or Policy Iteration. Alternatively, methods such as Q-learning can learn the solution to an MDP through trial and error.

Please note that MDPs assume full observability of the environment's state, which is not always the case in practice. For scenarios where the state isn't fully observable, variations like Partially Observable Markov Decision Processes (POMDPs) may be used.

## Rewards in Reinforcement Learning

In Reinforcement Learning (RL), the goal of an agent is to learn a policy that maximizes the cumulative reward it receives from the environment over time. The reward plays a central role in this learning process.

### Definition

A **reward**, `R`, is a scalar feedback signal that indicates how well the agent is doing at each step. The agent's objective is to learn to select actions in a way that maximizes the total reward over an episode, or even over its entire lifespan.

### Immediate vs. Cumulative Reward

- **Immediate Reward**: It is the reward the agent receives immediately after taking an action at a particular state. It is denoted as `R(s, a, s')`, where `s` is the current state, `a` is the action taken, and `s'` is the next state.

- **Cumulative Reward**: Also known as the return, it is the total reward the agent expects to gather over the future, starting from a particular state. It is usually a discounted sum of the expected future rewards, where rewards further in the future are exponentially less important. The discount factor `γ` (gamma) ranges from 0 to 1.

The cumulative reward `G_t` at time `t` is given by:
$G_t=R_{t+1}+\lambda R_{t+2}+\ldots=\sum_{k=0}^{\infty} \lambda^k R_{t+k+1}$

 However,calculating G_t directly in a complex environment can be quite challenging as it requires knowledge of all future rewards

That's why we use the value function in practice:

### Policy Iteration
In Policy Iteration, **Value function (V)** and **Policy (π)** represent two different but closely related concepts:

1. **V (Value function)**: The value function is a function that estimates the expected total return (i.e., long-term reward) that can be obtained in a certain state by following a certain policy. V is a function of states. For a given policy π and state s, V(s) represents the total return expected by executing policy π in state s.

2. **π (Policy)**: The policy is a mapping function from states to actions, determining the actions an agent should take in a given state. Policies can be deterministic, where a specific action is selected for each state; or stochastic, where a probability of selection is assigned to all possible actions for each state.

In the policy iteration process, we iteratively improve the value function and policy through two steps: policy evaluation and policy improvement. First, we calculate the value function V based on the current policy π; then, we improve the policy π to choose the action that maximizes the value function V in each state. This process continues until the policy converges to the optimal policy, and the corresponding value function is the optimal value function.

### Example
Let's use a simple environment, Gridworld, to explain Policy Iteration. Suppose we have a 4x4 gridworld where an agent can move up, down, left, and right. Our goal is to move from the top-left corner (0,0) to the bottom-right corner (3,3).

First, we initialize a random policy π, which is at each state, the agent has an equal chance to choose any of the four directions. We also initialize a value function V, where the value of all states is 0.

We first perform Policy Evaluation, which is to evaluate the value of each state according to the current policy. For each state, we calculate the expected return from all states that the agent may reach according to the current policy. For example, for the state (0,0), the agent has four actions: up, down, left, and right. But going up and to the left will keep the agent in the same place because it is a boundary, and going down and to the right will move the agent to (1,0) and (0,1) respectively. Assuming the reward for each move is -1, we can update V(0,0) as follows:

V(0,0) = 0.25 * (-1 + γ*V(0,0)) + 0.25 * (-1 + γ*V(1,0)) + 0.25 * (-1 + γ*V(0,0)) + 0.25 * (-1 + γ*V(0,1))
Here γ is the discount factor, we can assume it to be 1.

Iteration 1:

Policy Evaluation: 我们使用当前的策略（每个动作的概率相等）来更新状态值。例如，我们计算状态(0,0)的值为V(0,0) = 0.25 * (-1 + V(0,0)) + 0.25 * (-1 + V(1,0)) + 0.25 * (-1 + V(0,0)) + 0.25 * (-1 + V(0,1)) = -1。我们用这样的方式更新所有的状态值。这时，所有的状态值为-1，除了目标状态(3,3)，其值为0。
Policy Improvement: 我们检查是否可以改进策略。对于每个状态，我们看看采取其他动作能否得到更好的期望回报。在这个例子中，由于所有的状态值都是-1，策略保持不变。
Iteration 2:

Policy Evaluation: 我们再次使用当前策略来更新状态值。例如，状态(0,0)的新值为V(0,0) = 0.25 * (-1 + V(0,0)) + 0.25 * (-1 + V(1,0)) + 0.25 * (-1 + V(0,0)) + 0.25 * (-1 + V(0,1)) = -1.75。
Policy Improvement: 这次，由于状态(2,2)，(2,3)，(3,2)的状态值为-1（它们是靠近目标状态的状态），而其他的状态值都小于-1，所以我们会改进策略，使得从这些状态开始，智能体会直接向目标状态移动。
Iteration 3:

Policy Evaluation: 状态值会继续下降，因为除了目标状态附近的几个状态，其他状态的价值还没有得到改善。
Policy Improvement: 更多的状态会更新他们的策略，比如状态(1,2)，(2,1)，因为通过新的策略，他们可以在下一步就到达已经知道有更高价值的状态。
这个过程会不断进行，直到策略不再改变。最后，我们会发现最优策略就是在每个状态，都选择可以直接朝向目标状态移动的动作。

Then we perform Policy Improvement. For each state, we look at whether we can improve the value of the state by changing the action. For example, for the state (0,0), we can calculate the expected return for each action and then choose the action with the maximum expected return as the new policy. In this case, moving to (1,0) or (0,1) might give a higher return, because we might have calculated during the Policy Evaluation stage that the value of these states might be higher than that of (0,0). So, we can change the policy π(0,0) to always move down or to the right.

We do such improvement for all states, and then perform Policy Evaluation again. We repeat Policy Evaluation and Policy Improvement until the policy no longer changes. At this point, we have obtained the optimal policy.

在标准的Policy Iteration方法中，我们会在Policy Evaluation阶段通过迭代更新价值函数，直到达到一个稳定状态，这个稳定状态就是根据当前策略计算出的价值函数不再发生大的改变。然后，我们会在Policy Improvement阶段尝试改变策略来提高价值函数。

但是，实际上并不需要等价值函数完全收敛才进行Policy Improvement。在许多实际情况下，为了提高计算效率，我们只进行有限次数的Policy Evaluation迭代（例如一次或几次），然后就进行Policy Improvement，这种方法被称为“Modified Policy Iteration”。

无论是使用标准的Policy Iteration还是Modified Policy Iteration，Policy Evaluation和Policy Improvement两个阶段都会交替进行，直到策略不再发生改变，这时我们得到的就是最优策略

在策略评估的过程中,我们会首先根据所有状态的当前值计算每个状态的新值，但是我们并不会立即更新这些值。取而代之的是，我们会等到每个状态的新值都被计算出来之后，才会同步更新所有的状态值。

这也是为什么我们需要多次迭代，因为每个状态的价值更新都依赖于其他状态的价值，所以需要不断迭代更新，直到所有状态的价值收敛，也就是不再发生显著变化。

在Policy Evaluation步骤，我们会根据当前策略计算出状态值函数（state-value function）。这个过程可能需要多次迭代直到状态值函数收敛，即状态值不再有明显的变化。
在Policy Improvement步骤，我们根据更新后的状态值函数检查是否有更好的策略（比当前策略能带来更高的期望回报的策略）。如果存在更好的策略，我们就更新当前策略。
这个过程会一直重复进行，直到策略不再改变，此时我们可以认为找到了最优策略。这个过程可能会需要多个Iteration，即多次的Policy Evaluation和Policy Improvement。

在Policy Iteration中，我们会先在Policy Evaluation步骤使用当前策略计算和更新状态值函数。这个过程可能需要多次迭代直到状态值函数收敛，即状态值不再有明显的变化。然后在Policy Improvement步骤，我们根据更新后的状态值函数检查是否有更好的策略（比当前策略能带来更高的期望回报的策略）。如果存在更好的策略，我们就更新当前策略。这两个步骤会交替进行，直到策略不再变化，即我们找到了最优策略。


### Value Iteration
在进行 Value Iteration 的过程中，我们不是计算每个动作的期望回报，然后选取最大值。而是对于每个状态，我们都会考虑所有可能的动作，然后计算在采取该动作后可能到达的每个新状态的期望回报，然后选取这些期望回报中的最大值。

更具体的说，对于状态 (0,0)，我们有四个可能的动作：向左、向上、向右、向下。每个动作会带来 -1 的即时奖励，并可能将智能体带到一个新的状态。然后我们考虑在这个新状态下的状态值，乘以一个折扣因子 γ（通常小于1），再加上即时奖励，就得到了采取该动作后可能到达的新状态的期望回报。

然后，我们会选取这四个期望回报中的最大值作为新的状态值。这就是在 Value Iteration 中我们更新状态值的方式。

### 差別：
在Policy Iteration中，我们首先选择一个策略，然后通过Policy Evaluation步骤来评估这个策略的价值，即计算每个状态的状态值。然后，在Policy Improvement步骤中，我们会寻找新的策略，该策略会对每个状态选择可以使得期望回报最大的动作。我们会反复进行这两个步骤，直到策略不再改变。

在Value Iteration中，我们并没有明确地进行策略改进的步骤。相反，我们通过在每个状态中寻找能够最大化期望回报的动作来直接更新状态值。这实际上是隐含地改进了策略，因为通过这种方式，我们在每个状态都尽可能选择能够带来最大回报的动作。在足够多的迭代之后，状态值会收敛到最优状态值，并且我们可以通过选择在每个状态中能够最大化期望回报的动作来得到最优策略。

Iteration 1:

我们更新每个状态的值。例如，我们计算状态(0,0)的值为V(0,0) = max[0.25 * (-1 + V(0,0)), 0.25 * (-1 + V(1,0)), 0.25 * (-1 + V(0,0)), 0.25 * (-1 + V(0,1))] = max[-1, -1, -1, -1] = -1。我们用这样的方式更新所有的状态值。这时，所有的状态值为-1，除了目标状态(3,3)，其值为0。

Iteration 2:

我们再次更新每个状态的值。例如，状态(0,0)的新值为V(0,0) = max[0.25 * (-1 + V(0,0)), 0.25 * (-1 + V(1,0)), 0.25 * (-1 + V(0,0)), 0.25 * (-1 + V(0,1))] = V(0,0) = max[-2, -2, -2, -2] = -2

我们会看到，由于目标状态(3,3)附近的状态（比如状态(2,2)，(2,3)，(3,2)）可以直接到达目标状态，所以它们的状态值不再下降，并保持在-1。

Iteration 3:

此时，更多的状态（比如状态(1,2)，(2,1)）会因为可以在下一步到达状态值为-1的状态，所以它们的状态值也不再下降，并保持在-2。

# Q-Learning
Q Learning的思想完全根据value iteration得到。但要明确一点是value iteration每次都对所有的Q值更新一遍，也就是所有的状态和动作。但事实上在实际情况下我们没办法遍历所有的状态，还有所有的动作，我们只能得到有限的系列样本。因此，只能使用有限的样本进行操作。那么，怎么处理？Q Learning提出了一种更新Q值的办法：
Q(s, a) ← Q(s, a) + α[r + γ max_a' Q(s', a') - Q(s, a)]

其中：

α 是学习率，决定了新信息对当前Q值的影响程度。
γ 是折扣因子，决定了未来回报对当前Q值的影响程度。
max_a' Q(s', a') 是在下一状态s'下所有可能动作的最大Q值。

γ max_a' Q(s', a') - Q(s, a) 反映的是实际获取的预期回报与当前预期回报的差值，也就是预期回报的“误差”或者说是“提升空间”。我们希望通过调整 Q(s, a) 来减小这个差值，即让智能体的预期回报更接近实际的最大预期回报。

学习率 α 决定了我们调整 Q(s, a) 的步长。如果 α 大，则学习步长大，反之则小。α 也可以视情况设置为动态值，例如随着学习的进行逐渐减小，这样可以在初期快速学习，在后期进行更精细的调整。

从学习的角度来看，我们希望 γ max_a' Q(s', a') 大于 Q(s, a)。这是因为 γ max_a' Q(s', a') 代表了在新的状态 s' 下选择最优动作 a' 所能获取的最大期望回报，如果这个值大于 Q(s, a)，即当前状态 s 下执行动作 a 的预期回报，那么就说明有提升空间，我们需要调整策略以提高预期回报。

但是，也有可能 γ max_a' Q(s', a') 小于或等于 Q(s, a)，这时候就意味着在新的状态 s' 下选择最优动作 a' 的期望回报并没有超过当前的预期回报，这可能是因为智能体已经学习到了一个较优的策略，或者是因为环境发生了变化，以前的策略已经不再适用。

让我们用一个简单的例子来解释Q-Learning的工作过程。在这个例子中，我们有一个智能体，其可选择两个动作A和B，有两个状态S1和S2。假设我们的奖励函数和状态转换函数如下：

当处于S1时，选择动作A，会收到奖励0并留在S1；选择动作B，会收到奖励1并转移到S2。
当处于S2时，选择任何动作，都会收到奖励-1并转移到S1。
假设初始的Q值都为0，学习率α为0.5，折扣因子γ为0.9。然后，我们开始进行Q-Learning。

步骤1：智能体在S1，选择动作B，收到奖励1，转移到S2。更新Q(S1, B)：
Q(S1, B) ← Q(S1, B) + α [r + γ max_a' Q(s', a') - Q(S1, B)]
= 0 + 0.5 [1 + 0.9 * max{Q(S2, A), Q(S2, B)} - 0]
= 0 + 0.5 [1 + 0.9 * max{0, 0}]
= 0.5

步骤2：智能体在S2，随机选择一个动作，比如A，收到奖励-1，转移到S1。更新Q(S2, A)：
Q(S2, A) ← Q(S2, A) + α [r + γ max_a' Q(s', a') - Q(S2, A)]
= 0 + 0.5 [-1 + 0.9 * max{Q(S1, A), Q(S1, B)} - 0]
= 0 + 0.5 [-1 + 0.9 * max{0, 0.5}]
= -0.25

然后，这个过程会不断进行，更新Q值，直到Q值收敛。这个过程可能需要很多次迭代。

通常用ε-greedy策略是一种简单的方法，旨在解决探索（exploration）与利用（exploitation）的权衡问题。在Q-Learning中，我们需要一个策略来决定在给定状态下选择哪个动作。ε-greedy策略就是这样的策略，它以ε的概率随机选择一个动作，以1-ε的概率选择当前Q值最高的动作。

随着训练的进行，我们对环境的理解会逐渐加深，也就更倾向于根据已有的知识做出最优的选择（即利用）。这时，我们可以逐渐减小ε的值，以便更多地选择Q值最高的动作。但同时，为了保证有一定的探索性，ε通常不会减小到0

我们可以看到需要使用某一个policy来生成动作，也就是说这个policy不是优化的那个policy，所以Q-Learning算法叫做Off-policy的算法。另一方面，因为Q-Learning完全不考虑model模型也就是环境的具体情况，只考虑看到的环境及reward，因此是model-free的方法。

傳統的Qlearning是用 Q table 儲存Q值，但在實際情況維度會因為過大而很難儲存，所以倒不如把它變成一個function

## 价值函数近似Value Function Approximation
把Q(s,a) -> f(s,a)後無論s的維度有多大，都可以通過f轉換成Q值。
如果我们就用w来统一表示函数f的参数，那么就有
Q(s,a) -> f(s,a,w)（逼近，並不等於）

## 4 高维状态输入，低维动作输出的表示问题
其实就是Q(s) 約等於 f(s,w)，只把状态s作为输入，但是输出的时候输出每一个动作的Q值，也就是输出一个向量[Q(s,a1),Q(s,a2)...Q(s,an)]
，记住这里输出是一个值，只不过是包含了所有动作的Q值的向量而已。这样我们就只要输入状态s，而且还同时可以得到所有的动作Q值，也将更方便的进行Q-Learning中动作的选择与Q值更新.


