# Mathematical background

This document provides a complete formal description of every component of
the baseline GFM simulation. The goal is to show that the stochastic
evolutionary process is fully specified — every quantity that appears in the
code has an exact mathematical counterpart here.

---

## 1. State spaces and notation

### Phenotype space

All phenotypes live in $\mathbb{R}^n$, where $n \geq 1$ is the number of
independent phenotypic traits. The baseline uses $n = 4$.

### Individual

An individual at generation $t$ is characterised entirely by its
**phenotype vector**

$$\mathbf{p} \in \mathbb{R}^n.$$

No genotype, ancestry, or age information is stored. (`individual.py`)

### Population

A population at generation $t$ is an **ordered multiset** of $N_t$ individuals:

$$\mathcal{P}_t = \{\mathbf{p}_1^{(t)},\, \mathbf{p}_2^{(t)},\, \ldots,\, \mathbf{p}_{N_t}^{(t)}\} \subset \mathbb{R}^n.$$

The target population size $N$ is a fixed parameter; $N_t$ may fall below $N$
only if the population goes extinct (see §6). (`population.py`)

### Environment

The environment is characterised by a single time-varying quantity: the
**optimal phenotype** $\boldsymbol{\alpha}(t) \in \mathbb{R}^n$. (`environment.py`)

---

## 2. Initialisation

### Population initialisation

Each initial phenotype is drawn independently from an isotropic Gaussian
centred on the initial optimum $\boldsymbol{\alpha}(0) = \boldsymbol{\alpha}_0$:

$$\mathbf{p}_i^{(0)} \overset{\mathrm{i.i.d.}}{\sim} \mathcal{N}\!\left(\boldsymbol{\alpha}_0,\; \sigma_{\mathrm{init}}^2 \mathbf{I}_n\right), \qquad i = 1, \ldots, N,$$

where $\sigma_{\mathrm{init}}$ (`init_scale`) controls the initial spread.

**Calibration rule.** Because $\|\mathbf{p} - \boldsymbol{\alpha}_0\|^2 \sim \sigma_{\mathrm{init}}^2 \chi^2(n)$, the
expected fitness in generation 0 is

$$\mathbb{E}[\varphi_0] = \left(1 + \frac{\sigma_{\mathrm{init}}^2}{\sigma^2}\right)^{-n/2}.$$

The rule of thumb $\sigma_{\mathrm{init}} = \sigma / \sqrt{n}$ gives
$\mathbb{E}[\varphi_0] = (1 + 1/n)^{-n/2} \approx e^{-1/2} \approx 0.61$
for large $n$, ensuring the population starts with reasonable average fitness.

### Environment initialisation

The initial optimal phenotype is set to

$$\boldsymbol{\alpha}(0) = \boldsymbol{\alpha}_0 = \mathbf{0}_n$$

(the origin of phenotype space). (`config.py`)

---

## 3. Fitness function

The fitness of individual $\mathbf{p}$ in environment $\boldsymbol{\alpha}$ is

$$\varphi(\mathbf{p}, \boldsymbol{\alpha}) = \exp\!\left(-\frac{\|\mathbf{p} - \boldsymbol{\alpha}\|^2}{2\sigma^2}\right) \in (0, 1].$$

This is the standard Gaussian fitness kernel of Fisher's Geometric Model
(Orr 1998, Martin & Lenormand 2006). The parameter $\sigma > 0$ (`sigma`)
is the **selection tolerance**: it sets the characteristic scale of
phenotypic distances that are tolerated.

- $\varphi = 1$ if and only if $\mathbf{p} = \boldsymbol{\alpha}$ (perfect adaptation).
- $\varphi \to 0$ as $\|\mathbf{p} - \boldsymbol{\alpha}\| \to \infty$.
- The fitness iso-surface at level $f_0$ is the sphere of radius
  $r_0 = \sigma\sqrt{-2\ln f_0}$.

For example, the hard threshold $\tau$ (`threshold`) = 0.01 corresponds to a
maximum tolerated distance of

$$r_\tau = \sigma\sqrt{-2\ln\tau} = 0.2\sqrt{-2\ln 0.01} \approx 0.61.$$

(`selection.py`, `fitness_function`)

---

## 4. The evolutionary loop

Let $\mathcal{P}_t$ denote the population at the **start** of generation $t$
(after environment has been updated to $\boldsymbol{\alpha}(t)$). One generation
applies four operators in the following order:

$$\mathcal{P}_t \xrightarrow{\;\text{mutation}\;} \tilde{\mathcal{P}}_t \xrightarrow{\;\text{selection}\;} \mathcal{S}_t \xrightarrow{\;\text{reproduction}\;} \mathcal{P}_{t+1}^* \xrightarrow{\;\text{env. update}\;} \text{set } \boldsymbol{\alpha}(t+1).$$

The population that enters generation $t+1$ is $\mathcal{P}_{t+1} = \mathcal{P}_{t+1}^*$.

---

## 5. Step 1 — Mutation (`IsotropicMutation`)

Mutation is applied **before** selection, so new variation is exposed to
selection within the same generation.

### Two-level Bernoulli process

For each individual $\mathbf{p} \in \mathcal{P}_t$:

1. **Individual-level gate.** With probability $\mu$ (`mu`) the individual
   is a *mutant*; with probability $1 - \mu$ it passes unchanged.

2. **Trait-level perturbation.** For each mutant, every trait $i \in \{1,\ldots,n\}$
   mutates independently with probability $\mu_c$ (`mu_c`). The mutation
   increment is drawn from a zero-mean isotropic Gaussian:

$$\Delta p_i \sim \mathcal{N}(0,\, \xi^2), \qquad \text{independently for each } i.$$

Combining the two levels, the post-mutation phenotype $\tilde{\mathbf{p}}$ of an
individual with pre-mutation phenotype $\mathbf{p}$ is

$$\tilde{p}_i = \begin{cases}
  p_i + \Delta p_i, & \text{with probability } \mu \cdot \mu_c, \\
  p_i,              & \text{with probability } 1 - \mu \cdot \mu_c.
\end{cases}$$

This is a **compound Bernoulli–Gaussian** model. The marginal distribution of
$\tilde{p}_i$ is a two-component mixture:

$$\tilde{p}_i \sim (1 - \mu\mu_c)\,\delta_{p_i} + \mu\mu_c\,\mathcal{N}(p_i,\, \xi^2).$$

### Mutation radius distribution

For a fully mutating individual ($\mu = \mu_c = 1$), the squared displacement
$\|\Delta\mathbf{p}\|^2 = \sum_{i=1}^n \Delta p_i^2 \sim \xi^2 \chi^2(n)$, so

$$\mathbb{E}\!\left[\|\Delta\mathbf{p}\|^2\right] = n\xi^2, \qquad
\mathrm{SD}\!\left[\|\Delta\mathbf{p}\|\right] \approx \xi\sqrt{n} \text{ (for large } n\text{)}.$$

Fisher's classical result is that a random mutation is beneficial if and only
if its displacement moves the phenotype closer to the optimum. The probability
of this event decreases as $n$ grows, which is why $\xi$ must be scaled down
in higher-dimensional spaces.

(`mutation.py`, `IsotropicMutation._mutate_individual`)

---

## 6. Step 2 — Selection (`TwoStageSelection`)

The default selection operator is a **two-stage** procedure.

### Stage 1 — Hard threshold (viability selection)

Every individual $\tilde{\mathbf{p}} \in \tilde{\mathcal{P}}_t$ survives if and only if
its fitness meets a minimum viability threshold $\tau$ (`threshold`):

$$\mathcal{S}_t^{(1)} = \left\{\tilde{\mathbf{p}} \in \tilde{\mathcal{P}}_t \;\Big|\; \varphi(\tilde{\mathbf{p}}, \boldsymbol{\alpha}(t)) \geq \tau \right\}.$$

This is equivalent to requiring

$$\|\tilde{\mathbf{p}} - \boldsymbol{\alpha}(t)\| \leq \sigma\sqrt{-2\ln\tau}.$$

If $\mathcal{S}_t^{(1)} = \emptyset$ the population is declared **extinct** at
generation $t$ and the simulation halts.

### Stage 2 — Fitness-proportionate sampling (fecundity selection)

From the survivors $\mathcal{S}_t^{(1)}$, exactly $N$ **parent copies** are
drawn with replacement. The probability that individual $j$ is chosen as a
parent in one draw is

$$\pi_j = \frac{\varphi(\tilde{\mathbf{p}}_j, \boldsymbol{\alpha}(t))}{\displaystyle\sum_{k \in \mathcal{S}_t^{(1)}} \varphi(\tilde{\mathbf{p}}_k, \boldsymbol{\alpha}(t))}.$$

The selected parent multiset is

$$\mathcal{S}_t = \left(J_1, J_2, \ldots, J_N\right), \quad
J_i \overset{\mathrm{i.i.d.}}{\sim} \mathrm{Categorical}(\boldsymbol{\pi}),$$

where $|\mathcal{S}_t| = N$ always (barring extinction). This is a
**Wright–Fisher sampling step** on the survivor pool.

**Combined effect.** Stage 1 removes universally unfit genotypes (hard selection);
Stage 2 continuously favours above-threshold individuals in proportion to how
much better they are than their peers (soft selection). The two stages are
complementary: Stage 1 prevents the fitness of the worst individual from
dragging down the sampling probabilities, and Stage 2 prevents genetic drift
from dominating when all survivors have similar fitness.

(`selection.py`, `TwoStageSelection.select`)

---

## 7. Step 3 — Reproduction (`AsexualReproduction`)

Given the parent multiset $\mathcal{S}_t$ of size $N$, each parent produces
exactly **one clonal offspring**:

$$\mathbf{p}_i^{(t+1)} = \tilde{\mathbf{p}}_{J_i}, \qquad i = 1, \ldots, N.$$

The new population is $\mathcal{P}_{t+1}^* = \{\mathbf{p}_1^{(t+1)}, \ldots, \mathbf{p}_N^{(t+1)}\}$,
which has exactly $N$ individuals (no further mutation is applied here;
mutation occurs at the start of the **next** generation).

### Offspring count distribution

Let $k_j$ be the number of offspring of survivor $j \in \mathcal{S}_t^{(1)}$
(i.e. the number of times index $j$ appears in the draw). The vector
$(k_1, \ldots, k_{|\mathcal{S}_t^{(1)}|})$ follows a **multinomial distribution**:

$$(k_1, \ldots, k_m) \sim \mathrm{Multinomial}\!\left(N,\, \boldsymbol{\pi}\right),$$

where $m = |\mathcal{S}_t^{(1)}|$ and $\boldsymbol{\pi}$ are the Stage-2
probabilities from §6.

The **effective number of parents** — the number of survivors with at least
one offspring — is

$$N_{\mathrm{par}} = \#\{j : k_j \geq 1\},$$

a random variable with distribution concentrated around
$m \left(1 - (1 - 1/m)^N\right) \approx m(1 - e^{-N/m})$.

The effective population size in the Wright–Fisher sense is

$$N_e = \frac{N - 1}{\mathbb{E}\!\left[\sum_j k_j(k_j-1)\right] / (N(N-1))} \leq N,$$

with equality when all $\pi_j$ are equal (neutral drift).

(`reproduction.py`, `AsexualReproduction.reproduce`)

---

## 8. Step 4 — Environment update (`LinearShiftEnvironment`)

After reproduction, the optimal phenotype is updated according to a
**linear drift with additive Gaussian noise**:

$$\boldsymbol{\alpha}(t+1) = \boldsymbol{\alpha}(t) + \boldsymbol{\varepsilon}_t,
\qquad
\boldsymbol{\varepsilon}_t \sim \mathcal{N}(\mathbf{c},\, \delta^2 \mathbf{I}_n),$$

where
- $\mathbf{c} \in \mathbb{R}^n$ (`c`) is the **mean drift vector** — the deterministic
  component of environmental change per generation,
- $\delta \geq 0$ (`delta`) is the **stochastic volatility** — the standard
  deviation of isotropic fluctuations around $\mathbf{c}$.

In the baseline, $\mathbf{c} = (0.01, 0.01, 0.01, 0.01)^\top$ and $\delta = 0.01$.

Unrolling the recursion gives the closed-form trajectory

$$\boldsymbol{\alpha}(t) = \boldsymbol{\alpha}_0 + t\mathbf{c} + \sum_{s=0}^{t-1}(\boldsymbol{\varepsilon}_s - \mathbf{c}),$$

so $\boldsymbol{\alpha}(t) \sim \mathcal{N}({\boldsymbol{\alpha}_0 + t\mathbf{c}},\; t\delta^2 \mathbf{I}_n)$.
The optimum drifts linearly at speed $\|\mathbf{c}\|$ with Brownian diffusion
of variance $\delta^2 t$ in each dimension.

**When $\delta = 0$** (deterministic drift): $\boldsymbol{\alpha}(t) = \boldsymbol{\alpha}_0 + t\mathbf{c}$.

(`environment.py`, `LinearShiftEnvironment.update`)

---

## 9. The critical drift threshold

A central prediction of FGM under linear environmental change is the existence
of a **critical drift speed** $\|\mathbf{c}^*\|$ above which no finite population
can adapt indefinitely — the lag between mean phenotype and optimum grows
without bound, and extinction is certain.

A population that perfectly tracks the optimum must acquire a mean phenotypic
shift of $\|\mathbf{c}\|$ per generation. The maximum achievable shift per
generation is limited by the available mutational variance and the strength of
selection. Qualitatively, the lag $d(t) = \|\bar{\mathbf{p}}(t) - \boldsymbol{\alpha}(t)\|$
satisfies an approximate recursion

$$d(t+1) \approx d(t) + \|\mathbf{c}\| - \frac{d(t)}{C(\sigma, n, N)},$$

where $C$ collects the effects of selection strength and population size.
The equilibrium lag $d^* = C \|\mathbf{c}\|$ is stable; if $\|\mathbf{c}\|$ exceeds
a threshold that depends on $\sigma$, $n$, and $N$, no stable equilibrium exists
and the population goes extinct. Empirically (from the drift sweep in
`experiments/drift_c*.json`) the critical threshold is approximately
$\|\mathbf{c}^*\| \approx 0.010$–$0.011$ for the baseline parameters.

---

## 10. Collected statistics (`SimulationStats`)

After step 3 (reproduction) and before step 4 (environment update), the
following quantities are recorded for each generation $t$:

| Symbol | Formula | Name in code |
|---|---|---|
| $\bar{\mathbf{p}}(t)$ | $\frac{1}{N}\sum_{i=1}^N \mathbf{p}_i^{(t+1)}$ | `mean_phenotype` |
| $\bar{\varphi}(t)$ | $\frac{1}{N}\sum_{i=1}^N \varphi(\mathbf{p}_i^{(t+1)}, \boldsymbol{\alpha}(t))$ | `mean_fitness` |
| $d(t)$ | $\|\bar{\mathbf{p}}(t) - \boldsymbol{\alpha}(t)\|$ | `distance_from_optimum` |
| $V(t)$ | $\frac{1}{n}\mathrm{tr}\,\widehat{\mathrm{Cov}}(\mathcal{P}_{t+1}^*)$ | `phenotype_variance` |
| $N_t$ | $\left|\mathcal{P}_{t+1}^*\right| = N$ | `population_size` |
| $N_{\mathrm{par}}(t)$ | $\#\{j : k_j \geq 1\}$ | `n_parents` |
| $\tilde{k}(t)$ | $\mathrm{median}\{k_j : k_j \geq 1\}$ | `median_offspring` |
| $k_{\max}(t)$ | $\max_j k_j$ | `max_offspring` |

where

$$V(t) = \frac{1}{n}\sum_{d=1}^n \widehat{\mathrm{Var}}\!\left[\{p_d^{(t+1)}\}_{i=1}^N\right]$$

is the mean per-dimension variance, a scalar proxy for **phenotypic diversity**
(genetic diversity in the absence of an explicit genotype layer).

(`stats.py`, `SimulationStats.record`)

---

## 11. Complete state-space summary

The full state of the simulation at the start of generation $t$ is the pair

$$\Omega_t = (\mathcal{P}_t,\, \boldsymbol{\alpha}(t)) \in (\mathbb{R}^n)^N \times \mathbb{R}^n.$$

Given $\Omega_t$, the state $\Omega_{t+1}$ is determined by the four-operator
composition:

$$\Omega_{t+1} = \Phi_4\!\left(\Phi_3\!\left(\Phi_2\!\left(\Phi_1(\mathcal{P}_t,\, \boldsymbol{\alpha}(t))\right)\right)\right),$$

where:

| Operator | Map | Randomness |
|---|---|---|
| $\Phi_1$ — Mutation | $(\mathcal{P}_t, \boldsymbol{\alpha}) \mapsto (\tilde{\mathcal{P}}_t, \boldsymbol{\alpha})$ | Compound Bernoulli–Gaussian per trait |
| $\Phi_2$ — Selection stage 1 | $(\tilde{\mathcal{P}}_t, \boldsymbol{\alpha}) \mapsto (\mathcal{S}_t^{(1)}, \boldsymbol{\alpha})$ | Deterministic threshold |
| $\Phi_3$ — Selection stage 2 + Reproduction | $(\mathcal{S}_t^{(1)}, \boldsymbol{\alpha}) \mapsto (\mathcal{P}_{t+1}, \boldsymbol{\alpha})$ | Multinomial$(N, \boldsymbol{\pi})$ |
| $\Phi_4$ — Environment update | $(\mathcal{P}_{t+1}, \boldsymbol{\alpha}(t)) \mapsto (\mathcal{P}_{t+1}, \boldsymbol{\alpha}(t+1))$ | $\mathcal{N}(\mathbf{c}, \delta^2\mathbf{I}_n)$ |

The process $(\Omega_t)_{t \geq 0}$ is a **finite-state-space Markov chain**
(when phenotypes are discretised) or a **Markov process on
$(\mathbb{R}^n)^N \times \mathbb{R}^n$** in the continuous limit. All
randomness is conditionally independent given $\Omega_t$.

---

## 12. Key parameter reference

| Symbol | Parameter | Default | Role |
|---|---|---|---|
| $n$ | `n` | 4 | Phenotype space dimensionality |
| $N$ | `N` | 100 | Target population size |
| $\sigma$ | `sigma` | 0.20 | Selection tolerance (fitness kernel width) |
| $\tau$ | `threshold` | 0.01 | Hard viability threshold |
| $\mu$ | `mu` | 0.10 | Per-individual mutation probability |
| $\mu_c$ | `mu_c` | 0.50 | Per-trait mutation probability (given individual mutates) |
| $\xi$ | `xi` | 0.05 | Per-trait mutation step size (std dev) |
| $\mathbf{c}$ | `c` | $(0.01)^n$ | Mean drift vector |
| $\delta$ | `delta` | 0.01 | Environmental volatility (std dev of $\boldsymbol{\varepsilon}_t$) |
| $\sigma_{\mathrm{init}}$ | `init_scale` | 0.10 | Spread of initial phenotypes |
| $\boldsymbol{\alpha}_0$ | `alpha0` | $\mathbf{0}_n$ | Initial optimal phenotype |
| $T$ | `max_generations` | 200 | Number of generations |

---

## 13. References

- Fisher, R.A. (1930). *The Genetical Theory of Natural Selection*. Oxford University Press.
- Orr, H.A. (1998). The population genetics of adaptation: the distribution of factors fixed during adaptive evolution. *Evolution*, 52(4), 935–949.
- Martin, G. & Lenormand, T. (2006). A general multivariate extension of Fisher's geometrical model and the distribution of mutation fitness effects across species. *Evolution*, 60(5), 893–907.
- Bürger, R. & Lynch, M. (1995). Evolution and extinction in a changing environment: a quantitative-genetic analysis. *Evolution*, 49(1), 151–163.
- Kopp, M. & Hermisson, J. (2009). The genetic basis of phenotypic adaptation I: fixation of beneficial mutations in the moving optimum model. *Genetics*, 182(1), 233–249.
