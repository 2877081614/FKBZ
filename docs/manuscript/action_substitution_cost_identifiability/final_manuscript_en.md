# Paired Counterfactual Auditing of Resource-Cost Measurement in Dynamically Masked Sequential Allocation

Manuscript status: W1-10 L2/M2 scientific final (journal format not selected)  
Positioning: L2 measurement, diagnosis, and resource-credit decomposition module  
Empirical scope: AirDefense v1 with frozen factorized joint PPO

## Abstract

<!-- A01 -->
Dynamic resource allocation requires team outcomes to be translated into local resource credit
for a current action. In an autoregressive joint action with dynamic masking, however, an
engagement changes both the legal same-step suffix and subsequent policy responses, so an
episode-level cost difference can mix direct expenditure with action substitution. We audit this
measurement in AirDefense v1 using no-engage/engage paired counterfactual trajectories under
frozen factorized joint PPO. Common random numbers, exact marginalization over legal targets,
and a stepwise cost ledger decompose total substitution cost into same-step other-unit, future
probe, and future other-unit components. An initial future-only identity left residuals in 287
of 7,776 target-conditioned ledger records, with a maximum residual of 2.0. Adding the
same-step component reduced the maximum error of the complete identity to
\(8.88\times10^{-16}\). Independent confirmation retained all nine newly trained source
models and 108 contexts with zero observation-hash overlap with the earlier formal audit. The
95% lower bound on block-level substituted shots was positive for policy seeds 17, 18, and 19
in `time_pressure/resource`. Cost-sign masking nevertheless occurred in only 2/9 missile
contexts and 5/9 laser contexts, so the preregistered cross-resource gate failed. More
rollouts can reduce paired-sampling variance, but they cannot remove structural path mixing
from cumulative cost. The ledger is therefore a measurement and diagnostic module for
dynamically masked sequential resource allocation, not a validated online PPO improvement or
evidence of cross-environment credit generalization.

**Keywords:** dynamic action masking; sequential resource allocation; counterfactual credit;
resource-cost measurement; action substitution; common random numbers

## 1. Introduction

<!-- I01 -->
Dynamic resource allocation requires a policy to balance mission outcomes against finite
resource expenditure. In air-defense allocation, an engagement both acts on the current target
and consumes physically interpretable ammunition or energy, thereby changing the actions
available later in the episode. Auditing a local engage/no-op decision consequently requires
more than an episode-level performance score. It also requires a defensible account of the
resource credit attributable to the current action.

<!-- I02 -->
Multi-agent reinforcement learning commonly estimates local action contributions from team
returns, episode returns, or counterfactual return differences. Difference rewards, COMA,
Shapley credit, and difference-return methods have established that a global outcome cannot be
unconditionally assigned to one agent or current action [E01-E05]. Temporal credit and causal
effect decompositions further show that current-action outcomes propagate jointly with
exogenous randomness, subsequent actions, and responses from other agents [E06-E09, E19,
E20]. The broad difficulty of mapping team outcomes to local credit is therefore not new. The
unresolved measurement question here is narrower: what does a cumulative cost difference
measure under a particular sequential action structure and physical resource-cost definition?

<!-- I03 -->
This question becomes concrete in autoregressive joint actions with dynamic masking. Units act
in a fixed order, and a target selected by an earlier unit is removed from the legal set of
later units. A current action can also alter subsequent states, resource availability, legal
sets, and policy roles. Autoregressive MARL, sequential team credit, and invalid action masking
have addressed sequential modeling, credit allocation, and action legality [E12-E14, E21].
They do not by themselves determine whether episode-level resource cost remains a stable readout
of direct current-action cost when a local intervention changes both the same-step suffix and
future actions.

<!-- I04 -->
We address this question with a paired no-engage/engage (N/E) counterfactual audit under frozen
factorized joint PPO in AirDefense v1 (Fig. 1). Both branches start from the same saved state,
share environment and policy random-number tapes, exactly marginalize over legal current
targets, and continue with the frozen stochastic policy. A stepwise ledger separates total
substitution cost into same-step other-unit, future probe, and future other-unit components, so
the episode-level cost difference can be reconstructed record by record (Fig. 2). Common random
numbers reduce paired variance; they are not treated as a source of structural identifiability.

<!-- I05 -->
The audit separates mechanism discovery from independent confirmation. Earlier policy seeds
were used only to discover action substitution. Confirmation used policy seeds 17, 18, and 19,
all nine source models without behavioral selection, and 108 contexts with zero observation-hash
overlap with the earlier formal audit. Same-step and future substitution replicated across the
new seeds, but positive substitution did not imply that cumulative cost necessarily changed sign.
Both missile and laser contexts showed positive substituted shots, while the cross-resource
cost-sign-masking gate failed (Figs. 3-5). The three pressure configurations are internal
AirDefense v1 conditions, not independent environments.

<!-- I06 -->
The study makes three bounded contributions. First, it operationalizes a known counterfactual
credit problem as a local resource-cost measurement audit for dynamically masked sequential
allocation, separating direct expenditure from action-mediated cost changes. Second, it
provides a reproducible protocol combining N/E pairing, exact target marginalization, and a
three-component stepwise ledger, and shows why the same-step suffix cannot be omitted. Third,
it confirms substitution using new policy seeds and contexts while preserving scenario- and
resource-type failure boundaries. The work is positioned as a measurement and diagnostic
module within a broader credit-assignment method. It does not claim an online method superior
to PPO, nor does it present BPCE/MCH-PPO or a GNN as a validated contribution.

## 2. Related Work

### 2.1 Multi-agent counterfactual credit

<!-- RW01 -->
Difference rewards isolate an agent's marginal contribution through a default-action utility,
COMA marginalizes the current agent's action while holding other actions fixed, and Shapley and
difference-return methods organize local credit through cooperative allocation or policy
gradients [E01-E05]. These methods establish counterfactual comparison as a central tool for
team credit and identify biases that can arise when baseline-independence conditions fail. We
do not reintroduce counterfactual credit. We instead study a setting in which "holding other
actions fixed" may not define a legal autoregressive suffix under dynamic masking, and we
explicitly record the resource-cost changes generated by branch-specific policy responses.

### 2.2 Temporal credit and downstream behavioral mediation

<!-- RW02 -->
CCA, HCA, RUDDER, COCOA, and causal reward redistribution address temporal credit arising from
exogenous randomness, hindsight information, delayed returns, contribution objects, or latent
reward structure [E06-E09, E23]. Recent causal MARL studies additionally decompose the total
effect of an action into paths mediated by other-agent behavior and state transitions [E19,
E20]. The general observation that downstream actions mediate current-action effects is already
established. Our narrower object is a physically interpretable resource cost, for which the
same-step masked suffix, future probe, and future other-unit paths are recorded separately in
an operational ledger that can be checked row by row.

### 2.3 Sequential joint actions and dynamic masking

<!-- RW03 -->
HAPPO and HATRPO study sequential multi-agent updates, MAT represents joint actions
autoregressively, invalid action masking establishes properties of masked policy gradients, and
CAPO and related work address unit-level credit in fixed-order cooperative teams [E12-E14, E21,
E22]. Sequential decision making, autoregressive policies, and action masks are therefore not
contributions of this study. The remaining measurement consequence is that an intervention
which changes a suffix legal set cannot assume fixed same-step other-unit cost. That component
must be audited separately from strictly future substitution.

### 2.4 Resource constraints and opportunity value

<!-- RW04a -->
CPO, MACPO, and scalable constrained MARL control policy-level cumulative cost and safety
constraints through trust-region, Lagrangian, or distributed optimization [E16-E18]. These
methods assess budget feasibility, not whether the cost of one local action belongs to direct
expenditure or to downstream policy responses. We treat resource cost as a local measurement
object rather than a new constrained-optimization objective. Our resource-restoration audit did
not yield a broadly applicable safety opportunity-value label, so an enlarged legal action set
does not by itself support an opportunity oracle.

### 2.5 Paired simulation and measurement validity

<!-- RW04b -->
Common random numbers reduce the variance of a difference by sharing random streams across two
simulated systems [E15]. In our audit, CRN aligns exogenous inputs in the N/E branches and exact
target marginalization removes current-target sampling error. These operations improve
precision, but they do not determine which action paths enter the cost difference. We therefore
separate variance control from measurement validity: structural interpretation comes from the
intervention identity, legal branch construction, and complete cost ledger rather than from
random-number correlation.

### 2.6 Positioning of the present study

<!-- RW05 -->
Prior work covers global-to-local credit, downstream behavioral mediation, sequential joint
actions, dynamic masking, resource constraints, and paired variance reduction. The remaining
contribution is their combination for a specific measurement object: a local N/E intervention
under dynamic legal sets, a decomposition of same-step and two future resource-cost paths, a
record-level identity, and explicit scenario and resource-type sign boundaries. The study is
thus a measurement and diagnostic component that complements broader counterfactual and causal
effect decompositions, rather than a new sequential MARL algorithm.

## 3. Problem Formulation and Evaluation Scope

### 3.1 AirDefense v1 dynamic resource allocation

<!-- PF01 -->
We study dynamic air-defense resource allocation in AirDefenseResourceAssignmentEnv v1
(AirDefense v1). Each decision step contains three heterogeneous defense units, up to five live
targets, and two protected zones. The default medium configuration has two missile units and
one laser unit. Missile cost, capacity, hit probability, and range are 2.0, 3, 0.88, and 85;
the corresponding laser values are 0.5, 10, 0.68, and 55. A 142-dimensional normalized state
encodes zones, targets, defense units, and episode information. Each unit chooses a live,
in-range target or no-op from the joint space \(\{0,\ldots,5\}^3\), where index 5 denotes
no-op. Ammunition, cooldown, energy, target survival, and range define a dynamic legal-action
mask. At most one unit can be assigned to a target within a step, and a legal engagement incurs
its resource cost immediately. The ledger uses actual cumulative resource cost only, not the
composite reward containing interception, damage, time, and terminal terms. An episode ends
when all targets are inactive or cumulative damage reaches 2.5, and is truncated at 50 steps
(Table 1; Supplementary Methods S1).

### 3.2 Frozen factorized joint policy

<!-- PF02 -->
The source policy is factorized joint PPO and is not proposed by this study. For the fixed unit
order \(0\rightarrow1\rightarrow2\), the conditional action distribution for unit \(i\) is

\[
\pi(a_i\mid s,a_{<i})=
\begin{cases}
1-\sigma(g_i), & a_i=\mathrm{no\mbox{-}op},\\
\sigma(g_i)\mathrm{Softmax}(\ell_i\mid\mathcal{L}_i(s,a_{<i}))_{a_i},
& a_i\in\mathcal{L}_i(s,a_{<i}),
\end{cases}
\]

where \(g_i\) is the engage logit, \(\ell_i\) contains target logits, and
\(\mathcal{L}_i\) is determined jointly by environment feasibility and target occupancy in the
action prefix. Joint log probability is the sum of the three conditional log probabilities,
and each complete joint action advances the environment once. "Factorized" denotes a
conditional decomposition, not unit independence. Source models were trained with the joint PPO
ratio and clipped surrogate; no parameters were updated during confirmation.

### 3.3 Local resource-credit estimand

<!-- PF03 -->
At a state visited by the source policy, we freeze the environment snapshot, probe unit \(i\),
decision step \(t\), and its legal target set, and compare two branches controlled only through
the probe's current-action identity. The \(N\) branch forces no-op; the \(E\) branch forces
engagement of one specified legal target. The episode-level cumulative cost difference is

\[
\Delta C_{\mathrm{episode}}
:=C_{\mathrm{episode}}(E)-C_{\mathrm{episode}}(N).
\]

A positive value means that current engagement increased cumulative episode cost. Substitution
quantities use the opposite saving direction, \(N-E\), so a positive value means that the
\(E\) branch displaced expenditure that would have occurred under \(N\). The estimand is a
local counterfactual cost decomposition conditional on the state, probe, legal-target
distribution, and frozen continuation policy. It is not the causal effect of policy training
or an identification of all potential action paths.

### 3.4 Scope and non-claims

<!-- PF04 -->
Formal confirmation covers `medium`, `time_pressure`, and `heterogeneity_pressure` within the
same AirDefense v1 environment family. We do not evaluate out-of-distribution transfer across
environments, claim that online PPO has been improved, or treat BPCE/MCH-PPO or a GNN as a
validated solution.

## 4. Paired Counterfactual Resource-Cost Decomposition

### 4.1 N/E paired intervention

<!-- M01 -->
Each context stores the complete environment state, normalized observation, decision step,
probe unit, legal targets, and conditional action probabilities. Replay restores the same
snapshot and fixed joint-action prefix before constructing the \(N\) and \(E\) branches. Units
after the probe recompute their dynamic masks and are sampled separately in each branch. The
intervention can therefore change the same-step suffix legal set and later states without
forcing an action that has become illegal (Fig. 1).

### 4.2 Common random numbers and exact target marginalization

<!-- M02 -->
Each repeat generates paired random-number tapes for the N/E branches. The environment tape
indexes hit random numbers by environment step and target, while the policy tape indexes
conditional-action uniforms by step and unit. Both branches use the same indexed values. When
states or masks diverge, the same uniform is mapped through each branch's conditional
distribution. CRN reduces paired-estimation variance but neither defines nor guarantees the
structural components of the ledger.

<!-- M03 -->
The \(E\) branch is run for every legal current target rather than for one sampled target, and
the outcomes are averaged using target probabilities conditional on engagement. Normalized
weights must sum to one within \(10^{-12}\). This removes current-target sampling error, while
the same-step suffix and future continuation remain stochastic under the frozen policy and
shared uniform tape (Fig. 2).

### 4.3 Same-step and future cost ledger

<!-- M04 -->
For \(B\in\{N,E\}\), let \(C_{t,i}(B)\) and \(C_{t,-i}(B)\) denote current-step costs
for the probe and other units, and let \(C_{>t,i}(B)\) and \(C_{>t,-i}(B)\) denote their
strictly future cumulative costs. Probe direct cost is

\[
C_{\mathrm{direct}}:=C_{t,i}(E)-C_{t,i}(N).
\]

<!-- M05 -->
Same-step other-unit substitution cost is

\[
Sub_{\mathrm{cost,same}}:=C_{t,-i}(N)-C_{t,-i}(E).
\]

This term records the change in other-unit resource cost within the joint-action suffix after
dynamic target occupancy is altered. It cannot be folded into a future-only component.

<!-- M06 -->
Future probe and future other-unit substitution costs are

\[
Sub_{\mathrm{cost,future,probe}}:=C_{>t,i}(N)-C_{>t,i}(E),
\]

\[
Sub_{\mathrm{cost,future,other}}:=C_{>t,-i}(N)-C_{>t,-i}(E).
\]

Total substitution cost and strictly future substituted shots are

\[
\boxed{Sub_{\mathrm{cost,total}}
=Sub_{\mathrm{cost,same}}
+Sub_{\mathrm{cost,future,probe}}
+Sub_{\mathrm{cost,future,other}}},
\]

\[
Sub_{\mathrm{shot}}:=Shots_{>t}(N)-Shots_{>t}(E).
\]

\(Sub_{\mathrm{shot}}\) excludes the same-step suffix and is not interchangeable with
\(Sub_{\mathrm{cost,total}}\).

### 4.4 Episode-cost identity and sign masking

<!-- M07 -->
Every target-conditioned cost record satisfies

\[
\boxed{\Delta C_{\mathrm{episode}}
=C_{\mathrm{direct}}-Sub_{\mathrm{cost,total}}}.
\]

The absolute-error tolerance is \(10^{-6}\) for the total ledger, probe/other
subdecomposition, and N/E protocol residuals. "Exact" refers only to algebraic reconstruction
under these frozen definitions, not to statistical unbiasedness for an arbitrary causal
estimand.

<!-- M08 -->
For \(C_{\mathrm{direct}}>0\), the substitution ratio and cost-sign-masking indicator are

\[
\rho_{\mathrm{sub}}
:=\frac{Sub_{\mathrm{cost,total}}}{C_{\mathrm{direct}}},
\qquad
I_{\mathrm{mask}}
:=\mathbb{1}[C_{\mathrm{direct}}>0\land
\Delta C_{\mathrm{episode}}\le0].
\]

\(\rho_{\mathrm{sub}}\) is a cost ratio, not a probability.
\(I_{\mathrm{mask}}=1\) means only that the episode-level cost difference did not reveal a
positive current direct cost.

### 4.5 Identifiability boundary

<!-- M09 -->
The decomposition is defined by the saved snapshot, local N/E intervention, exact conditional
target marginalization, shared random tapes, and frozen stochastic continuation. During
confirmation, the actor runs in evaluation mode and a no-gradient context, and parameter-wise
comparison before and after evaluation must be zero. The ledger identifies an operational
decomposition under frozen policy responses. It does not identify every unobserved mediator,
alternative unit ordering, or effect in another environment.

## 5. Experimental Protocol

### 5.1 Discovery and independent confirmation

<!-- P01 -->
R1 used policy seeds 8, 9, and 10 for mechanism discovery and problem narrowing; R2 was
reserved for independent confirmation. Preliminary label audits varied target semantics,
deterministic versus stochastic continuation, and short-horizon definitions. Their role was to
narrow the final question to whether cumulative resource cost mixes direct current cost with
downstream substitution, not to serve as the main method (Table S1).

### 5.2 Source models and context independence

<!-- P02 -->
R2 retained seeds 17, 18, and 19 for each of three scenarios, yielding nine factorized joint PPO
source models. All nine entered confirmation without selection by reward, cost, all-noop rate,
or any other behavioral outcome. From 24 candidate episodes per scenario-seed block, the
protocol selected six safety and six resource contexts using frozen slot scores; each resource
slot contained three missile and three laser contexts. Selection did not inspect downstream N/E
cost outcomes, and the complete rule is given in Supplementary Methods S3. The final 108
contexts had zero observation-hash overlap with
the earlier auditable observations. Zero overlap prevents reuse of known states but does not
establish out-of-distribution generalization (Tables 1-2).

### 5.3 Paired sampling and statistical units

<!-- P03 -->
Each context used 32 N/E CRN repeats, with exact probability marginalization over all legal
targets within every repeat. A ledger row is one context-repeat-target record weighted by its
conditional target probability; a repeat is one N/E pair; a context is a frozen state, probe,
and slot; a block is a scenario-policy-seed-slot combination; and a seed is an independently
trained source-policy seed. Context intervals use the 32 repeats as samples, whereas block and
group intervals use contexts. Target ledger rows are not treated as independent contexts.

### 5.4 Statistical intervals and confirmation gates

<!-- P04 -->
For observations \(x_1,\ldots,x_n\), we report the 95% normal-approximation interval
\(\bar{x}\pm1.96s/\sqrt n\). Integrity gates require all nine source models and 108 contexts,
zero old-hash overlap, 32 repeats per context, zero actor-parameter change, no more than 266,198
transitions, and passing software regression. Every target-conditioned record must also have
\(C_{\mathrm{direct}}>0\), with protocol and decomposition errors no greater than
\(10^{-6}\).

<!-- P05 -->
P-C1 tests the complete cost identity, probe/other subdecomposition, and positive direct cost.
P-C2 is restricted to 18 `time_pressure/resource` contexts and jointly tests positive
substitution contexts, block intervals, seed-level masking, and substitution in contexts with
non-positive cumulative cost. P-C3 is stratified by missile and laser. Each resource type must
have a positive context-level 95% lower bound for \(Sub_{\mathrm{shot}}\), positive means for
at least two of three seeds, and mean cost-sign masking in at least three contexts before the
cross-resource gate can pass. All thresholds were frozen before independent confirmation
(Table 3).

### 5.5 Research-integrity handling

<!-- P06 -->
The initial R2 analysis used a future-only identity and produced non-zero residuals in 287 of
7,776 target ledger records, with a maximum residual of 2.0. Inspection showed that current
probe engagement could change same-step other-unit suffix actions through the conflict-free
autoregressive mask, a component absent from the original identity. Adding
\(Sub_{\mathrm{cost,same}}\) reduced the maximum error of the expanded identity to
\(8.88\times10^{-16}\). The original results were archived. Source models, contexts,
environment and policy random tapes, and all gates were held fixed, and the full analysis was
rerun once (Table S3).

## 6. Results

### 6.1 Short-horizon local resource-credit labels were not actionable

<!-- RES-6.1-01 -->
The frozen short-horizon audit did not produce actionable local engagement labels in
`time_pressure/resource`. Across 18 contexts, the safety-resource labels were
`0 ENGAGE / 0 STOP / 18 AMBIGUOUS` (Table S1). This outcome did not distinguish ambiguity
caused by direct current-action expenditure from ambiguity caused by subsequent policy
responses.

<!-- RES-6.1-02 -->
We therefore tested a narrower question: whether episode-level cumulative resource cost can
read out local resource cost for the current action. The analysis was restricted to local N/E
interventions under a frozen policy. Short-horizon label failure was not interpreted as a
performance result for BPCE, PPO, or any online algorithm.

### 6.2 Paired trajectories revealed future shot substitution

<!-- RES-6.2-01 -->
R1 compared the N/E branches in `time_pressure/resource` contexts from earlier policy seeds 8,
9, and 10. Both branches started from the same frozen context, shared random tapes, and exactly
marginalized over legal targets. Substituted shots were defined as
\(Sub_{\mathrm{shot}}=Shots_{>t}(N)-Shots_{>t}(E)\).

<!-- RES-6.2-02 -->
All 18 R1 contexts had a positive mean and positive 95% lower bound for
\(Sub_{\mathrm{shot}}\) (Fig. 3a). Mean future substituted shots were 0.990 and mean future
substitution cost was 1.995. The R1 cost quantity included future substitution only, not the
same-step suffix component later added in R2.

<!-- RES-6.2-03 -->
All 11 R1 contexts with non-positive mean episode-level cost differences had positive future
substitution cost, yielding an explanation count of 11/11 and a maximum decomposition error of
\(4.00\times10^{-15}\) (Fig. 3d). R1 served only as mechanism discovery with earlier seeds;
replication in policy seeds not used for label design was evaluated separately in R2.

### 6.3 Same-step suffix cost was required for a complete ledger

<!-- RES-6.3-01 -->
The initial R2 run revealed that the future-only identity omitted changes in same-step
autoregressive suffix actions. Residuals exceeded \(10^{-6}\) in 287 of 7,776 target-cost
records, with a maximum of 2.0 (Fig. 2d). This omission was not preregistered as a positive
hypothesis before it was observed, and the initial ledger was archived.

<!-- RES-6.3-02 -->
After adding same-step other-unit substitution, mean total substitution cost in
`time_pressure/resource` was 0.864, comprising 0.147 from same-step substitution and 0.718
from future substitution. The future components accounted for approximately 83% of total
substitution cost (Fig. 4a-b). The complete ledger retained the future substitution seen in R1
and added the same-step component absent from the original identity.

<!-- RES-6.3-03 -->
The correction did not change source models, contexts, random tapes, target-marginalization
rules, or statistical gates. Across all 7,776 records, the corrected maximum protocol and
decomposition error was \(8.88\times10^{-16}\), and the maximum actor-parameter change was
zero (Table S3). Exactness here refers only to the record-level algebraic identity.

### 6.4 Action substitution replicated across three new policy seeds

<!-- RES-6.4-01 -->
R2 used all 9 of 9 new source models and all 108 of 108 new contexts, with zero old-context
observation-hash overlap. The evaluation produced 3,456 context-repeat records, 7,776
target-cost ledger records, and 157,485 additional transitions. Maximum actor-parameter change
was zero, and the software regression suite reported 264 passed tests (Tables 1-2).

<!-- RES-6.4-02 -->
Within `time_pressure/resource` for seeds 17, 18, and 19, 13/18 contexts had a positive mean
\(Sub_{\mathrm{shot}}\), and 13/18 had a positive 95% lower bound (Fig. 3b). These counts
exceeded the preregistered thresholds of 12 of 18 positive means and 6 of 18 positive lower
bounds.

<!-- RES-6.4-03 -->
Seed-block means and 95% intervals for \(Sub_{\mathrm{shot}}\) were 0.878 [0.757, 0.999] for
seed 17, 0.260 [0.029, 0.492] for seed 18, and 0.511 [0.166, 0.856] for seed 19
(Fig. 3c). All three lower bounds were positive. Cost-sign-masking rates were 0.969, 0.271,
and 0.526, so two of three seeds had masking rates of at least 50%.

<!-- RES-6.4-04 -->
Seven new-seed contexts had non-positive mean episode-level cost differences, and all seven had
positive \(Sub_{\mathrm{cost,total}}\) (Fig. 3d). These observations support replication of
action substitution across the new source-policy seeds. They do not establish stability across
arbitrary seeds, algorithms, or environments.

### 6.5 Substitution persisted, but sign masking depended on scenario and resource type

<!-- RES-6.5-01 -->
Positive mean substituted shots and substitution cost were observed in all three resource-slot
scenarios, with different aggregate magnitudes (Fig. 5a; Table 4).

| Scenario | Contexts | \(Sub_{\mathrm{shot}}\) | \(Sub_{\mathrm{cost,total}}\) | \(\rho_{\mathrm{sub}}\) | Masking rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| medium | 18 | 0.544 | 0.949 | 0.747 | 0.620 |
| time pressure | 18 | 0.550 | 0.864 | 0.873 | 0.589 |
| heterogeneity pressure | 18 | 0.876 | 1.435 | 0.972 | 0.865 |

These scenario aggregates do not replace seed- or block-level uncertainty and do not support a
ranking across independent environments.

<!-- RES-6.5-02 -->
Within `time_pressure/resource`, both missile and laser contexts had positive mean
\(Sub_{\mathrm{shot}}\) and positive 95% lower bounds, but their cost-offset ratios and masking
counts differed (Figs. 5b-c; Table 4).

| Resource type | Contexts | \(Sub_{\mathrm{shot}}\) | 95% lower bound | \(\rho_{\mathrm{sub}}\) | Masked contexts |
| --- | ---: | ---: | ---: | ---: | ---: |
| missile | 9 | 0.373 | 0.133 | 0.571 | 2 |
| laser | 9 | 0.726 | 0.497 | 1.175 | 5 |

<!-- RES-6.5-03 -->
Mean cost-sign masking occurred in 2/9 missile contexts and 5/9 laser contexts. Missile
therefore fell below the preregistered three-context threshold, and P-C3 failed (Fig. 5d;
Table 3). The failure does not negate positive action substitution for missile contexts. It
limits the claim that both resource types consistently reach the threshold required to reverse
the cumulative-cost sign.

### 6.6 Resource restoration did not yield a broadly applicable safety opportunity label

<!-- RES-6.6-01 -->
R1 added a resource-restoration branch, \(E-R\), to separate current engagement from restoration
of expended ammunition. Engagement, hit outcome, immediate cost, and cooldown matched \(E\);
one unit of probe ammunition was restored before the next policy observation. The comparison
measured only subsequent legal actions and safety outcomes.

<!-- RES-6.6-02 -->
Reliable positive opportunity value occurred in 5 of 18 `time_pressure/resource` contexts and
2 of 18 `heterogeneity_pressure/resource` contexts. Only 1 of 18 safety contexts in each
scenario met the criterion (Table S2). Both reliable heterogeneous-resource contexts came from
seed 9, and all seven reliable resource contexts involved missile units.

<!-- RES-6.6-03 -->
Restoration enlarged some future action sets. Mean future probe reuse and added legal-action
edges were 1.000 and 3.904 in time/resource, and 0.333 and 1.261 in
heterogeneity/resource. These action-set changes did not produce a consistent lower-bound
safety benefit across scenarios, seeds, and resource types. The short-horizon protocol also
remained at `0 ENGAGE / 0 STOP / 18 AMBIGUOUS`.

<!-- RES-6.6-04 -->
The evidence therefore did not support a broadly applicable ammunition opportunity-cost label,
and the opportunity-oracle route was stopped according to its preregistered decision. This does
not imply that every ammunition unit lacks future value. It limits a direct transition from
the offline measurement diagnosis to a shared online auxiliary target. Online BPCE/MCH-PPO and
GNN repair are not positive Results claims in this study.

## 7. Discussion

### 7.1 From episode cost to local resource credit

<!-- D00 -->
The central advance is an auditable measurement object for a known credit problem in dynamically
masked sequential resource allocation. Positive direct expenditure from a current engagement
can be displaced by same-step suffix and future policy responses, so an episode-level cost
difference need not track local current-action resource cost. Ledger closure and positive
substitution across new policy seeds indicate that this mixing was not confined to the
discovery policies. The failed cross-resource gate, however, separates the presence of action
substitution from substitution strong enough to mask the cumulative-cost sign. The evidence
supports a conditional measurement diagnosis, not a claim that every episode cost is a biased
local-credit label.

### 7.2 Structural mixing is not removed by additional rollouts

<!-- D03 -->
Same-step suffix substitution arises from conditional dependence within the joint action. Once
an earlier unit occupies a target, that target is removed from later legal sets. Changing a
probe from no-op to engage can therefore alter actions of other units in the same joint-action
suffix. The systematic residuals of the future-only ledger and record-level closure after
adding the same-step component are consistent with dynamic target occupancy as the direct
structural source of this term. The evidence does not exclude changes in its magnitude under
alternative unit orders or conflict rules.

<!-- D01 -->
Future substitution reflects the joint influence of current engagement on later state, resource
availability, and policy roles. Even with shared exogenous random tapes, the policy can produce
different future actions under branch-specific states and legal masks. This divergence is part
of the policy response being measured, not sampling noise to be removed. Additional repeats or
CRN can reduce Monte Carlo uncertainty for the same estimand, but cannot recover
\(C_{\mathrm{direct}}\) from
\(C_{\mathrm{direct}}-Sub_{\mathrm{cost,total}}\). Sampling precision and measurement
validity are therefore distinct [E15].

### 7.3 Relation to counterfactual credit, sequential decisions, and resource methods

<!-- D02a -->
Difference rewards, COMA, and Shapley-style methods establish counterfactual default actions or
action marginalization as tools for isolating individual contribution [E01-E04]. Our distinction
is that dynamic masking can make a fixed-other-action comparison illegal. The N/E ledger
instead preserves suffix responses under each branch's legal conditional distribution. It is
an operational audit of a specific resource-cost estimand, not a replacement for general
counterfactual baselines.

<!-- D02b -->
Temporal credit and causal decomposition already cover the general idea that downstream actions
mediate current-action effects [E05-E09, E19, E20, E23]. Our narrower role is to separate a
physically interpretable cost into same-step suffix, future probe, and future other-unit
components, and to test when those components are sufficient to mask direct-cost sign. The
ledger complements broader causal effect decompositions but does not identify all state
propagation or unobserved mediator paths.

<!-- D02c -->
Sequential policy updates, autoregressive joint policies, invalid action masking, and fixed-order
team credit all have prior precedent [E12-E14, E21, E22]. The added result is a measurement
consequence: when prefix actions dynamically change legal sets, a local intervention also
changes the joint-action suffix. This conclusion is restricted to the current conflict-free
order 0-1-2 implementation and does not establish order invariance.

<!-- D02d -->
Constrained RL and MARL optimize policy-level cumulative budgets or safety constraints, whereas
our ledger records local N/E cost paths under a frozen policy [E16-E18]. Restoring ammunition
can enlarge a future legal action set, but we did not observe a stable safety opportunity value
across scenarios, seeds, and resource types. More available actions cannot therefore substitute
directly for local resource credit or support a shared opportunity oracle.

<!-- D02e -->
CRN improves the precision of differences between stochastic systems [E15]. We align N/E random
tapes while allowing state- and mask-dependent actions to diverge normally. CRN neither defines
same-step or future substitution nor converts a mixed estimand into a pure local effect. The
measurement rests on the joint constraints of intervention identity, legal branch execution,
exact target marginalization, and the complete ledger.

### 7.4 Scenario and resource type condition cost-sign masking

<!-- D04 -->
Resource-stratified results support a conditional interpretation rather than a cross-type rule.
Both missile and laser contexts showed positive substituted shots, but missile did not reach
the stable sign-masking threshold. The lower direct cost of laser engagement could make complete
offset by same-step and future substitution easier, which is consistent with the stratified
\(\rho_{\mathrm{sub}}\) values. The two resources also differ in capacity, range, hit
probability, and cooldown, so direct cost cannot be isolated as the sole cause. Likewise, the
three scenarios define internal AirDefense v1 pressure conditions, not a ranking across
environments.

### 7.5 Implications for future online methods

<!-- D05 -->
The results impose design requirements on future methods rather than provide a validated
algorithmic answer. An online credit method that supervises engage/no-op decisions with
episode-level cost should separate direct current cost, same-step suffix substitution, and
future policy-mediated substitution, while treating resource type explicitly. BPCE/MCH-PPO has
not produced a stable performance contribution, and a GNN has not been validated as a repair
mechanism. Both remain research questions generated by the present boundaries.

## 8. Limitations

<!-- L01 -->
First, the empirical scope is one AirDefense v1 environment family. Three scenarios and two
resource types are within-environment stress tests, not independent environments, task domains,
or resource mechanisms. P-C3 failed, so the evidence does not establish stable cost-sign
masking for both resource types and cannot be extended to arbitrary dynamic allocation
environments.

<!-- L02 -->
Second, identifiability is restricted to frozen factorized joint PPO, the fixed unit order
0-1-2, conflict-free target occupancy, and a local N/E intervention. CRN, target
marginalization, and stochastic continuation improve reproducibility but do not manipulate
unit order, conflict rules, or alternative policy architectures. The ledger also does not
identify all state-mediated paths, unobserved mediators, or policy-training effects. Block- and
resource-level intervals contain few contexts and use the normal approximation
\(\bar{x}\pm1.96s/\sqrt n\). They operationalize the frozen confirmation gates rather than
provide high-precision inference for a broader population of environments or policies.

<!-- L03 -->
Third, conclusions depend on the episode-level resource cost and the missile/laser expenditure
definitions used here. Ledger closure does not imply safety benefit, mission return, or
ammunition opportunity value. Resource restoration did not yield a stable opportunity label,
and online BPCE/MCH-PPO did not pass stability and performance gates. These negative results
limit movement from offline diagnosis to a shared online auxiliary signal, but do not imply
that ammunition lacks future value in every context.

<!-- L04 -->
Fourth, independent confirmation still covers only three new policy seeds per scenario and one
source algorithm. Cross-algorithm, action-order, environment, and real-system tests remain
absent, and a GNN has not been evaluated. Until these questions pass separate gates, the
conclusions stop at measurement, mechanism confirmation, and conditional boundaries under a
frozen policy.

## 9. Conclusion

<!-- C01 -->
We operationalized local resource credit in dynamically masked sequential allocation as a
paired counterfactual measurement audit. N/E pairing, common random numbers, exact target
marginalization, and a stepwise ledger separated direct current cost from same-step other-unit,
future probe, and future other-unit substitution. The complete three-component ledger
reconstructed episode-level cost differences across 7,776 target-conditioned records and
confirmed positive substitution across three new policy seeds. These results show that more
rollouts can improve paired precision but cannot turn episode cost containing policy responses
into a pure local-cost readout. Missile contexts did not reach the cross-resource sign-masking
threshold, so the claim is limited to measurement, mechanism confirmation, and conditional
boundaries in three internal AirDefense v1 scenarios under frozen factorized joint PPO. It is
not a validated online PPO improvement and does not support a shared opportunity-cost oracle
or GNN repair claim.

## Data and Code Availability

The simulation configurations, model manifests, paired counterfactual ledgers, derived summary
tables, figure source data, environment implementation, and audit code supporting this study
are maintained in the project repository and mapped to the manuscript through reproducibility
and traceability files. At the W1-10 freeze, these materials have not been deposited in a public
repository and no DOI, accession identifier, or formal licence has been assigned. This final
scientific draft therefore does not claim public data or code availability. Release scope,
versioned archives, licences, and persistent identifiers must be fixed before external
submission.

## References (E-ID placeholders)

Reference formatting will be completed after a target journal is selected. E-IDs map directly
to `literature_evidence_matrix.md`.

1. [E01] Wolpert & Tumer (2002), Difference rewards.
2. [E02] Foerster et al. (2018), COMA.
3. [E03] Wang et al. (2020), Shapley Q-value.
4. [E04] Nguyen et al. (2018), global-reward credit assignment.
5. [E05] Castellini et al. (2022), difference-return policy gradients.
6. [E06] Mesnard et al. (2021), counterfactual credit assignment.
7. [E07] Harutyunyan et al. (2019), hindsight credit assignment.
8. [E08] Meulemans et al. (2023), COCOA.
9. [E09] Arjona-Medina et al. (2019), RUDDER.
10. [E10] Wu et al. (2018), action-dependent factorized baselines.
11. [E11] Tucker et al. (2018), action-dependent baseline analysis.
12. [E12] Kuba et al. (2022), HAPPO/HATRPO.
13. [E13] Wen et al. (2022), Multi-Agent Transformer.
14. [E14] Huang & Ontañón (2020), invalid action masking.
15. [E15] Glasserman & Yao (1992), common random numbers.
16. [E16] Achiam et al. (2017), constrained policy optimization.
17. [E17] Gu et al. (2021), multi-agent constrained policy optimization.
18. [E18] Zhang et al. (2024), scalable constrained MARL.
19. [E19] Triantafyllou et al. (2024), agent-specific effects.
20. [E20] Triantafyllou et al. (2025), counterfactual effect decomposition.
21. [E21] Deshmukh et al. (2026), CAPO.
22. [E22] Li et al. (2026), CCPO.
23. [E23] Zhang et al. (2023), causal reward decomposition.
24. [E24] Li et al. (2026), counterfactual Shapley credit.
