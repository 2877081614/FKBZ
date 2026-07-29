# W1-09 English Manuscript Reverse Outline

Updated: 2026-07-28  
Manuscript: `manuscript_draft_en.md`  
Rule: one primary job per Paragraph ID

## 1. Abstract, Introduction, and Related Work

| Paragraph ID | Opening sentence | Single job | Claim ID | Evidence ID | Relation to previous paragraph |
| --- | --- | --- | --- | --- | --- |
| A01 | Dynamic resource allocation requires team outcomes to be translated into local resource credit | summary | C1-C4, C7 | EV-R2-01 to EV-R2-13, BD-03 | whole-paper compression |
| I01 | Dynamic resource allocation requires a policy to balance mission outcomes against finite resource expenditure | context | — | AirDefense v1 | expands the object |
| I02 | MARL commonly estimates local contributions from team or counterfactual returns | known gap | C1 | E01-E09, E19, E20 | moves to known credit problem |
| I03 | This question becomes concrete in autoregressive joint actions with dynamic masking | exact gap | C1, C2 | E12-E14, E21 | narrows the bottleneck |
| I04 | We address this question with a paired N/E audit | approach | C2 | T04-T18, EV-R2-06 | answers the gap |
| I05 | The audit separates mechanism discovery from independent confirmation | evidence preview | C3, C4 | EV-R2-01, EV-R2-08 to EV-R2-13 | previews evidence and boundary |
| I06 | The study makes three bounded contributions | contribution | C1-C4, C6-C8 | W1-02 L2 | freezes scope |
| RW01 | Counterfactual methods establish local credit from team outcomes | comparison | C1 | E01-E05 | names direct precedents |
| RW02 | Temporal and causal methods address downstream behavioral mediation | comparison | C1, C2 | E06-E09, E19, E20, E23 | extends to temporal precedents |
| RW03 | Sequential policies and dynamic masking have prior precedent | comparison | C2 | E12-E14, E21, E22 | isolates the suffix gap |
| RW04a | Constrained MARL controls policy-level cumulative cost | comparison | C5 | E16-E18 | separates budgets from attribution |
| RW04b | CRN reduces variance in paired simulation | comparison | C2 | E15 | separates precision from validity |
| RW05 | Prior work covers the broad concepts used here | positioning | C2, C4, C7 | W1-02 L2 | positions the measurement module |

## 2. Problem Formulation, Method, and Protocol

| Paragraph ID | Opening sentence | Single job | Claim ID | Evidence ID | Relation to previous paragraph |
| --- | --- | --- | --- | --- | --- |
| PF01 | We study dynamic allocation in AirDefense v1 | context | — | environment design, Table 1 | defines the task |
| PF02 | The source policy is factorized joint PPO | method | C1 | T01-T03 | defines the conditional policy |
| PF03 | We freeze a context and compare two local branches | estimand | C1 | T04-T15 | defines the measurement |
| PF04 | Confirmation covers three configurations in one environment family | limitation | C7, C8 | BD-01, BD-02 | bounds the scope |
| M01 | Each context stores a complete snapshot for N/E replay | method | C1, C2 | T04-T08 | implements the intervention |
| M02 | Each repeat uses paired random-number tapes | variance control | C2 | T05, E15 | controls stochastic variation |
| M03 | Every legal target is exactly marginalized | integration | C2 | EV-R2-03 | removes target-sampling error |
| M04 | Probe direct cost is the current-step E-N difference | definition | C2 | T08 | defines direct cost |
| M05 | Same-step substitution uses the N-E saving direction | definition | C2 | T09 | adds the suffix component |
| M06 | Future substitution separates probe and other units | definition | C2 | T10-T15 | completes the ledger |
| M07 | Every target-conditioned record satisfies the complete identity | identity | C2 | EV-R2-06 | closes the algebra |
| M08 | Ratio and sign masking are defined for positive direct cost | metric | C4 | T16, T17 | defines boundary metrics |
| M09 | Identification is operational under frozen policy responses | limitation | C1-C4 | T18, E19, E20 | limits causal interpretation |
| P01 | R1 discovers and R2 independently confirms | protocol | C1, C3 | EV-R1-01, EV-R2-01 | separates evidence roles |
| P02 | R2 retains all nine new source models | independence | C3 | EV-R2-01, EV-R2-02 | establishes confirmation independence |
| P03 | Each context uses 32 paired repeats | sampling | C2, C3 | EV-R2-03 | freezes statistical units |
| P04 | Intervals and integrity gates are frozen | statistics | C2, C3 | EV-R2-06, EV-R2-07 | defines quality control |
| P05 | P-C1, P-C2, and P-C3 test identity, replication, and resource boundary | gates | C3-C5 | preregistered gates | defines decisions |
| P06 | The initial future-only identity omitted same-step suffix cost | integrity | C2 | EV-R2-04 to EV-R2-07 | discloses the correction |

## 3. Results

| Paragraph ID | Opening sentence | Single job | Claim ID | Evidence ID | Relation to previous paragraph |
| --- | --- | --- | --- | --- | --- |
| RES-6.1-01 | The short-horizon audit produced no actionable label | negative result | C5 | EV-BPCE-01 | motivates narrowing |
| RES-6.1-02 | We therefore tested a narrower cumulative-cost question | scope | C7 | BD-01 | narrows the target |
| RES-6.2-01 | R1 compared paired branches under earlier seeds | setup | C1 | EV-R1-01 | begins discovery |
| RES-6.2-02 | All 18 R1 contexts had positive substitution lower bounds | result | C1 | EV-R1-01 | reports discovery |
| RES-6.2-03 | All non-positive-cost contexts had positive future substitution | result | C1 | EV-R1-02 | links substitution and mixing |
| RES-6.3-01 | The initial R2 run exposed the omitted same-step component | integrity result | C2 | EV-R2-04, EV-R2-05 | motivates complete ledger |
| RES-6.3-02 | The complete ledger separated same-step and future cost | result | C2 | EV-R2-13 | reports composition |
| RES-6.3-03 | Corrected records closed algebraically with a frozen actor | validation | C2 | EV-R2-06, EV-R2-07 | validates integrity |
| RES-6.4-01 | R2 used all new models and contexts | independence | C3 | EV-R2-01 to EV-R2-03 | establishes confirmation sample |
| RES-6.4-02 | Thirteen of 18 new contexts had positive lower bounds | result | C3 | EV-R2-08 | confirms at context level |
| RES-6.4-03 | All three seed-block lower bounds were positive | result | C3 | EV-R2-09 | confirms at seed level |
| RES-6.4-04 | All seven non-positive-cost contexts had positive total substitution | result | C1, C3 | EV-R2-10 | confirms mechanism consistency |
| RES-6.5-01 | All three scenarios showed positive substitution with different magnitudes | boundary result | C4 | EV-R2-13 | tests scenario boundary |
| RES-6.5-02 | Both resources showed substitution but different offset strength | boundary result | C4 | EV-R2-11, EV-R2-12 | compares resource types |
| RES-6.5-03 | Missile failed the P-C3 masking threshold | failed gate | C4 | BD-03 | rejects cross-resource scope |
| RES-6.6-01 | Resource restoration separated engagement from ammunition availability | setup | C5 | EV-R1-03 | defines opportunity audit |
| RES-6.6-02 | Reliable opportunity value appeared in few contexts | negative result | C5 | EV-R1-03 | reports insufficient coverage |
| RES-6.6-03 | Larger action sets did not yield consistent safety benefit | negative result | C5 | EV-R1-03, EV-BPCE-01 | rules out action-count proxy |
| RES-6.6-04 | The shared opportunity-oracle route was stopped | decision | C5, C7 | BD-01 | freezes negative conclusion |

## 4. Discussion, Limitations, and Conclusion

| Paragraph ID | Opening sentence | Single job | Claim ID | Evidence ID | Relation to previous paragraph |
| --- | --- | --- | --- | --- | --- |
| D00 | The advance is an auditable measurement object | implication | C1-C4 | EV-R2-06, EV-R2-09, BD-03 | interprets the central result |
| D03 | Same-step substitution arises from joint-action dependence | mechanism | C2 | EV-R2-04, EV-R2-05, E14 | interprets suffix mixing |
| D01 | Future substitution reflects later state and policy-role changes | mechanism | C1, C2 | EV-R2-06, E15 | separates structure from variance |
| D02a | General counterfactual credit has direct precedents | comparison | C1, C2 | E01-E04 | limits methodological novelty |
| D02b | Downstream mediation has temporal and causal precedents | comparison | C1, C2 | E05-E09, E19, E20, E23 | limits theoretical novelty |
| D02c | Sequential policies and masking have precedents | comparison | C2 | E12-E14, E21, E22 | limits action-structure novelty |
| D02d | Constraint optimization and local attribution are different estimands | comparison | C5 | E16-E18, EV-R1-03 | interprets opportunity failure |
| D02e | CRN controls paired-difference variance only | comparison | C2 | E15 | rules out variance confounding |
| D04 | Resource stratification supports only a conditional explanation | mechanism boundary | C4 | EV-R2-11 to EV-R2-13 | interprets P-C3 failure |
| D05 | The evidence provides design requirements, not an online algorithm | implication | C6-C8 | BD-01, BD-02, EV-BPCE-02 | stops algorithmic extrapolation |
| L01 | Empirical scope is one environment family | limitation | C3, C4 | BD-03 | states environment boundary |
| L02 | Identifiability is limited to one policy and action order | limitation | C1, C2 | T18, E19, E20 | states intervention boundary |
| L03 | Conclusions depend on the current resource-cost definition | limitation | C5-C7 | EV-R1-03, EV-BPCE-02 | states cost and algorithm boundary |
| L04 | Confirmation does not cover algorithms or environments | limitation | C8 | BD-02 | states transfer and GNN boundary |
| C01 | We operationalized local resource credit as a paired audit | conclusion | C1-C4, C6-C8 | EV-R2-06, EV-R2-09, BD-01 to BD-03 | contribution-evidence-meaning-boundary |

## 5. Check Result

- All 66 Paragraph IDs have one primary function.
- English paragraphs preserve the Chinese claim, evidence, condition, and limitation, but use
  section-specific English argument order.
- Results report observations and gate outcomes; mechanism language is confined to Discussion.
- No additional scientific claim appears in the English reverse outline.
