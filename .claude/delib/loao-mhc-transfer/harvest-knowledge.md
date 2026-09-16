vault_matches:
  - page: /Users/freddy/Documents/vault/conversations/2026-09-06_ce6515be_tcr-epitope-immrep23-baseline-build.md
    matched_keyword: TCR
    prior: prior-only
    scope: >-
      "The task is string-pair binary classification: given a peptide (short protein fragment, ~9 amino acids) and a CDR3b (the ~12-letter binding tip of a T-cell receptor beta chain), predict whether they bind."
    components:
      - '"peptide (short protein fragment, ~9 amino acids)"'
      - '"CDR3b (the ~12-letter binding tip of a T-cell receptor beta chain)"'
      - '"k-NN lookup baseline"'
      - '"ESM-2 embedding cache"'
    edges:
      - >-
        "Features must include interaction blocks [p, t, |p−t|, p⊙t]. A linear model on plain concat([p, t]) computes w_p·p + w_t·t; within one peptide the first term is constant, so the induced TCR ranking is identical for every peptide — structurally unable to express peptide-specific binding."
      - >-
        "Split on connected components of the bipartite peptide–TCR graph, not on peptides."
    status: >-
      "k-NN's 0.500 on unseen peptides is arithmetic, not measurement. Its database is empty for unseen peptides, so all 1,066 rows take one constant score, and a constant column scores exactly 0.5 by construction."
  - page: /Users/freddy/Documents/vault/conversations/2026-09-07_aa30fd7b_cognate-phase-b-vdjdb-eval-b3-verdicts.md
    matched_keyword: cognate
    prior: prior-only
    scope: >-
      "Continuation of a TCR–epitope binding-prediction project on the IMMREP23 benchmark. The task is string-pair binary classification: given a peptide (~9 amino acids) and a CDR3b (the ~12-letter binding tip of a T-cell receptor beta chain), predict whether they bind."
    components:
      - '"88-peptide evaluation set built from VDJdb"'
      - '"48 seen in IMMREP23 training, 40 unseen"'
      - '"19 alleles, HLA-A\\*02:01 on 30 of 88 peptides"'
    edges:
      - >-
        "Evaluation set drawn from VDJdb, with every IMMREP23 training TCR removed — not just exact pairs. Rejected: removing only verbatim (peptide, CDR3b) pairs."
      - >-
        "Negatives assigned per peptide, 5 per positive, mirroring the IMMREP23 organisers' construction. Rejected: uniform sampling of negative peptides."
    status: >-
      "Unseen: +0.000 [−0.006, +0.005], p = 0.994." 
  - page: /Users/freddy/Documents/vault/conversations/2026-09-07_c1979175_cognate-phase-d-two-forks-comparison.md
    matched_keyword: cognate
    prior: prior-only
    scope: >-
      "Phase D work order for cognate (TCR–epitope binding prediction)."
    components:
      - '"k-NN, edit distance (BASELINE)"'
      - '"fork 1 · 1b two-tower"'
      - '"fork 2 · 2a cross-attention"'
      - '"fork 2 · 2b mean-pool CONTROL"'
    edges:
      - >-
        "1b is the only construction with a mechanism on unseen peptides — ~43,190 distinct scores where k-NN produces exactly 1."
      - >-
        "2a/2b differ by exactly one boolean. Same projections, pooling, interaction features, MLP, optimiser, split, negatives, seeds. Rejected: two separately-written models, which would have made 2a − 2b uninterpretable."
    status: >-
      "Nothing has ever scored above chance on unseen peptides."
session_matches: []
unmatched_keywords:
  - pmhc
  - leave-one-allele-out
  - transfer
  - HLA
  - allele
