# AtomFormer: Open-Vocabulary Odor Retrieval for Arbitrary Molecules and Mixtures

**Abstract.** Odor prediction has relied on two implicit assumptions: that a closed set of expert-defined labels is necessary for training, and that individual molecules must serve as the encoding unit—forcing mixtures to be decomposed into constituents, each predicted separately, then aggregated through ad hoc fusion mechanisms. Here we show that both assumptions can be eliminated simultaneously. We introduce AtomFormer, which uses graph isomorphism convolutional layers (GIN) as a molecular tokenizer to encode atomic features into chemically informed tokens. By operating at the atomic level, the model accepts any number of molecules (N = 1, 2, 3, …) in a single forward pass without distinguishing molecular boundaries—the Transformer's self-attention mechanism enables atoms from different molecules to interact directly, eliminating the need for separate mixture aggregation. On a rigorously decontaminated independent test set of 6,305 binary mixture pairs, AtomFormer achieves macro-AUROC = 0.9347 with a single 16.9-million-parameter model. Ablation experiments confirm that GIN provides chemically interpretable tokens for the Transformer, while the Transformer enables cross-molecule prediction—the model's atomic-level encoding naturally supports multi-molecule inputs without special adaptation. These results demonstrate that combining a local chemical tokenizer (GIN) with a global feature integrator (Transformer) provides an effective paradigm for multi-molecule property prediction beyond traditional molecular-level encoding.

---

## 1. Introduction

### 1.1 Two Implicit Assumptions

Odor prediction—the task of mapping molecular structure to perceptual odor description—has been a central challenge in computational olfaction for over a decade. Despite significant progress, the field has operated under two implicit assumptions that have fundamentally constrained model design, data requirements, and practical applicability.

**Assumption 1: A closed, expert-defined label set is required for supervised odor prediction.** From the DREAM challenge's 20 descriptor labels [Keller et al., 2017] to the Principal Odor Map's (POM) 55 labels [Lee et al., 2023, *Science*] and Sisson et al.'s 74 labels [Sisson et al., 2025, *ACS Omega*], every existing system has required a fixed vocabulary of odor categories. Adding new descriptors or adapting to a different cultural odor classification system requires re-annotation of training data and retraining of the model. This closed-label paradigm fundamentally limits the scalability of odor prediction systems to new descriptor sets.

**Assumption 2: Individual molecules must serve as the fundamental encoding unit.** All existing approaches—POM's graph neural network [Lee et al., 2023], Sisson et al.'s GIN and MPNN architectures [Sisson et al., 2025], POMMIX's mixture adaptation [Tom et al., 2025], and Sanchez-Lengeling et al.'s machine olfaction framework [Sanchez-Lengeling et al., 2019]—take complete molecules as input units and encode them independently. When applied to mixtures, these approaches must decompose the blend into constituent molecules, embed each separately, and then combine the individual predictions through separate fusion mechanisms such as averaging or learned attention over molecular embeddings. This decomposition loses cross-molecule interactions at the atomic level—a hydroxyl group from molecule A interacting with an aldehyde group from molecule B may produce a perceptual effect that is not simply the sum of their individual contributions.

The practical consequences of these two assumptions are significant. Closed-label systems cannot generalize beyond their training vocabulary, so comparing predictions across different label systems requires manual mapping. Similarly, molecular-level encoding treats mixtures as second-class inputs—practical applications in perfumery and flavor development, where blends of three to fifty molecules are the norm, require additional engineering beyond the core prediction model.

### 1.2 Prior Work

**Existing odor prediction models.** Efforts to predict odor from molecular structure have a long history. Early work by Omatu et al. [2012] applied neural networks to odor classification, while the DREAM challenge [Keller et al., 2017] established the foundational benchmark framework for structure-odor prediction, tasking teams with predicting 20 descriptor labels for a curated set of molecules. The winning approach used a random forest on computed molecular fingerprints. Poivet et al. [2018] proposed a medicinal chemistry approach to functional odor classification, demonstrating that expert-driven feature engineering could complement purely data-driven methods. POM [Lee et al., 2023] scaled this to 55 labels with a graph neural network ensemble, showing that learned molecular representations outperform fixed fingerprints. Sisson et al. [2025] systematically compared GIN and MPNN architectures, finding that GIN achieves competitive AUROC (0.852 on 74 labels) with far fewer parameters. POMMIX [Tom et al., 2025] extended POM to mixture-mixture similarity by encoding each component independently and aggregating through statistical pooling. However, all these methods operate within fixed label sets and require molecular-level encoding with separate mixture aggregation—neither assumption has been challenged.

**Molecular tokenization and multi-molecule representation learning.** Recent work in molecular representation learning has independently converged on a key insight: the functional group or fragment, rather than the whole molecule, is the appropriate semantic unit for molecular perception. Multiple approaches [Wang et al., 2026; Yan et al., 2025; Zhu et al., 2025] have explored learning from fragment-level representations at various granularities, reflecting a broader shift from whole-molecule encoding toward tokenized molecular representations. This convergence mirrors the trajectory of vision transformers, where ViT [Dosovitskiy et al., 2021] demonstrated that patch embeddings—rather than pixel-level processing—provide the right semantic granularity for visual reasoning. We adopt the same philosophy: GIN layers serve as a molecular tokenizer, converting raw atomic features into chemically meaningful tokens that capture local structural environments.

GNN-based approaches have also emerged for mixture property prediction in other domains, such as fuel mixtures [Leenhouts et al., 2025], general molecular mixtures [Zhang et al., 2024], and drug response prediction [Partin et al., 2026], further highlighting the broad interest in multi-molecule prediction tasks. However, these methods typically rely on pooling molecule-level representations before prediction, an operation that may discard information critical to highly nonlinear phenomena such as odor perception.

**Text embeddings as retrieval backbones.** Kurfalı et al. [2025, *Cognition*] found that word embedding spaces encode olfactory semantic relationships, suggesting that general-purpose text embeddings may already capture odor-relevant structure without specialized training. The rapid development of large language models has produced general-purpose text embedding models with strong performance on diverse semantic tasks. This raises a fundamental question: rather than training a classifier on a fixed label set, can we simply predict the embedding of a molecule's odor description and retrieve the nearest text from a library? We argue that this is feasible when the vocabulary consists of sufficiently specialized domain terms, where text co-occurrence artifacts are minimized.

Here we introduce AtomFormer, a model that natively supports odor prediction for arbitrary numbers of molecules. Its design combines three elements: a molecular tokenizer (GINConv × 3) that encodes each atom's local chemical environment through learned message passing; a Transformer encoder (× 2, 8 attention heads) that enables cross-molecule attention across all atoms without distinguishing molecular boundaries; and an LLM embedding retrieval backbone that replaces fixed-label classification with nearest-neighbor search in a continuous semantic space. The model achieves macro-AUROC = 0.9347 on a rigorously decontaminated binary mixture test set with stable performance across training conditions.

## 2. Retrieval Backbone, Model Architecture, Data, and Evaluation

### 2.1 Quality of the BGE-M3 Embedding Space

BGE-M3 [Chen et al., 2024] is a general-purpose text embedding model based on the Transformer architecture, trained on large-scale multi-lingual data to produce 1024-dimensional embeddings. Using a general-purpose embedding space for odor prediction, however, requires attention to several potential pitfalls.

A known risk with general-purpose text embeddings is that similarity can reflect textual co-occurrence rather than genuine perceptual relationships. However, the TGSC dataset used in this study consists of professional odor descriptors curated from The Good Scents Company, where multi-word descriptions are ordered by perceptual dominance. Cross-modal terms such as "sweet" are rare in this dataset (only 3 compounds) and were handled by promoting subsequent words. The resulting vocabulary consists of specialized odor descriptors that appear predominantly in perfumery rather than general-domain corpora, making their embedding distances more reliable as proxies for perceptual similarity.

Moreover, text embedding models encode multi-word descriptions with implicit position weighting—earlier words contribute more to the overall embedding vector—which aligns with the standard annotation convention in perfumery, where descriptors are ordered by perceptual dominance (first word = most prominent note). As a result, grouping compounds by their first-word descriptor captures the primary perceptual note in a way that is consistent with both the embedding geometry and human annotation practice. To verify this, we constructed 123 odor groups from the TGSC database—each containing all compounds sharing the same first-word descriptor. Using group centroids as prototypes for one-vs-rest classification, the multi-descriptor approach achieved AUROC = 0.9825, with mean intra-group cosine 0.7907 and inter-group cosine 0.5690 (separation 0.2217). These results confirm that BGE-M3 effectively separates professional odor categories without olfactory fine-tuning. We note that the publicly available pyrfume/GoodScents release provides cleaner text but discards the original descriptor order, which is critical for models relying on position-weighted embeddings. We therefore opted to crawl and clean the data directly from The Good Scents Company website, preserving the original annotation order at the cost of additional cleaning effort.

**High-dimensional spaces preserve fine-grained odor detail.** As discussed above, compounds sharing the same first-word descriptor are grouped into the same odor category. However, certain broad categories contain molecules whose descriptor words span a remarkably wide semantic range. For example, alpha-acetoxystyrene carries the description "animal, floral, fecal, phenolic"—four terms spanning entirely different olfactory dimensions. In traditional classification approaches, these secondary odor dimensions are discarded by construction. In a high-dimensional embedding space, however, these differences are preserved and reflected in spatial coordinates—the intra-group cosine of 0.79 (rather than >0.9) indicates that secondary odor dimensions remain distinguishable. This is consistent with studies showing that olfactory space requires high-dimensional representations [Magnasco et al., 2015; Lee et al., 2023]. Our choice of 1024 dimensions may be somewhat redundant, but empirically it introduces no negative effects.

**Semantic continuity in high-dimensional spaces.** Beyond preserving fine-grained odor detail, high-dimensional spaces permit smooth interpolation between arbitrary odor points. Traditional olfactory classification systems—from Linnaeus's seven categories to Piesse's musical scale analogy, Crocker-Henderson's four dimensions, and Jellinek's prism model—support at most two-dimensional odor maps. The example of a single molecule carrying multiple widely divergent descriptors (alpha-acetoxystyrene) suggests that odor transitions cannot be adequately captured in low dimensions and that interpolation between arbitrary odor categories should be permitted. High-dimensional spaces support this: the linear path between any two odor vectors remains between them without collapsing to an unrelated third category. We verified this on 4,500 interpolation points across 500 random odor group pairs—in every case, the top-2 nearest neighbors included the two endpoint categories. This property is essential for the continuous retrieval paradigm, where predictions must be able to fall anywhere in the semantic space rather than at discrete label positions.

### 2.2 Model Architecture

AtomFormer operates directly on atomic graphs. For any input—a single molecule, a binary mixture, or a complex blend—all atoms from all constituent molecules are concatenated into a single graph with no molecular boundary annotations. The architecture consists of four stages:

**Stage 1: Atom feature encoding.** Each atom is represented by a 9-dimensional feature vector encoding:
- **Element type** (8 dimensions, one-hot): C, N, O, F, S, Cl, Br, I
- **Chirality** (2 dimensions, one-hot): tetrahedral CW, tetrahedral CCW
- **Degree** (1 dimension, normalized by /5)
- **Formal charge** (1 dimension, normalized by /5)
- **Ring membership** (1 dimension, binary)

Zero-padding expands these 9 features to 64 dimensions to match the GIN hidden dimension. These 9-dimensional features capture basic atomic properties that are universally available from the molecular graph, without requiring precomputed descriptors or external chemical databases.

**Stage 2: Molecular tokenizer (GINConv × 3, hidden = 64).** Three layers of graph isomorphism convolution [Xu et al., 2019] perform message passing along chemical bonds. Each GINConv layer updates atom representations by combining each atom’s features with the summed features of its neighbors through a learnable MLP (two Linear–BatchNorm–ReLU blocks) with a residual connection controlled by a learned epsilon parameter. With three layers, each atom's representation encodes its local structural environment within approximately 3 chemical bonds—sufficient to capture most functional groups (hydroxyl, carbonyl, ester, aromatic rings, etc.). We conceptualize these GIN layers as a **molecular tokenizer**, analogous to ViT's patch embedding [Dosovitskiy et al., 2021]: raw atomic features (analogous to pixels) are transformed into chemically informed token vectors (analogous to patch embeddings) through message passing.

**Stage 3: Linear projection and Transformer encoder.** A linear layer projects each atom's 64-dimensional representation to 1024 dimensions. Two Transformer encoder layers [Vaswani et al., 2017] with 8 attention heads perform all-to-all attention among all atom tokens. Because the input graph contains no molecular boundary information, atom tokens from different molecules can attend to each other freely—this is the critical architectural innovation that eliminates Assumption 2. The Transformer uses batch-first attention with key padding masking for efficient variable-length sequence handling. We omit positional encoding in the Transformer, ensuring that different SMILES representations of the same molecule produce identical outputs—a desirable invariance property for molecular graphs [Vinyals et al., 2016].

**Stage 4: Pooling, normalization, and training objective.** A masked mean pooling operation averages the variable-length sequence of atom representations over real atoms only (excluding padding tokens). The pooled vector is L2-normalized and trained via Mean Squared Error (MSE) to match the BGE-M3 embedding of the input's odor description.


The L2-normalized prediction is trained via MSE against the BGE-M3 embedding of the corresponding odor description, aligning the model output with the retrieval library's metric space. During inference, the predicted embedding is compared against all library embeddings via cosine similarity; the top-100 nearest descriptions are aggregated through temperature-weighted consensus voting (T = 0.1), where each neighbor contributes a softmax-weighted vote for its first-word descriptor, and the highest-weighted words are reported as the prediction.

**Training details.** We use the Adam optimizer with learning rate 1 × 10⁻³, batch size 1024, and 60 total epochs. A linear warmup schedule increases the learning rate from 0 to 1 × 10⁻³ over the first 3 epochs, followed by cosine annealing (T_max = 30) [Loshchilov & Hutter, 2017]. The warmup was found to be critical: without it, early training was unstable and the model's performance decreased by approximately 5 percentage points. A WeightedRandomSampler applies a 100:1 weight ratio for monomers vs. blender pairs, compensating for the fact that monomer training data (3,690 compounds) is orders of magnitude smaller than blender data (553,160 pairs). The final model has 16.9 million trainable parameters.

### 2.3 Comparison Models

To isolate the contribution of each architectural component, we trained two ablated variants on the same TGSC data:

- **GIN-only (no Transformer):** Removes the Transformer encoder entirely. The 64-dimensional atom representations from the GIN layers are projected to 1024 dimensions and pooled via simple scatter mean (no masked mean pool). This model tests whether the Transformer's cross-molecule attention is necessary for mixture prediction, or whether the GIN's local features alone suffice.

- **TFM-only (no GIN):** Removes the GIN layers entirely. Raw 9-dimensional atom features are projected directly to 1024 dimensions via a linear layer, then processed by the Transformer encoder (×2, 8 heads, no positional encoding). This model tests whether the Transformer can learn chemical representations from scratch without the inductive bias of message passing, or whether the molecular tokenizer (GIN) provides essential chemical priors.

For the classification experiments in Section 3.3, we additionally trained:
- **GIN-only (classification):** A GIN×5 network with Set2Set readout followed by a 135-class binary cross-entropy classifier, replicating the GIN architecture of Sisson et al. [2025] as a classification baseline.
- **GIN+TFM+cls:** The same GIN+Transformer backbone as the full model, but with the MSE→BGE head replaced by a 135-class binary cross-entropy classifier.

### 2.4 Datasets

**Monomer training: TGSC.** Monomer data was obtained directly from The Good Scents Company website [Leffingwell, 2001]. Each compound listing includes a full odor description with descriptor words ordered by perceptual dominance (first word = most prominent note), preserving the original annotation convention. The dataset contains 3,670 compounds after cleaning, each with a 1024-dimensional BGE-M3 embedding computed from its original description. Compounds were split into training (3,434) and test (236), with test molecules selected to have no overlap with blender training data. Since the model operates through semantic embedding alignment rather than exact token matching, perfect text cleaning is not required—minor residual artifacts in the descriptions do not materially affect the learned embeddings or downstream prediction performance.

**GoodScents (classification experiments).** For the classification experiments in Section 3.3, we use the pyrfume/GoodScents dataset, which also originates from The Good Scents Company but has been thoroughly cleaned. A side effect of this cleaning is the loss of descriptor order information (discussed in Section 2.1), making it better suited for classification than for position-sensitive retrieval. The dataset provides 3,090 monomer compounds with first-word descriptor labels drawn from a 135-category vocabulary (MIST_LABELS), split into 2,637 training and 340 test molecules.

**Mixture training: TGSC Blender.** The TGSC blender dataset was crawled and cleaned from The GoodScents Company website. The TGSC Blender database provides 559,465 binary mixture pairs (A + B → odor group label). Each entry lists two components by SMILES and CAS number, along with an odor group classification for the resulting blend. We apply standard RDKit canonical SMILES deduplication to ensure consistent molecular representations. The training uses all 553,160 pairs with a 100:1 WeightedRandomSampler relative to monomers.

A notable characteristic of this dataset is that approximately 73% of reciprocal pairs (A+B and B+A) carry different odor group labels. We do not treat these as errors—they likely reflect blends with different component ratios, where the dominant aroma note shifts depending on which component is listed first.

**Blender independent test set.** We constructed a rigorously decontaminated test set of 6,305 binary mixture pairs. The test set construction ensured that no sorted (A,B) key appears in the training set, preventing both A+B and B+A leakage. The resulting pairs span 108 unique odor groups (104 with sufficient samples for AUROC computation). To ensure clean AUROC evaluation, the test set excludes A+B/B+A conflicting pairs, so each pair appears in only one direction. The test set distribution closely mirrors the training set: the Spearman correlation between training and test odor group proportions is r = 0.74 (p < 10⁻¹⁸), confirming that the test set is representative across common and rare odor categories.

**Data preparation note.** All SMILES strings are standardized via RDKit canonical SMILES throughout the pipeline. Molecules that fail RDKit parsing are excluded. 

### 2.5 Evaluation Metrics

**Primary metric: macro-AUROC.** For the blender independent test set, we follow the established protocol of POM [Lee et al., 2023] and Sisson et al. [2025]: for each odor group present in the test set, we compute the one-vs-rest AUROC, then average across groups. The predicted representation z is compared against group centroids (average embedding of all library descriptions belonging to each group), and the negative of cosine distance is used as the ranking score. This enables direct comparison with published results.

**Consensus retrieval (conR@K).** For single-molecule evaluation (TGSC, GS test), we use temperature-weighted consensus voting. From the top-100 nearest neighbors in the retrieval library, each neighbor's contribution is weighted by softmax(cosine_similarity / T) with T = 0.1 (determined by hyperparameter search across T ∈ {0.05, 0.1, 0.2, 0.5, 1.0, 5.0}). The first word of each neighbor's description receives its weight, and the top-K highest-weighted unique words are reported. This method substantially outperforms raw cosine retrieval, particularly for single-label targets.


---

## 3. Results and Discussion

### 3.1 Odor Prediction in Mixtures

**Blender independent test set performance.** On the rigorously decontaminated blender test set of 6,305 pairs (104 odor groups with computable AUROC), AtomFormer achieves macro-AUROC = 0.9347 and cosR@3 = 88.4%. This exceeds POM's reported macro-AUROC of 0.894 on single-molecule prediction [Lee et al., 2023] and Sisson et al.'s reported blender AUROC of 0.7627 [Sisson et al., 2025]—both of which required ensemble methods or task-specific architectures. AtomFormer achieves higher performance with a single 16.9M-parameter model trained on publicly available data without expert annotation. We further examine the model's performance across individual odor groups and its behavior under label ambiguity.

**Table 1: Per-group AUROC by training frequency**

| Training frequency | Groups | Mean AUROC | Examples |
|:-----------------:|:-----:|:----------:|:---------|
| Rare (<100 pairs) | 13 | 0.8627 | cucumber, pine, alcoholic, cooling |
| Mid (100–1K pairs) | 41 | 0.9344 | tobacco, smoky, mushroom, pungent |
| Common (>1K pairs) | 50 | 0.9538 | floral, fruity, green, woody |
| **Overall** | **104** | **0.9347** | |

Breaking down the macro-AUROC by odor group reveals a clear relationship between training frequency and prediction accuracy. The full per-group AUROC table is provided in the Supporting Material.

**Label ambiguity robustness.** The TGSC Blender data presents a previously underappreciated characteristic: approximately 73% of reciprocal pairs (A+B and B+A) carry different odor labels. For example, menthol + bornyl acetate → "woody" when listed as menthol + bornyl acetate, but → "herbal" when listed as bornyl acetate + menthol. The cosine similarity between these conflicting label embeddings is 0.53, close to the random baseline of 0.49—they are genuinely different semantic targets, not transcription errors. In a classification paradigm, this creates a direct label conflict: the same input (A+B, which is graph-isomorphic to B+A since atom order is irrelevant) must predict two different classes.

AtomFormer handles this ambiguity naturally. Because the model has no positional encoding and treats all atoms as an unordered set, the graphs for A+B and B+A are identical, producing identical outputs. Under MSE training, when the same input is paired with two different target embeddings, the optimal solution in a continuous space is nearly the semantic midpoint of the two targets—the point that minimizes the expected MSE to both. We verified this on 1,000 conflicting AB/BA pairs:

**Table 2: Label ambiguity verification — model output z relative to conflicting labels**

| Metric | z → midpoint | z → label A | z → label B | label A ↔ B |
|:------|:---------:|:----------:|:----------:|:----------:|
| Mean cosine | **0.9424** | 0.8295 | 0.8240 | 0.5398 |
| Median cosine | **0.9617** | 0.8486 | 0.8389 | 0.5383 |

The model output z has mean cosine 0.9424 to the midpoint of the two conflicting labels, compared to 0.8295 and 0.8240 to either endpoint. The model output z is closer to the midpoint than to either endpoint in 80.6% of cases, confirming that the embedding consistently falls between the two conflicting labels. The remaining 19.4% likely reflects cases where the model's knowledge learned from other training examples—particularly molecules with similar structural features—pulls the prediction away from the exact midpoint, a reasonable outcome given that blender odor group labels are single coarse descriptors that may not capture the full perceptual nuance of each mixture. Combined with the smooth interpolation property of the high-dimensional space (Section 2.1), the nearest-neighbor retrieval naturally returns both original labels as the top-2 predictions, providing informative cues about the underlying label ambiguity rather than a single arbitrary choice.


### 3.2 Architectural Roles

To isolate the contributions of the GIN layers and the Transformer encoder, we compare the full model against two ablated variants with all other model components unchanged.

**Table 3: Ablation study — architectural component contributions**

| Model | BL AUROC | TGSC conR@3 | Params |
|:----|:--------:|:-----------:|:-----:|
| **Full** (GIN+TFM→BGE) | **0.9347** | **50.4%** | 16.9M |
| GIN-only (no Transformer) | 0.8185 | 44.9% | ~93K |
| TFM-only (no GIN) | — | — | ~16.8M |

Removing the Transformer drops blender AUROC from 0.9347 to 0.8185, while removing GIN prevents convergence entirely (22.3% TGSC conR@1 for all 30 epochs). The GIN-only result is consistent with Sisson et al.'s reported GIN AUROC of 0.852 (74 labels). The gap of Blender testset to the full model is therefore attributable to the Transformer encoder. Notably, the GIN-only model’s monomer conR@3 is close to the full model’s (44.9% vs. 50.4%), confirming that GIN alone captures intra-molecular chemical features effectively. The consensus recall metrics may appear modest overall, but this is because the regression objective (MSE→BGE) preserves fine-grained odor information in a 1024-dimensional continuous space, making the retrieval task considerably more challenging than traditional classification over a fixed set of 100–200 labels. Below, we adopt a classification objective to compare our architecture with a pure GIN classification baseline.

### 3.3 Regression vs. classification paradigm

Classification has been the standard approach in odor analysis. Here we compare the classification and regression paradigms in detail.

**Table 4: Regression vs. classification paradigm comparison**

| Model | Objective | Training data | BL AUROC | GS monomer AUROC |
|:----|:---------:|:------------:|:--------:|:---------------:|
| **Full** (GIN+TFM→BGE) | MSE→BGE | TGSC+BL | **0.9347** | — |
| GIN-only (classification) | BCE (135) | GS only | — | **0.8274** |
| GIN+TFM+cls | BCE (135) | GS only | — | 0.7361 |
| GIN+TFM+cls | BCE (135) | GS+BL | 0.7847 | 0.8175 |

Comparing rows 2–4 of the table reveals the Transformer's data requirement. Under a classification setup (BCE, 135 labels), the GIN+TFM architecture trained on GS monomers alone (row 3) achieves only 0.7361 on the GS test set, far below the pure GIN classifier (row 2, 0.8274). This gap reflects the Transformer's 17-fold parameter increase (16.9M vs. 0.99M). When augmented with 557K blender pairs (row 4), the same architecture reaches 0.8175 on GS monomers, matching the GIN classifier. This confirms that AtomFormer captures monomer-level information with sufficient data.

Comparing rows 1 and 4 highlights the impact of label ambiguity on mixture prediction. Both models use the same GIN+TFM backbone and training data (TGSC+BL), but with different objectives. The regression-based model (row 1) achieves 0.9347 on the blender test, while the classification model (row 4) reaches only 0.7847. The 0.15 gap arises because approximately 73% of reciprocal A+B/B+A blender pairs carry different labels, forcing the classification head to average two conflicting targets. The regression model avoids this by operating in a continuous embedding space, where the optimal solution is the semantic midpoint—not a compromise but a meaningful interpolation (Section 3.1).

**Robustness to training conditions.** Beyond the deliberate architectural choices, we assessed the model's sensitivity to stochastic variations and training hyperparameters. On the monomer-only setting (GS nosweet, 3,286 compounds), we conducted 50 independent training runs with different random seeds (42–91). The GS test R@1 had a mean of 28.4% and median of 28.8%, with the majority of runs falling within a narrow 28–31% range. For the full model with blender data, we tested 3 random seeds and observed no significant difference in blender AUROC. The only hyperparameter that materially affects performance is the learning rate warmup: training without the 3-epoch linear warmup leads to an approximately 5-percentage-point drop in final R@1. Once warmup is applied, the model converges reliably across learning rates (1 × 10⁻³ to 5 × 10⁻⁴) and batch sizes (512–1024).




## 4. Conclusion

Through a GIN+Transformer architecture trained with a regression objective in the BGE-M3 embedding space, we achieve macro-AUROC = 0.9347 on mixture prediction, exceeding prior methods by a considerable margin. Ablation studies confirm that GIN layers extract local chemical features while the Transformer enables cross-molecule interactions, each playing a distinct and complementary role. Comparisons between regression and classification paradigms show that our approach matches pure GIN classifiers on monomer tasks while avoiding the label conflict problems that limit classification-based mixture models. 

In future work, we plan to extend the model to handle concentration-weighted mixtures, enabling prediction for arbitrary component ratios—a critical step toward practical perfumery and flavor development. More broadly, the GIN+Transformer architecture validated here demonstrates its effectiveness for multi-mixture property prediction, and we believe it can be adapted to other domains where understanding interactions between multiple molecules is essential.

---

## Data Availability

All training data (TGSC monomer and blender data) is publicly available through The Good Scents Company website. The BGE-M3 embedding model is publicly available through the Hugging Face model hub. Processed data files, model weights, and evaluation scripts are available at [repository link].

## Code Availability

The AtomFormer source code, training scripts, and evaluation pipelines are available at [repository link]. The model architecture and training procedure are described in sufficient detail in Section 2.2 for independent reproduction.

## Author Contributions

To be completed.

## Competing Interests

The authors declare no competing interests.

## Acknowledgments

To be completed.

---

## References

1. Chen, J., Xiao, S., Zhang, P., et al. (2024). BGE-M3: Multi-lingual multi-granularity embedding for dense retrieval. *arXiv preprint*.
2. Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. *ICLR 2021*.
3. Tom, G., Ser, C. T., Rajaonson, E. M., et al. (2025). From molecules to mixtures: Learning representations of olfactory mixture similarity using inductive biases. *arXiv preprint*, 2501.16271.
4. Keller, A., Gerkin, R. C., Guan, Y., et al. (2017). Predicting human olfactory perception from chemical features of odorants. *Science*, 355(6327), 820-826.
5. Kurfali, M., et al. (2025). Representations of smells: The next frontier for language models? *Cognition*, 264, 106243.
6. Lee, B. K., Mayhew, E. J., Sanchez-Lengeling, B., et al. (2023). A principal odor map unifies diverse tasks in olfactory perception. *Science*, 381(6661), 999-1006.
7. Leenhouts, R. J., Larsson, T., Verhelst, S., & Vermeire, F. H. (2025). Property prediction of fuel mixtures using pooled graph neural networks. *Fuel*, 381, 133214.
8. Leffingwell, J. C. (2001). PMP 2001 database of odor descriptions. *Leffingwell & Associates*.
9. Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic gradient descent with warm restarts. *ICLR 2017*.
10. Magnasco, M. O., Keller, A., & Vosshall, L. B. (2015). On the dimensionality of olfactory space. *bioRxiv*, 022103.
11. Omatu, S., Araki, H., & Fujinaka, T. (2012). Intelligent classification of odor data using neural networks. *Advances in Computational Intelligence*.
12. Partin, A., et al. (2026). Benchmarking community drug response prediction. *Briefings in Bioinformatics*, 27(1), bbaf667.
13. Poivet, E., Tahirova, N., Peterlin, Z., et al. (2018). Functional odor classification through a medicinal chemistry approach. *Science Advances*, 4(10), eaao6086.
14. Sanchez-Lengeling, B., et al. (2019). Machine learning for scent: Learning generalizable perceptual representations of small molecules. *arXiv preprint*, 1910.10685.
15. Sisson, B., et al. (2025). Deep learning for odor prediction on aroma-chemical blends. *ACS Omega*, 10(9), 8980-8992.
16. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. *NeurIPS 2017*.
17. Vinyals, O., Bengio, S., & Kudlur, M. (2016). Order matters: Sequence to sequence for sets. *ICLR 2016*.
18. Wang, Y., et al. (2026). BiScale-GTR: Fragment-aware graph transformers for multi-scale molecular representation learning. *arXiv preprint*, 2604.06336.
19. Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How powerful are graph neural networks? *ICLR 2019*.
20. Yan, Z., et al. (2025). HIGHT: Hierarchical graph tokenization for graph-language LLM. *ICML 2025*.
21. Zhang, H., Lai, T., Chen, J., et al. (2024). Learning molecular mixture property using chemistry-aware graph neural network. *PRX Energy*, 3, 023006.
22. Zhu, J., et al. (2025). FragmentNet: Adaptive graph fragmentation for graph-to-sequence molecular representation learning. *arXiv preprint*, 2502.01184.