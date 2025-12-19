## Towards More Economical Context-Augmented LLM Generation by Reusing Stored KV Cache

Hanchen Li ∗ (Student), Yuhan Liu ∗ (Student), Yihua Cheng (Student), Kuntai Du (Student), Junchen Jiang {lihanc2002, yuhanl, yihua98, kuntai, junchenj}@uchicago.edu

## The University of Chicago

## Abstract

Across large language model (LLM) applications, we observe an emerging trend for reusing KV caches to save the prefill delays of processing repeated input texts in different LLM inputs. This has led to a broad design space, including colocating stored KV caches with (or close to) GPUs to various KV cache compression. However, a key question remains unanswered: can these delay reductions also be economically favorable? Specifically, we ask whether a developer can use public cloud services to store precomputed KV caches and reuse them to save delay without incurring more costs in terms of compute, storage, and network. To answer this question, we propose an validated analytical model for the cloud cost (in compute, storage, and network) of storing and reusing KV caches based on various workload parameters, such as reuse frequency, generated text lengths, model sizes, etc. Preliminary results show that KV cache reusing is able to save both delay and cloud cost across a range of workloads with long context. And we call more efforts on building more economical context augmented LLM by KV cache reusing.

## 1 Introduction

Nowadays, a wide range of applications ( e.g., document-based question answering) often use large language models (LLMs) after prepending the LLM inputs with long contexts ( e.g., related documents) that may contain thousands of tokens or more. In this paper, we refer such long prefixes as contexts.

Prior work found that different user requests may need to use the same context [3]. Therefore, the computation results of the context (in the format of KV caches), once computed and cached in the GPU, can be reused across those jobs that use the same context [2,4]. For example, two requests, 'What was the highest source of revenue of Apple in the last year?' and 'Which supplier for Apple offers the lowest average price?', may need to use the same annual fiscal report as the context.

However, the design challenge of such KV cache reusing scheme, is where to store the pre-computed KV cache of the context ? One potential design is to store the KV cache in the GPU [2,4], which is feasible when there are only limited number of contexts that the user will access ( e.g., a dedicated server that answers the question about a specific software), but is less favorable when a wide range of context may be accessed, as the GPU will not have sufficient GPU memory to store the KV cache of all the contexts.

∗ Equal Contribution

Figure 1: Illustration of text recomputation and KV cache reusing.

![Image](CAG_vs_RAG_b272f234-referenced_artifacts/img_001.png)

![Image](CAG_vs_RAG_b272f234-referenced_artifacts/img_002.png)

An alternative solution is to store the KV cache of the context in a dedicated storage server and transfer the KV cache into GPU memory when new jobs requesting contexts arrive. Acknowledging that such an approach incurs extra transfer delay and storage cost, we argue that such an approach is favorable compared with text recomputation in terms of both end-to-end delay and economic cost. By building an analytical model for developers to compare service costs given their workload pattern and cloud service pricing policy, we show that reusing KV cache from storage is more economical than text recomputation. Moreover, we validate this result by simulation under various workloads. This proves that the endto-end delay and economic cost of serving LLM inference jobs with KV cache are consistently lower than without storing KV cache. We hope this poster sparks more discussions toward realizing the full potential of such systems.

## 2 Cost Modeling

Analytical Model: In order to compare the operation costs of text recomputation and KV cache reusing, we built an analytical model to measure their costs. For the first text recomputation pipeline, the inference serving cost ( i.e., GPU compute) for one request could be estimated by the GPU use time per request times the number of requests using the context within a time period. The breakdown cost can be expressed by the following equation:

<!-- formula-not-decoded -->

where CGPU is the cost of the GPU instance per hour, N is number of requests querying the same context, Lcontext is the number of context tokens, Lprompt is the token length of the actual prompt, and Loutput is the number of generated tokens. Tprefill and Tdecode are time (hour) spent on the computation job given specific token numbers.

Figure 2: The cost and end-to-end delay for Llama-7B by varying input lengths.

![Image](CAG_vs_RAG_b272f234-referenced_artifacts/img_003.png)

For the context reusing pipeline, we separate the cost into three parts: inference, context storage, and transmission cost:

## 3 Preliminary Results

<!-- formula-not-decoded -->

where C storage is the storage cost per hour × GB, T is the period of serving, S storage is the size of the KV cache, SLO is the service level objective for time to first token, and C storage is the cost per hour to maintain the transmission link that satisfies SLO for loading the KV cache.

Insights: By analyzing the model with real pricing data from AWS, we find that Reusing KV cache could be more economical as long as the context is reused more than once per hour. And that the storage cost is usually only a minimal portion of the total cost. Take Llama-7B model as an example. A context of length 10K-token requires a storage space of 5.2 GB. With Amazon Elastic Block Storage(EBS) that could be directly mounted onto instance, the highest tier storage with 4GB/S throughput, called io2, costs 0.125 per GB × month [1] Since our IO operations are not frequent, we do not need to pay much extra provisioned IOPS fee. Within the period of one hour LLM serving. The total storage and transmission cost spent this hour for this 10K-token context is $ 8 . 8 e -41 . On the other hand, the GPU inference cost for prefilling is around $6 e -3 for a single request 2 . This is already more than 7 times larger than the fixed cost of storage during this hour. And if more requests reuse this piece of context in this hour, this fixed cost in reusing KV cache will be negligible in total cost. We can simplify the ratio between C text and C KV:

C text C KV ≈ 1 + N -1 N T prefill ( L context ) T decode ( L output )+ T prefill ( L prompt ) . This demonstrates that when context is long while the prompt and output are short, such as in many article Q&amp;A tasks [3], the savings of loading KV cache from storage could be significant.

1 io2 storage cost per hour × size = 0 . 00017 ∗ 5 . 2 = 0 . 00088

2 Single V100 GPU cost per hour × 1 h 3600 s × Tprefill ( 10 K ) = $3 × 1 3600 × 7 = 0 . 0058

Evaluation setup: In the following experiment, we run Llama-7B model on an Amazon's EC2 p3.8xlarge instance (4 V100 GPUs). We apply HuggingFace's model parallelism to run inference on four GPUs. The dataset is TriviaQA [3], which contains 200 pieces of different contexts. We assume each context is used five times and vary two workload parameters, the input lengths and output lengths.

## Takeaways:

- When varying the input length from 1-10K, as shown in Figure 2(a), using KV cache reuse can significantly lower both the end-to-end delay (by 1.1 to 2.9 × ) and the cost of using cloud services (by about 1.3 to 3.6 × ). The benefits of KV cache reuse become more significant as the length of the input increases. For shorter inputs (like 1,000 units), the time saved by skipping the prefilling delay does not compromise the delay it takes to load the KV cache.
- In Figure 2(b) where the output lengths are varied from 1 to 100 tokens, the delay saving ranges from 1.6-3.5 × and the cost saving ranges from 1.7-4.5 × . We observe that the longer the output is, the less saving KV cache reusing can bring. The insight is that KV cache reusing is only able to save delay for the first token's generation, thus, the longer the output is, the more likely the saving will be amortized.

## References

- [1] Amazon EBS pricing. https://aws.amazon.com/ebs/ pricing/ .
- [2] Anonymous. Chunkattention: Efficient attention on KV cache with chunking sharing and batching, 2024.
- [3] Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension, 2017.
- [4] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles , 2023.