# Lecture 10: Creating Custom Instruction Datasets
## "0 to LLM" Workshop - Building the Foundation for Specialized Models

---

## Slide 1: Introduction & Context (3 minutes)

### Why Instruction Datasets Matter
- **The Data Bottleneck**: Quality instruction datasets are the limiting factor in fine-tuning
- **Connection to Lectures 7-8**: We learned HOW to fine-tune, now we learn WHAT to fine-tune on
- **Quality vs Quantity**: 1000 high-quality examples > 10,000 poor examples
- **Real-world Impact**: Determines whether your model follows instructions or hallucinates

### The Challenge
```
Base Model + Poor Dataset = Confused Model
Base Model + Quality Dataset = Specialized Assistant
```

---

## Slide 2: Four Methods Overview (2 minutes)

| Method | Flexibility | Ease | Cost | Accuracy | Time | Examples Needed |
|--------|-------------|------|------|----------|------|-----------------|
| **1. Open Source Datasets** | Low | High | Low | Medium | Low | 1,000-5,000 |
| **2. Manual Expert Creation** | High | Low | High | High | High | 500-2,000 |
| **3. Template-Based** | Medium | Medium | Medium | Medium | Medium | 2,000-10,000 |
| **4. LLM-Generated (Alpaca)** | High | Medium | Low | Medium | Low | 5,000-50,000 |

---

## Slide 3: Method 1 - Open Source Datasets (5 minutes)

### Process
1. **Search** existing datasets (Hugging Face, Papers with Code)
2. **Filter** by domain and task type
3. **Adapt** format to your needs
4. **Validate** quality and relevance

### Examples
- **Alpaca**: General instruction following
- **FLAN**: Multi-task instruction following
- **Code Alpaca**: Programming tasks
- **Medical QA**: Healthcare domain

### Pros & Cons
| ✅ Pros | ❌ Cons |
|---------|---------|
| • Immediate availability | • Limited domain coverage |
| • Proven quality | • May not match exact use case |
| • Free/cheap | • Requires adaptation |
| • Large datasets available | • Generic responses |

### When to Use
- **Best for**: General tasks, prototyping, learning
- **Avoid for**: Highly specialized domains, unique requirements

---

## Slide 4: Method 2 - Manual Expert Creation (5 minutes)

### Process
1. **Define** task requirements and scope
2. **Recruit** domain experts
3. **Create** instruction-response pairs
4. **Review** and validate quality
5. **Iterate** based on feedback

### Quality Control
- **Expert Guidelines**: Clear instructions for creators
- **Review Process**: Multiple experts validate each example
- **Consistency Checks**: Ensure uniform style and format
- **Pilot Testing**: Test on small subset before scaling

### Pros & Cons
| ✅ Pros | ❌ Cons |
|---------|---------|
| • Highest quality | • Very expensive |
| • Domain-specific | • Time-consuming |
| • Customized to needs | • Requires expert knowledge |
| • Proven effectiveness | • Difficult to scale |

### When to Use
- **Best for**: Critical applications, specialized domains
- **Examples**: Medical diagnosis, legal advice, technical support

---

## Slide 5: Method 3 - Template-Based Creation (5 minutes)

### Process
1. **Design** templates for your domain
2. **Create** parameterized templates
3. **Generate** variations systematically
4. **Validate** generated examples
5. **Refine** templates based on quality

### Template Example
```
Instruction: Analyze the sentiment of this {business_type} review.
Input: {review_text}
Output: This review expresses {sentiment} sentiment because {reasoning}.
```

### Pros & Cons
| ✅ Pros | ❌ Cons |
|---------|---------|
| • Scalable | • Limited creativity |
| • Consistent format | • May be repetitive |
| • Cost-effective | • Requires template design |
| • Good for structured tasks | • Less natural language |

### When to Use
- **Best for**: Structured tasks, QA systems, classification
- **Examples**: Customer service, data analysis, reporting

---

## Slide 6: Method 4 - LLM-Generated (Alpaca Method) (5 minutes)
Alpaca is most well known IFT dataset, with just 50k instructions. 
### Process
1. **Create** seed examples (8-10 high-quality pairs)
2. **Design** generation prompts
3. **Use LLM** to generate variations
4. **Filter** and validate outputs
5. **Iterate** with improved prompts

### Alpaca Method Details
```python
# Seed example format
{
    "instruction": "Write a poem about AI",
    "input": "",
    "output": "In circuits deep and code so bright..."
}

# Generation prompt
"Given these examples, generate 10 more instruction-response pairs..."
```

### Pros & Cons
| ✅ Pros | ❌ Cons |
|---------|---------|
| • Highly scalable | • Quality varies |
| • Cost-effective | • May inherit biases |
| • Creative variations | • Requires good seeds |
| • Fast generation | • Needs validation |

### When to Use
- **Best for**: Large datasets, creative tasks, rapid prototyping
- **Examples**: Creative writing, general Q&A, brainstorming

---

## Slide 7: Evaluation Framework for CEOs & CTOs (3 minutes)

### Decision Matrix
| Factor | Weight | Open Source | Manual | Template | LLM-Generated |
|--------|--------|-------------|--------|----------|---------------|
| **Flexibility** (10%) | 10 | 2 | 10 | 6 | 8 |
| **Ease** (20%) | 20 | 10 | 2 | 6 | 7 |
| **Cost** (25%) | 25 | 10 | 2 | 6 | 8 |
| **Accuracy** (30%) | 30 | 5 | 10 | 6 | 6 |
| **Time** (15%) | 15 | 10 | 2 | 6 | 8 |
| **Total Score** | 100 | **6.8** | **4.4** | **6.0** | **7.1** |

### Recommendations by Use Case
- **MVP/Prototype**: Open Source (6.8)
- **Production System**: Manual + Template Hybrid
- **Large Scale**: LLM-Generated (7.1)
- **Critical Domain**: Manual (4.4)

---

## Slide 8: Best Practices & Recommendations (2 minutes)

### Dataset Size Guidelines
| Use Case | Minimum Examples | Recommended | Maximum |
|----------|------------------|-------------|---------|
| **Simple Tasks** | 500 | 1,000-2,000 | 5,000 |
| **Complex Tasks** | 1,000 | 2,000-5,000 | 10,000 |
| **Specialized Domain** | 2,000 | 5,000-10,000 | 20,000 |
| **Multi-task** | 5,000 | 10,000-20,000 | 50,000 |

### Quality Metrics
- **Relevance**: Does response address instruction?
- **Accuracy**: Is information correct?
- **Completeness**: Does response fully answer?
- **Consistency**: Are similar instructions handled similarly?

### Common Pitfalls
- ❌ **Too generic**: Vague instructions produce poor responses
- ❌ **Inconsistent format**: Mixed formats confuse the model
- ❌ **Poor validation**: Skipping quality checks
- ❌ **Overfitting**: Too similar examples

---

