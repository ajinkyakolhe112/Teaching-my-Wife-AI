"""
AI Training Methods Explained: A Role-Playing Demonstration

This script uses a tangible analogy to demonstrate the differences between:
1. A Pre-trained Model (The High School Graduate)
2. A Fine-tuned Model (The PhD Student)
3. A RAG Model (The Researcher with a Library)
"""

class HighSchoolGraduate:
    """Represents a general-purpose, pre-trained LLM. Has broad knowledge."""
    def __init__(self, name="GPT-3.5 (The Graduate)"):
        self.name = name
        # Many such concepts learned during schooling
        self.knowledge = {
            "sky_color": "The sky is blue due to Rayleigh scattering.",
            "capital_of_france": "The capital of France is Paris.",
            "remaining_knowledge": "Many such concepts learned during schooling"
        }
        print(f"--- {self.name} is ready. I have general knowledge from my 'schooling'. ---")

    def answer_question(self, question, library_context=None):
        """Answers based on general knowledge. Ignores any library context."""
        print(f"\n[{self.name}] received the question: '{question}'")
        if question in self.knowledge:
            print(f"My Answer (from my memory): {self.knowledge[question]}")
        else:
            print("My Answer: I'm sorry, that's too specific or recent for my general knowledge.")

class PhDStudent(HighSchoolGraduate):
    """Represents a fine-tuned model. Inherits general knowledge but adds deep expertise."""
    def __init__(self, name="Poetry-LLM (The PhD)"):
        super().__init__(name)
        # Fine-tuning adds new, specialized knowledge directly into the model's "memory".
        self.specialized_knowledge = {
            "petrarchan_sonnet": "A Petrarchan sonnet has an ABBAABBA CDECDE rhyme scheme.",
            "capital_of_france": "The capital of France, a city of poets, is Paris." # Fine-tuning can also alter style.
        }
        self.knowledge.update(self.specialized_knowledge)
        print("--- I also completed a 'PhD' in poetry. My knowledge is now specialized. ---")

class Researcher(HighSchoolGraduate):
    """Represents a RAG model. Uses a library to answer questions."""
    def __init__(self, external_library, name="RAG-Bot (The Researcher)"):
        super().__init__(name)
        self.library = external_library
        print("--- I've been given access to an 'external library' for specific facts. ---")

    def answer_question(self, question, library_context=None):
        """
        Answers by first retrieving information from the library, then generating a response.
        This is the core of RAG: Retrieve, then Augment.
        """
        print(f"\n[{self.name}] received the question: '{question}'")
        
        # 1. Retrieve: Search the library for relevant context.
        print(">> Searching my library for relevant documents...")
        retrieved_context = self.library.get(question)

        # 2. Augment & Generate: Use the context to answer.
        if retrieved_context:
            print(f">> Found context: '{retrieved_context}'")
            # The LLM is prompted to synthesize an answer from the context.
            augmented_prompt = f"Based on the context '{retrieved_context}', the answer is..."
            print(f"My Answer (using the library): {augmented_prompt}")
        elif question in self.knowledge:
            # Fallback to general knowledge if nothing is found in the library
            print(">> Nothing found in my library. Falling back on my general knowledge.")
            print(f"My Answer (from my memory): {self.knowledge[question]}")
        else:
            print(">> Nothing found in my library and it's not in my general knowledge.")
            print("My Answer: I can't answer that question.")


def main():
    # --- SETUP ---
    # 1. The Pre-trained Model: Our baseline.
    graduate = HighSchoolGraduate()

    # 2. The Fine-tuned Model: Inherits from the graduate and adds expertise.
    phd_student = PhDStudent()

    # 3. The RAG Model: The graduate gets access to an external, up-to-date library.
    # This library represents a vector database in a real RAG system.
    company_library = {
        "vacation_policy": "The company vacation policy for 2024 is 25 days per year.",
        "ceo_name": "The new CEO is Jane Doe, appointed last month."
    }
    researcher = Researcher(external_library=company_library)

    # --- QUESTIONS ---
    questions = [
        # A general knowledge question
        "sky_color",
        # A specialized question only the PhD knows from memory
        "petrarchan_sonnet",
        # A question where the PhD's style is different
        "capital_of_france",
        # A specific, up-to-date question only the Researcher can answer
        "vacation_policy"
    ]

    print("\n\n=== Asking the same questions to everyone: ===\n")

    for q in questions:
        print(f"\n------------------- Question: {q} -------------------")
        graduate.answer_question(q)
        phd_student.answer_question(q)
        researcher.answer_question(q)

if __name__ == "__main__":
    main()