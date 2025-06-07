"""
AI Training Methods Explained: A Role-Playing Demonstration

This script uses a tangible analogy to demonstrate the differences between:
1. A Pre-trained Model (The High School Graduate)
2. A Fine-tuned Model (The PhD Student)
3. A RAG Model (The Researcher with a Library)

This enhanced version combines descriptive text with a live demonstration
to create a comprehensive and interactive teaching tool.
"""

def print_header(title):
    """Helper function to print a clean header."""
    print("\n" + "="*60)
    print(f"  {title.upper()}")
    print("="*60)


class HighSchoolGraduate:
    """Represents a general-purpose, pre-trained LLM. This is our foundation."""
    def __init__(self, name="GPT-3.5 (The Graduate)"):
        self.name = name
        self.time_investment = "Massive (like 18 years of schooling)"
        self.cost = "Very High"
        
        # This is the "general knowledge" learned during the expensive pre-training phase.
        self.knowledge = {
            "sky_color": "The sky is blue due to Rayleigh scattering.",
            "capital_of_france": "The capital of France is Paris.",
            "water_boiling_point": "Water boils at 100 degrees Celsius at sea level."
        }
        
        print_header(f"Model Ready: {self.name} - The Pre-trained Model")
        print(f"Represents: A foundational LLM.")
        print(f"Time/Cost to create: {self.time_investment} / {self.cost}.")
        print("This 'schooling' process is called PRE-TRAINING.")
        print("I have broad, general knowledge about the world but nothing specific or recent.")

    def answer_question(self, question):
        """Answers based on its 'internalized' general knowledge."""
        print(f"\n[{self.name}] received the question: '{question}'")
        if question in self.knowledge:
            print(f"  My Answer (from my memory): {self.knowledge[question]}")
        else:
            print("  My Answer: I'm sorry, I don't have that specific information in my memory.")


class PhDStudent(HighSchoolGraduate):
    """Represents a fine-tuned model. Inherits general knowledge but adds deep, 'baked-in' expertise."""
    def __init__(self, name="Poetry-LLM (The PhD)"):
        # It starts as a HighSchoolGraduate, then specializes.
        super().__init__(name)
        
        self.time_investment = "Relatively Short (like a 2-year PhD program)"
        self.cost = "Lower than pre-training"

        # Fine-tuning "bakes in" new knowledge and, importantly, a new style.
        # It overwrites and adds to the original knowledge.
        specialized_knowledge = {
            "petrarchan_sonnet": "A Petrarchan sonnet has an ABBAABBA CDECDE rhyme scheme.",
            "capital_of_france": "Ah, Paris! The capital of France, a city of light and verse." # Style is altered.
        }
        self.knowledge.update(specialized_knowledge)
        
        print_header(f"Specialist Ready: {self.name} - The Fine-tuned Model")
        print(f"Builds upon: The High School Graduate.")
        print(f"Time/Cost to specialize: {self.time_investment} / {self.cost}.")
        print("This specialization process is called FINE-TUNING.")
        print("I've studied a specific topic (poetry) so deeply it's now part of my memory.")


class Researcher(HighSchoolGraduate):
    """Represents a RAG model. It's a generalist with access to an external library."""
    def __init__(self, external_library, name="RAG-Bot (The Researcher)"):
        # It also starts as a HighSchoolGraduate.
        super().__init__(name)
        
        self.library = external_library
        self.time_investment = "Medium (time to set up the library connection)"
        self.cost = "Moderate"
        self.requires_library = True

        print_header(f"Specialist Ready: {self.name} - The RAG Model")
        print(f"Builds upon: The High School Graduate.")
        print(f"Time/Cost to specialize: {self.time_investment} / {self.cost}.")
        print("This 'open-book' approach is called RETRIEVAL-AUGMENTED GENERATION (RAG).")
        print("I answer questions by looking up facts in my external library.")
        
    def answer_question(self, question):
        """
        Answers by first retrieving information, then generating a response.
        This is the core of RAG: Retrieve, then Augment.
        """
        print(f"\n[{self.name}] received the question: '{question}'")
        
        # 1. Retrieve: Search the library for relevant context.
        print("  >> Searching my library for relevant documents...")
        retrieved_context = self.library.get(question)

        # 2. Augment & Generate: Use the context to answer.
        if retrieved_context:
            print(f"  >> Found context: '{retrieved_context}'")
            # In a real system, the LLM synthesizes an answer. Here, we simulate that.
            print(f"  My Answer (based on the library): The library states, '{retrieved_context}'")
        elif question in self.knowledge:
            # Fallback to general knowledge if nothing is found in the library.
            print("  >> Nothing found in my library. Falling back on my general knowledge.")
            print(f"  My Answer (from my memory): {self.knowledge[question]}")
        else:
            print("  >> Nothing in my library or my general knowledge covers this.")
            print("  My Answer: I am unable to answer that question.")


def main():
    # --- PHASE 1: THE EXPENSIVE UPBRINGING (PRE-TRAINING) ---
    print_header("The Analogy: Pre-training is like raising a child")
    print("This first, most expensive step creates a generalist foundation model.")
    print("1. Birth to 5: Learning basic concepts and language.")
    print("2. 6 to 12: Primary education and fundamental knowledge.")
    print("3. 13 to 18: High school and critical thinking.")
    print("The result is our 'High School Graduate'.")

    # --- PHASE 2: CREATING THE SPECIALISTS ---
    # Our baseline Pre-trained model.
    graduate = HighSchoolGraduate()

    # The Fine-tuned model, which alters its internal knowledge.
    phd_student = PhDStudent()

    # The RAG model, which gets access to external, up-to-date information.
    # This dictionary simulates a real vector database.
    company_library = {
        "vacation_policy": "The company vacation policy for 2024 is 25 days per year.",
        "ceo_name": "The new CEO is Jane Doe, appointed last month."
    }
    researcher = Researcher(external_library=company_library)

    # --- PHASE 3: THE FINAL EXAM ---
    # We will ask the same questions to all three models to see how they differ.
    questions = [
        "sky_color",              # A general knowledge question everyone should know.
        "petrarchan_sonnet",      # A specialized question only the PhD knows from memory.
        "capital_of_france",      # A question where the PhD's style is different.
        "vacation_policy",        # A specific, private question only the Researcher can answer.
        "ceo_name"                # A recent, private question only the Researcher can answer.
    ]

    print_header("The Final Exam: Asking the same questions to everyone")

    for q in questions:
        print("\n" + "-"*25 + f" Question: {q} " + "-"*25)
        graduate.answer_question(q)
        phd_student.answer_question(q)
        researcher.answer_question(q)
        
    # --- PHASE 4: SUMMARY ---
    print_header("Key Takeaways")
    print("1. Pre-training (Graduate): Creates a broad, general-purpose model. Very expensive.")
    print("2. Fine-tuning (PhD): Modifies the model's internal memory to change its SKILL or STYLE.")
    print("3. RAG (Researcher): Connects the model to an external library to give it new, up-to-date KNOWLEDGE.")
    print("\nSimple Rule of Thumb:")
    print("- To change HOW the model behaves -> Fine-tune.")
    print("- To change WHAT the model knows -> RAG.")


if __name__ == "__main__":
    main()