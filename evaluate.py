import os
import csv
from langchain_community.document_loaders import TextLoader, DirectoryLoader, CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA

# ── 20 Evaluation Questions + Reference Answers ──────────────────────────────
EVAL_QUESTIONS = [
    {
        "question": "When were the first modern Olympic Games held and where?",
        "reference": "The first modern Olympic Games were held in Athens, Greece in 1896."
    },
    {
        "question": "Who founded the International Olympic Committee?",
        "reference": "Baron Pierre de Coubertin founded the International Olympic Committee in 1894."
    },
    {
        "question": "How often are the Olympic Games held?",
        "reference": "The Olympic Games are held every four years."
    },
    {
        "question": "What is the difference between Summer and Winter Olympics scheduling since 1994?",
        "reference": "Since 1994 the Summer and Winter Olympics alternate every two years."
    },
    {
        "question": "When did the Ancient Olympic Games begin?",
        "reference": "The most widely accepted date is 776 BC."
    },
    {
        "question": "Where were the Ancient Olympic Games held?",
        "reference": "They were held at the sanctuary of Zeus in Olympia, Greece."
    },
    {
        "question": "When did the Ancient Olympic Games officially end?",
        "reference": "They ended around 393 AD as Roman influence grew."
    },
    {
        "question": "What sports were featured in the Ancient Olympics?",
        "reference": "Wrestling, horse racing, and chariot racing were among the featured sports."
    },
    {
        "question": "What is the Youth Olympic Games age range?",
        "reference": "The Youth Olympic Games are for athletes aged 14 to 18."
    },
    {
        "question": "How many athletes competed at the 2020 Summer and 2022 Winter Olympics combined?",
        "reference": "Over 14,000 athletes competed across 448 events."
    },
    {
        "question": "What does IOC stand for and what is its role?",
        "reference": "IOC stands for International Olympic Committee. It governs the Olympic Movement and chooses the host city."
    },
    {
        "question": "What are NOCs in the context of the Olympics?",
        "reference": "NOCs are National Olympic Committees, which are part of the Olympic Movement."
    },
    {
        "question": "What is the Paralympic Games?",
        "reference": "The Paralympic Games are Olympic Games for athletes with disabilities."
    },
    {
        "question": "How many teams participate in the Olympic Games?",
        "reference": "More than 200 teams participate, each representing a sovereign state or territory."
    },
    {
        "question": "What was the sacred truce in the Ancient Olympics?",
        "reference": "A sacred truce allowed pilgrims to travel safely to the Olympic festival."
    },
    {
        "question": "What country has won the most gold medals historically?",
        "reference": "The United States has historically won the most gold medals in the Olympics."
    },
    {
        "question": "What does the Olympic Charter define?",
        "reference": "The Olympic Charter defines the structure and authority of the Olympic Movement."
    },
    {
        "question": "What shift happened regarding athlete eligibility in the modern Olympics?",
        "reference": "The Olympics shifted from pure amateurism to accepting professional athletes."
    },
    {
        "question": "What is the Deaflympics?",
        "reference": "The Deaflympics is an Olympic event endorsed by the IOC for deaf athletes."
    },
    {
        "question": "What role does mass media play in the modern Olympics?",
        "reference": "Mass media and corporate sponsorship have grown in importance for the modern Olympic Movement."
    },

    # ── Cross-document synthesis questions (require both txt + csv) ──
    {
        "question": "Which sports that appeared in the Ancient Olympics are also represented in the modern Olympic athlete dataset?",
        "reference": "Wrestling appeared in the Ancient Olympics and is also present in the modern athlete dataset. The Ancient Olympics also featured horse racing and chariot racing, while the modern dataset includes a wide variety of athletic and combat sports."
    },
    {
        "question": "The Olympic Movement emphasizes participation from sovereign states. Which countries appear most frequently in the athlete dataset and how does this reflect the global reach described in the Olympic overview?",
        "reference": "The Olympic Movement involves over 200 teams from sovereign states. The athlete dataset shows participation from countries like the United States, Soviet Union, and Germany, reflecting the global nature of the Games described in the Olympic overview."
    },
    {
        "question": "Based on the Olympic overview and the athlete dataset, how has the scale of Olympic participation changed over time?",
        "reference": "The Olympic overview states over 14000 athletes competed in 2020 and 2022 combined. The athlete dataset spanning historical Games shows participation has grown significantly from early Games, reflecting the expansion of the Olympic Movement over time."
    },
]

# ── RAG System Setup ──────────────────────────────────────────────────────────
class OlympicRAGSystem:
    def __init__(self, data_dir="data/", model_name="qwen2.5:1.5b"):
        self.data_dir = data_dir
        self.model_name = model_name
        self.vector_db_path = "vector_db"
        self.llm = OllamaLLM(model=self.model_name)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.qa_chain = None

    def initialize_system(self):
        if os.path.exists(self.vector_db_path) and os.listdir(self.vector_db_path):
            print("Loading existing knowledge base...")
            vector_db = Chroma(
                persist_directory=self.vector_db_path,
                embedding_function=self.embeddings
            )
        else:
            print("Building knowledge base from documents...")
            all_documents = []

            txt_loader = DirectoryLoader(
                self.data_dir,
                glob="*_clean.txt",
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'}
            )
            all_documents.extend(txt_loader.load())

            csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('_clean.csv')]
            for csv_file in csv_files:
                loader = CSVLoader(
                    file_path=os.path.join(self.data_dir, csv_file),
                    encoding="utf-8"
                )
                all_documents.extend(loader.load())

            splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
            chunks = splitter.split_documents(all_documents)

            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.vector_db_path
            )
            print("Knowledge base built and saved.")

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

    def ask_question(self, query):
        if not self.qa_chain:
            return "System not initialized.", []
        response = self.qa_chain.invoke({"query": query})
        return response["result"], response["source_documents"]


# ── LLM-as-Judge Evaluator ────────────────────────────────────────────────────
# Separate larger model used only for judging — more reliable than 0.5b
judge_llm = OllamaLLM(model="qwen2.5:1.5b", temperature=0)

def llm_judge(question, reference_answer, system_answer):
    """Use a stronger LLM to score the system answer vs the reference answer."""
    prompt = f"""You are a strict evaluation judge for a question answering system.

Your job is to compare the SYSTEM ANSWER to the REFERENCE ANSWER and assign a score.

QUESTION: {question}

REFERENCE ANSWER: {reference_answer}

SYSTEM ANSWER: {system_answer}

SCORING RULES:
- Score 1   → The system answer is factually correct and contains the key information from the reference answer. Minor extra details are acceptable.
- Score 0.5 → The system answer is partially correct. It contains some correct facts but is missing important details or includes minor inaccuracies.
- Score 0   → The system answer is wrong, hallucinated, off-topic, or says it does not know.

IMPORTANT:
- Ignore formatting differences. Focus only on factual correctness.
- If the system answer contains the correct core fact, score it 1 even if it has extra sentences.
- If the system answer contradicts the reference or makes up facts, score it 0.
- Do NOT be influenced by answer length.

YOU MUST REPLY WITH ONLY ONE OF THESE THREE VALUES: 1, 0.5, or 0
Do not write anything else. No explanation. No punctuation. Just the number."""

    score_str = judge_llm.invoke(prompt).strip()

    # Strict parsing — check exact matches first
    if score_str == "1":
        return 1.0
    if score_str == "0.5":
        return 0.5
    if score_str == "0":
        return 0.0

    # Fallback: search for value in response
    if "0.5" in score_str:
        return 0.5
    if "1" in score_str:
        return 1.0
    if "0" in score_str:
        return 0.0

    print(f"  [WARNING] Could not parse judge response: '{score_str}' — defaulting to 0")
    return 0.0


# ── Main Evaluation Pipeline ──────────────────────────────────────────────────
def run_evaluation():
    print("\n=== Olympic RAG Evaluation Pipeline ===\n")

    rag = OlympicRAGSystem()
    rag.initialize_system()

    results = []
    total_score = 0.0

    for i, item in enumerate(EVAL_QUESTIONS):
        question = item["question"]
        reference = item["reference"]

        print(f"[{i+1}/20] {question}")

        # Get RAG answer and source docs
        answer, docs = rag.ask_question(question)

        # Extract top 3 source file names
        sources = []
        relevance = []
        for doc in docs[:3]:
            src = os.path.basename(doc.metadata.get("source", "Unknown"))
            sources.append(src)
            # Mark relevant if source is not a data dictionary
            relevance.append("Relevant" if "dictionary" not in src.lower() else "Not Relevant")

        # Pad to 3 if fewer docs returned
        while len(sources) < 3:
            sources.append("N/A")
            relevance.append("N/A")

        # LLM judges the answer (uses separate 5b model)
        score = llm_judge(question, reference, answer)
        total_score += score

        print(f"  Score: {score} | Sources: {sources[0]}, {sources[1]}, {sources[2]}")

        results.append({
            "Q#": i + 1,
            "Question": question,
            "Reference Answer": reference,
            "System Answer": answer,
            "Score (0/0.5/1)": score,
            "Source 1": sources[0],
            "Source 1 Relevant": relevance[0],
            "Source 2": sources[1],
            "Source 2 Relevant": relevance[1],
            "Source 3": sources[2],
            "Source 3 Relevant": relevance[2],
        })

    average_accuracy = total_score / len(EVAL_QUESTIONS)
    print(f"\n=== Evaluation Complete ===")
    print(f"Total Score: {total_score} / {len(EVAL_QUESTIONS)}")
    print(f"Average Accuracy: {average_accuracy:.2%}")

    # Save results to CSV
    output_file = "evaluation_results.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        writer.writerow({})  # blank row
        writer.writerow({"Q#": "AVERAGE ACCURACY", "Question": f"{average_accuracy:.2%}"})

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    run_evaluation()