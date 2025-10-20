from dataclasses import dataclass

@dataclass(frozen=True)
class JudgeFields:
    context: str
    question: str
    answer: str

def qa_user_message(question: str) -> dict:
    # Zero-shot, closed-book: only the question
    return {"role": "user", "content": question}

def judge_system_message() -> dict:
    return {
        "role": "system",
        "content": (
            "You are a strict binary evaluator. "
            "Given a question, an answer, and a supporting text, reply with exactly one word: YES or NO. "
            "Reply YES only if the supporting text clearly entails that the answer correctly answers the question. "
            "Otherwise reply NO. Do not add any explanation."
        ),
    }

def judge_user_message(fields: JudgeFields) -> dict:
    # Keep language alignment: fields.context and fields.question should be in the same language as the model's answer.
    content = (
        "SUPPORTING TEXT:\n"
        f"{fields.context}\n\n"
        "QUESTION:\n"
        f"{fields.question}\n\n"
        "ANSWER:\n"
        f"{fields.answer}\n\n"
        "Is the answer correct given the supporting text? Reply with YES or NO."
    )
    return {"role": "user", "content": content}
